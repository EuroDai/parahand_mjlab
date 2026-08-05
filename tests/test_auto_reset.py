"""Tests for the auto_reset config flag."""

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_cfg(auto_reset: bool):
  cfg = cartpole_balance_env_cfg()
  cfg.episode_length_s = 0.5  # 10 steps at dt=0.05
  cfg.scene.num_envs = 4
  cfg.auto_reset = auto_reset
  return cfg


def _step_until_done_env(env):
  """Step with zero actions until at least one env is done. Return step outputs."""
  for _ in range(env.max_episode_length + 5):
    action = torch.zeros((env.num_envs, 1), device=env.device)
    result = env.step(action)
    terminated, truncated = result[2], result[3]
    if (terminated | truncated).any():
      return result
  pytest.fail("No env terminated within max_episode_length steps")


def test_auto_reset_true_resets_done_envs(device):
  """With auto_reset=True (default), done envs are reset during step."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=True), device=device)
  env.reset()
  _, _, terminated, truncated, _ = _step_until_done_env(env)
  done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)

  # Episode counter was reset to 0 for done envs.
  assert (env.episode_length_buf[done_ids] == 0).all()
  env.close()


def test_non_finite_reset_retries_only_affected_env(device, monkeypatch):
  cfg = _make_cfg(auto_reset=True)
  cfg.non_finite_reset_attempts = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()

  reset_env_ids = []
  original_reset_idx = env._reset_idx

  def tracked_reset_idx(env_ids, *, is_retry=False):
    reset_env_ids.append(env_ids.clone())
    original_reset_idx(env_ids, is_retry=is_retry)

  curriculum_compute = Mock(wraps=env.curriculum_manager.compute)
  monkeypatch.setattr(env, "_reset_idx", tracked_reset_idx)
  monkeypatch.setattr(env.curriculum_manager, "compute", curriculum_compute)

  checked_env_ids = torch.tensor([0, 1], device=env.device)
  env.sim.data.qpos[1, 0] = float("nan")
  env._recover_non_finite_reset_state(checked_env_ids)

  assert len(reset_env_ids) == 1
  assert reset_env_ids[0].cpu().tolist() == [1]
  curriculum_compute.assert_not_called()
  for mask in env.sim.nan_guard.detect_non_finite_fields(env.sim.data).values():
    assert not mask[1]
  env.close()


def test_auto_reset_false_preserves_terminal_state(device):
  """With auto_reset=False, done envs are NOT reset and obs is the terminal state."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env.reset()
  obs, _, terminated, truncated, _ = _step_until_done_env(env)
  done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)

  # Episode counter was NOT reset (still at max_episode_length).
  assert (env.episode_length_buf[done_ids] == env.max_episode_length).all()

  # The returned obs must reflect the current (post-decimation terminal) sim
  # state. Since no reset ran and the sim wasn't touched after step(), a fresh
  # observation_manager.compute() on the current sim state must match exactly.
  # This catches regressions where step() might return stale or post-reset obs.
  env.observation_manager._obs_buffer = None  # bypass cache
  fresh_obs = env.observation_manager.compute()
  for group in obs:
    returned = obs[group]
    current = fresh_obs[group]
    assert isinstance(returned, torch.Tensor) and isinstance(current, torch.Tensor)
    assert torch.equal(returned, current)
  env.close()


def test_auto_reset_false_explicit_reset_works(device):
  """After auto_reset=False, calling reset(env_ids=...) resets those envs."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env.reset()
  _, _, terminated, truncated, _ = _step_until_done_env(env)
  done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)

  # Manually reset done envs.
  env.reset(env_ids=done_ids)
  assert (env.episode_length_buf[done_ids] == 0).all()

  # Can continue stepping after manual reset.
  action = torch.zeros((env.num_envs, 1), device=env.device)
  obs, reward, _, _, _ = env.step(action)
  assert obs is not None
  assert reward is not None
  env.close()


def test_auto_reset_false_requires_manual_reset_before_next_step(device):
  """Raw env should reject another step until done envs are explicitly reset."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env.reset()
  _step_until_done_env(env)

  action = torch.zeros((env.num_envs, 1), device=env.device)
  with pytest.raises(RuntimeError, match="must be reset via reset"):
    env.step(action)

  env.close()


def _slice_obs(obs: dict, ids: torch.Tensor) -> dict[str, torch.Tensor]:
  """Return a new obs dict containing only the rows at ``ids`` (per group)."""
  return {k: v[ids] for k, v in obs.items() if isinstance(v, torch.Tensor)}


def test_auto_reset_false_user_loop_pattern(device):
  """Example: run your own training loop against an auto_reset=False env.

  The pattern is:
    1. After step(), derive done_ids from terminated | truncated.
    2. Slice obs[done_ids] to get the true terminal observation and use it for
      bootstrap / target computation.
    3. Call env.reset(env_ids=done_ids) to reset only the done envs.
    4. Continue stepping with the full batch.
  """
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  obs, _ = env.reset(seed=0)
  episode_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  last_terminal_obs: dict[str, torch.Tensor] | None = None

  action = torch.zeros((env.num_envs, 1), device=env.device)
  for _ in range((env.max_episode_length + 2) * 3):
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated | truncated
    if not done.any():
      continue

    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    last_terminal_obs = _slice_obs(obs, done_ids)  # feed this to your critic/replay

    episode_count[done_ids] += 1
    obs, _ = env.reset(env_ids=done_ids)
    if (episode_count >= 2).all():
      break

  assert (episode_count >= 2).all()
  assert last_terminal_obs is not None
  env.close()


def test_auto_reset_false_obs_differs_from_auto_reset_true(device):
  """Terminal obs (auto_reset=False) differs from post-reset obs (auto_reset=True)."""
  # Run with auto_reset=True, capture post-reset obs for done envs.
  env_on = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=True), device=device)
  env_on.reset(seed=42)
  obs_on, _, _, _, _ = _step_until_done_env(env_on)
  env_on.close()

  # Run with auto_reset=False with the same seed, capture terminal obs.
  env_off = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env_off.reset(seed=42)
  obs_off, _, _, _, _ = _step_until_done_env(env_off)
  env_off.close()

  # The observations should differ: one is post-reset, the other is terminal.
  for group in obs_on:
    on_val = obs_on[group]
    off_val = obs_off[group]
    assert isinstance(on_val, torch.Tensor) and isinstance(off_val, torch.Tensor)
    assert not torch.equal(on_val, off_val)
