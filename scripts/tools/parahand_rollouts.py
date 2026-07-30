"""Collect and replay ParaHand episodes grouped by termination reason.

Examples:

  uv run scripts/tools/parahand_rollouts.py collect \
    --checkpoint-file logs/rsl_rl/parahand_only_grasp_object/RUN/model_3000.pt

  uv run scripts/tools/parahand_rollouts.py list \
    --rollout-dir rollouts/parahand/RUN

  uv run scripts/tools/parahand_rollouts.py replay \
    --rollout-dir rollouts/parahand/RUN \
    --termination object_out_of_bounds
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import tyro
from tensordict import TensorDict

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import _apply_curriculum_stage_override
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.tasks.parahand_grasp.mdp.events import reset_primitive_object_pose
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

_DEFAULT_TASK = "Mjlab-Grasp-Object-ParaHand-Only"
_MANIFEST_NAME = "manifest.jsonl"


@dataclass(frozen=True)
class Collect:
  """Run a trained policy and save complete episodes without rendering."""

  checkpoint_file: str
  task: str = _DEFAULT_TASK
  output_dir: str | None = None
  num_envs: int = 256
  num_episodes: int = 1000
  max_steps: int | None = None
  curriculum_stage: int = 1
  device: str = "cuda:0"
  seed: int = 42


@dataclass(frozen=True)
class ListRollouts:
  """Print termination counts in a rollout directory."""

  rollout_dir: str


@dataclass(frozen=True)
class Replay:
  """Replay one saved episode in the interactive Viser viewer."""

  rollout_dir: str
  termination: str
  episode_index: int = 0
  curriculum_stage: int | None = None
  device: str = "cuda:0"


@dataclass(frozen=True)
class EpisodeRecord:
  episode_id: int
  path: str
  length: int
  return_value: float
  termination_reasons: list[str]
  shape_id: int
  size: list[float]


def _default_output_dir(checkpoint: Path) -> Path:
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  return Path("rollouts/parahand") / f"{checkpoint.stem}_{timestamp}"


def _append_manifest(path: Path, record: EpisodeRecord) -> None:
  with path.open("a", encoding="utf-8") as file:
    file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _read_manifest(rollout_dir: Path) -> list[EpisodeRecord]:
  manifest = rollout_dir / _MANIFEST_NAME
  if not manifest.exists():
    raise FileNotFoundError(f"Rollout manifest not found: {manifest}")
  records = []
  with manifest.open(encoding="utf-8") as file:
    for line in file:
      if line.strip():
        records.append(EpisodeRecord(**json.loads(line)))
  return records


def _termination_counts(records: list[EpisodeRecord]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for record in records:
    for reason in record.termination_reasons:
      counts[reason] = counts.get(reason, 0) + 1
  return counts


def _get_primitive_event(env: ManagerBasedRlEnv) -> reset_primitive_object_pose:
  event = env.event_manager.get_term_cfg("reset_object_pose").func
  if not isinstance(event, reset_primitive_object_pose):
    raise TypeError(
      "This tool requires the ParaHand analytic primitive reset event, "
      f"got {type(event).__name__}."
    )
  return event


def _get_lifting_command(env: ManagerBasedRlEnv) -> LiftingCommand:
  command = env.command_manager.get_term("object_pose")
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      "This tool requires a LiftingCommand named 'object_pose', "
      f"got {type(command).__name__}."
    )
  return command


def _load_policy(
  task: str,
  checkpoint: Path,
  env: RslRlVecEnvWrapper,
  device: str,
):
  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def collect(cfg: Collect) -> None:
  if cfg.num_envs <= 0 or cfg.num_episodes <= 0:
    raise ValueError("num_envs and num_episodes must be positive.")
  checkpoint = Path(cfg.checkpoint_file).resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
  output_dir = (
    Path(cfg.output_dir).resolve()
    if cfg.output_dir is not None
    else _default_output_dir(checkpoint).resolve()
  )
  output_dir.mkdir(parents=True, exist_ok=False)
  episodes_dir = output_dir / "episodes"
  episodes_dir.mkdir()

  configure_torch_backends()
  env_cfg = load_env_cfg(cfg.task)
  _apply_curriculum_stage_override(env_cfg, cfg.curriculum_stage)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.auto_reset = False

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped_env = RslRlVecEnvWrapper(
    raw_env, clip_actions=load_rl_cfg(cfg.task).clip_actions
  )
  try:
    raw_env.reset(seed=cfg.seed)
    policy = _load_policy(cfg.task, checkpoint, wrapped_env, cfg.device)
    obs = wrapped_env.get_observations()
    num_envs = raw_env.num_envs
    max_length = raw_env.max_episode_length
    qpos = raw_env.sim.data.qpos
    qvel = raw_env.sim.data.qvel
    qpos_history = torch.empty(
      (num_envs, max_length + 1, qpos.shape[1]),
      device=raw_env.device,
      dtype=qpos.dtype,
    )
    qvel_history = torch.empty(
      (num_envs, max_length + 1, qvel.shape[1]),
      device=raw_env.device,
      dtype=qvel.dtype,
    )
    qpos_history[:, 0] = qpos
    qvel_history[:, 0] = qvel
    lengths = torch.zeros(num_envs, device=raw_env.device, dtype=torch.long)
    returns = torch.zeros(num_envs, device=raw_env.device)
    primitive_event = _get_primitive_event(raw_env)
    command = _get_lifting_command(raw_env)
    initial_target_pos = command.target_pos.clone()
    manifest_path = output_dir / _MANIFEST_NAME
    episode_id = 0
    total_steps = 0

    print(f"[INFO] Collecting to {output_dir}")
    print(f"[INFO] Terminations: {raw_env.termination_manager.active_terms}")
    while episode_id < cfg.num_episodes:
      if cfg.max_steps is not None and total_steps >= cfg.max_steps:
        break
      with torch.no_grad():
        actions = policy(obs)
        obs_dict, reward, terminated, truncated, _ = raw_env.step(actions)
      total_steps += 1
      next_indices = lengths + 1
      env_indices = torch.arange(num_envs, device=raw_env.device)
      qpos_history[env_indices, next_indices] = qpos
      qvel_history[env_indices, next_indices] = qvel
      lengths += 1
      returns += reward
      done = terminated | truncated
      obs = TensorDict(obs_dict, batch_size=[num_envs])
      if not done.any():
        continue

      done_ids = done.nonzero(as_tuple=False).squeeze(-1)
      for env_id_tensor in done_ids:
        if episode_id >= cfg.num_episodes:
          break
        env_id = int(env_id_tensor.item())
        reasons = [
          name
          for name in raw_env.termination_manager.active_terms
          if bool(raw_env.termination_manager.get_term(name)[env_id].item())
        ]
        if not reasons:
          reasons = ["unknown"]
        length = int(lengths[env_id].item())
        filename = f"episode_{episode_id:06d}.pt"
        episode_path = episodes_dir / filename
        payload = {
          "format_version": 1,
          "task": cfg.task,
          "curriculum_stage": cfg.curriculum_stage,
          "checkpoint": str(checkpoint),
          "termination_reasons": reasons,
          "qpos": qpos_history[env_id, : length + 1].cpu(),
          "qvel": qvel_history[env_id, : length + 1].cpu(),
          "target_pos": initial_target_pos[env_id].cpu(),
          "env_origin": raw_env.scene.env_origins[env_id].cpu(),
          "shape_id": int(primitive_event.shape_ids[env_id].item()),
          "size": primitive_event.sizes[env_id].cpu(),
          "return": float(returns[env_id].item()),
        }
        torch.save(payload, episode_path)
        record = EpisodeRecord(
          episode_id=episode_id,
          path=str(episode_path.relative_to(output_dir)),
          length=length,
          return_value=payload["return"],
          termination_reasons=reasons,
          shape_id=payload["shape_id"],
          size=payload["size"].tolist(),
        )
        _append_manifest(manifest_path, record)
        episode_id += 1

      raw_env.reset(env_ids=done_ids)
      obs = wrapped_env.get_observations()
      qpos_history[done_ids, 0] = qpos[done_ids]
      qvel_history[done_ids, 0] = qvel[done_ids]
      lengths[done_ids] = 0
      returns[done_ids] = 0.0
      initial_target_pos[done_ids] = command.target_pos[done_ids]

      if episode_id and episode_id % 50 == 0:
        records = _read_manifest(output_dir)
        print(
          f"[INFO] Saved {episode_id}/{cfg.num_episodes}: "
          f"{_termination_counts(records)}"
        )
  finally:
    raw_env.close()

  records = _read_manifest(output_dir)
  print(f"[INFO] Finished: {len(records)} episodes")
  print(f"[INFO] Counts: {_termination_counts(records)}")


def _episode_env_origin(
  episode: dict[str, Any], object_q_adr: torch.Tensor
) -> torch.Tensor:
  saved_origin = episode.get("env_origin")
  if saved_origin is not None:
    return saved_origin

  # Format version 1 initially omitted env_origin. ParaHand's 1 m grid is centered
  # on either integers or half-integers, while reset object and target XY offsets
  # stay within 0.1 m. Pick the grid candidate nearest both saved positions.
  q_adr = object_q_adr[:2].cpu()
  object_xy = episode["qpos"][0, q_adr]
  target_xy = episode["target_pos"][:2]
  center = 0.5 * (object_xy + target_xy)
  integer = center.round()
  half_integer = (center - 0.5).round() + 0.5
  integer_error = (object_xy - integer).abs() + (target_xy - integer).abs()
  half_error = (object_xy - half_integer).abs() + (target_xy - half_integer).abs()
  origin_xy = torch.where(integer_error <= half_error, integer, half_integer)
  return torch.cat((origin_xy, torch.zeros(1, dtype=origin_xy.dtype)))


def _refresh_replay_point_cloud(env: ManagerBasedRlEnv) -> None:
  env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
  for group_name in ("actor", "critic"):
    term_cfg = env.observation_manager.get_term_cfg(group_name, "object_point_cloud_b")
    reset = getattr(term_cfg.func, "reset", None)
    if reset is not None:
      reset(env_ids)


class _PlaybackEnv:
  """Minimal environment adapter that feeds recorded states to Viser."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    episode: dict[str, Any],
    env_origin: torch.Tensor,
  ):
    self._env = env
    self._qpos = episode["qpos"].to(env.device)
    self._qvel = episode["qvel"].to(env.device)
    object_q_adr = env.scene["object"].data.indexing.free_joint_q_adr
    self._qpos[:, object_q_adr[:3]] -= env_origin.to(env.device)
    self._frame = 0
    self._write_frame()
    self._obs = self._compute_observations()

  def __getattr__(self, name: str):
    return getattr(self._env, name)

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self._env

  def get_observations(self):
    return self._obs

  def reset(self):
    self._frame = 0
    self._env.observation_manager.reset(
      torch.tensor([0], device=self._env.device, dtype=torch.long)
    )
    self._write_frame()
    self._obs = self._compute_observations()
    return self._obs, {}

  def step(self, actions: torch.Tensor):
    del actions
    self._frame = (self._frame + 1) % len(self._qpos)
    self._write_frame()
    self._obs = self._compute_observations()
    reward = torch.zeros(1, device=self._env.device)
    done = torch.zeros(1, device=self._env.device, dtype=torch.bool)
    return self._obs, reward, done, done, {}

  def _compute_observations(self):
    return self._env.observation_manager.compute(update_history=True)

  def _write_frame(self) -> None:
    self._env.sim.data.qpos[0] = self._qpos[self._frame]
    self._env.sim.data.qvel[0] = self._qvel[self._frame]
    self._env.sim.forward()
    self._env.scene.update(dt=0.0)
    self._env.sim.sense()


class _ZeroPolicy:
  def __init__(self, env: ManagerBasedRlEnv):
    self._shape = env.action_space.shape
    self._device = env.device

  def __call__(self, obs):
    del obs
    return torch.zeros(self._shape, device=self._device)


def _select_episode(cfg: Replay) -> tuple[Path, EpisodeRecord]:
  rollout_dir = Path(cfg.rollout_dir).resolve()
  matches = [
    record
    for record in _read_manifest(rollout_dir)
    if cfg.termination in record.termination_reasons
  ]
  if not matches:
    available = _termination_counts(_read_manifest(rollout_dir))
    raise ValueError(
      f"No episodes for termination '{cfg.termination}'. Available: {available}"
    )
  if not 0 <= cfg.episode_index < len(matches):
    raise IndexError(
      f"episode_index {cfg.episode_index} is outside [0, {len(matches) - 1}] "
      f"for termination '{cfg.termination}'."
    )
  return rollout_dir, matches[cfg.episode_index]


def replay(cfg: Replay) -> None:
  configure_torch_backends()
  rollout_dir, record = _select_episode(cfg)
  episode_path = rollout_dir / record.path
  episode = torch.load(episode_path, map_location="cpu", weights_only=True)
  env_cfg = load_env_cfg(episode["task"], play=True)
  recorded_stage = int(episode["curriculum_stage"])
  replay_stage = (
    recorded_stage if cfg.curriculum_stage is None else cfg.curriculum_stage
  )
  _apply_curriculum_stage_override(env_cfg, replay_stage)
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = int(1e9)
  env_cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    env.reset()
    primitive_event = _get_primitive_event(env)
    primitive_event.shape_ids[0] = int(episode["shape_id"])
    primitive_event.sizes[0] = episode["size"].to(env.device)
    primitive_event._write_primitive_model(  # pyright: ignore[reportPrivateUsage]
      env, torch.tensor([0], device=env.device)
    )
    env.sim.recompute_constants(RecomputeLevel.set_const)
    _refresh_replay_point_cloud(env)
    object_q_adr = env.scene["object"].data.indexing.free_joint_q_adr
    env_origin = _episode_env_origin(episode, object_q_adr)
    command = _get_lifting_command(env)
    command.target_pos[0] = episode["target_pos"].to(env.device) - env_origin.to(
      env.device
    )

    playback_env = _PlaybackEnv(env, episode, env_origin)
    print(
      f"[INFO] Replaying {record.path}: termination={record.termination_reasons}, "
      f"length={record.length}, return={record.return_value:.2f}"
    )
    ViserPlayViewer(cast(Any, playback_env), _ZeroPolicy(env), frame_rate=20.0).run()
  finally:
    env.close()


def list_rollouts(cfg: ListRollouts) -> None:
  records = _read_manifest(Path(cfg.rollout_dir).resolve())
  print(f"Episodes: {len(records)}")
  for name, count in sorted(_termination_counts(records).items()):
    print(f"{name}: {count}")


def main() -> None:
  command = tyro.cli(Collect | ListRollouts | Replay, config=mjlab.TYRO_FLAGS)
  if isinstance(command, Collect):
    collect(command)
  elif isinstance(command, ListRollouts):
    list_rollouts(command)
  else:
    replay(command)


if __name__ == "__main__":
  main()
