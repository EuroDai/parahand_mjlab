from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class object_lesson_curriculum:
  """基于成功率进行课程推进"""

  _FINAL_STAGE = 2

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._event_cfg = env.event_manager.get_term_cfg(cfg.params["event_name"])
    command = env.command_manager.get_term(cfg.params["command_name"])
    if not isinstance(command, LiftingCommand):
      raise TypeError(
        "object_lesson_curriculum requires a LiftingCommand, "
        f"got {type(command).__name__}."
      )
    self._command = command
    self._object: Entity = env.scene[cfg.params["object_name"]]
    target_range = self._command.cfg.target_position_range
    self._random_target_ranges = (target_range.x, target_range.y, target_range.z)
    self._set_target_randomization(enabled=False)

    self._promotion_threshold = float(cfg.params["promotion_threshold"])
    self._success_threshold = float(cfg.params["success_threshold"])
    self._window_size = int(cfg.params["success_window_size"])
    self._min_completed_episodes = int(cfg.params["min_completed_episodes"])
    if not 0.0 <= self._promotion_threshold <= 1.0:
      raise ValueError("promotion_threshold must be between 0 and 1.")
    if self._window_size <= 0:
      raise ValueError("success_window_size must be positive.")
    if not 0 < self._min_completed_episodes <= self._window_size:
      raise ValueError(
        "min_completed_episodes must be positive and no greater than "
        "success_window_size."
      )

    self._success_history = torch.zeros(
      self._window_size, dtype=torch.bool, device=env.device
    )
    self._history_count = 0
    self._write_index = 0
    self._completed_episodes = 0
    self._stage = 0
    self._event_cfg.params["curriculum_stage"] = self._stage

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    event_name: str,
    command_name: str,
    object_name: str,
    promotion_threshold: float,
    success_threshold: float,
    success_window_size: int,
    min_completed_episodes: int,
  ) -> dict[str, torch.Tensor]:
    del (
      event_name,
      command_name,
      object_name,
      promotion_threshold,
      success_threshold,
      success_window_size,
      min_completed_episodes,
    )
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

    completed_mask = env.episode_length_buf[env_ids] > 0
    completed_env_ids = env_ids[completed_mask]
    reported_success_rate = self._success_rate()
    if len(completed_env_ids) > 0:
      position_error = torch.linalg.vector_norm(
        self._object.data.root_link_pos_w[completed_env_ids]
        - self._command.target_pos[completed_env_ids],
        dim=-1,
      )
      self._append_successes(position_error < self._success_threshold)
      reported_success_rate = self._success_rate()
      if (
        self._stage < self._FINAL_STAGE
        and self._history_count >= self._min_completed_episodes
        and reported_success_rate >= self._promotion_threshold
      ):
        self._stage += 1
        self._event_cfg.params["curriculum_stage"] = self._stage
        if self._stage == 1:
          self._set_target_randomization(enabled=True)
        self._clear_success_history()

    return {
      "stage": torch.tensor(self._stage, device=env.device, dtype=torch.float32),
      "success_rate": torch.tensor(
        reported_success_rate, device=env.device, dtype=torch.float32
      ),
      "completed_episodes": torch.tensor(
        self._completed_episodes, device=env.device, dtype=torch.float32
      ),
      "window_count": torch.tensor(
        self._history_count, device=env.device, dtype=torch.float32
      ),
    }

  def _clear_success_history(self) -> None:
    self._success_history.zero_()
    self._history_count = 0
    self._write_index = 0

  def _set_target_randomization(self, *, enabled: bool) -> None:
    target_range = self._command.cfg.target_position_range
    x_range, y_range, z_range = self._random_target_ranges
    if enabled:
      target_range.x = x_range
      target_range.y = y_range
      target_range.z = z_range
      return
    x_midpoint = (x_range[0] + x_range[1]) * 0.5
    y_midpoint = (y_range[0] + y_range[1]) * 0.5
    z_midpoint = (z_range[0] + z_range[1]) * 0.5
    target_range.x = (x_midpoint, x_midpoint)
    target_range.y = (y_midpoint, y_midpoint)
    target_range.z = (z_midpoint, z_midpoint)

  def _append_successes(self, successes: torch.Tensor) -> None:
    successes = successes.flatten()
    num_values = len(successes)
    if num_values == 0:
      return
    self._completed_episodes += num_values
    if num_values >= self._window_size:
      self._success_history.copy_(successes[-self._window_size :])
      self._history_count = self._window_size
      self._write_index = 0
      return

    first_count = min(num_values, self._window_size - self._write_index)
    self._success_history[self._write_index : self._write_index + first_count] = (
      successes[:first_count]
    )
    remaining = num_values - first_count
    if remaining > 0:
      self._success_history[:remaining] = successes[first_count:]
    self._write_index = (self._write_index + num_values) % self._window_size
    self._history_count = min(self._history_count + num_values, self._window_size)

  def _success_rate(self) -> float:
    if self._history_count == 0:
      return 0.0
    if self._history_count < self._window_size:
      history = self._success_history[: self._history_count]
    else:
      history = self._success_history
    return float(history.float().mean().item())
