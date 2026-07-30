from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.tasks.parahand_grasp.mdp.actions import RelativeTendonLengthActionCfg
from mjlab.tasks.parahand_grasp.mdp.consts import (
  PRIMITIVE_DATASET_STAGE,
  primitive_randomization_fraction,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class object_lesson_curriculum:
  """基于成功率进行课程推进"""

  _FINAL_STAGE = PRIMITIVE_DATASET_STAGE

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._event_cfg = env.event_manager.get_term_cfg(cfg.params["event_name"])
    self._table_event_cfg = env.event_manager.get_term_cfg(
      cfg.params["table_event_name"]
    )
    self._robot_event_cfg = env.event_manager.get_term_cfg(
      cfg.params["robot_event_name"]
    )
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
    tendon_action = env.action_manager.get_term(cfg.params["tendon_action_name"])
    if not isinstance(tendon_action.cfg, RelativeTendonLengthActionCfg):
      raise TypeError("The tendon curriculum requires RelativeTendonLengthActionCfg.")
    self._tendon_action_cfg = tendon_action.cfg
    self._set_randomization_stage(0)

    self._promotion_thresholds = _stage_values(
      cfg.params["promotion_threshold"],
      PRIMITIVE_DATASET_STAGE,
      float,
    )
    self._success_threshold = float(cfg.params["success_threshold"])
    self._window_sizes = _stage_values(
      cfg.params["success_window_size"],
      PRIMITIVE_DATASET_STAGE,
      int,
    )
    self._minimum_episodes = _stage_values(
      cfg.params["min_completed_episodes"],
      PRIMITIVE_DATASET_STAGE,
      int,
    )
    for threshold in self._promotion_thresholds:
      if not 0.0 <= threshold <= 1.0:
        raise ValueError("promotion_threshold values must be between 0 and 1.")
    for window_size, minimum in zip(
      self._window_sizes, self._minimum_episodes, strict=True
    ):
      if window_size <= 0:
        raise ValueError("success_window_size values must be positive.")
      if not 0 < minimum <= window_size:
        raise ValueError(
          "min_completed_episodes values must be positive and no greater than "
          "their success_window_size."
        )

    max_window_size = max(self._window_sizes)
    self._success_history = torch.zeros(
      max_window_size, dtype=torch.bool, device=env.device
    )
    self._history_count = 0
    self._write_index = 0
    self._completed_episodes = 0
    self._stage = 0
    self._window_size = self._window_sizes[0]

  @property
  def stage(self) -> int:
    """Current logical lesson, including runner-managed Stage 2."""
    return self._stage

  def set_stage(self, stage: int) -> None:
    """Synchronize the logical lesson across distributed training ranks."""
    if not 0 <= stage <= self._FINAL_STAGE:
      raise ValueError(f"Curriculum stage must be in [0, {self._FINAL_STAGE}].")
    if stage < self._stage:
      raise ValueError("The object curriculum cannot move backwards.")
    self._stage = stage
    # Stage 6 uses a separate mesh environment. Primitive rehearsal remains at
    # the fully randomized Stage 5 settings.
    self._set_randomization_stage(min(stage, PRIMITIVE_DATASET_STAGE - 1))
    self._window_size = self._window_sizes[min(stage, PRIMITIVE_DATASET_STAGE - 1)]
    self._clear_success_history()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    event_name: str,
    table_event_name: str,
    robot_event_name: str,
    command_name: str,
    tendon_action_name: str,
    object_name: str,
    promotion_threshold: float,
    success_threshold: float,
    success_window_size: int,
    min_completed_episodes: int,
  ) -> dict[str, torch.Tensor]:
    del (
      event_name,
      table_event_name,
      robot_event_name,
      command_name,
      tendon_action_name,
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
      if self._stage < self._FINAL_STAGE and self._stage_is_complete():
        self.set_stage(self._stage + 1)

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

  def _set_randomization_stage(self, stage: int) -> None:
    fraction = primitive_randomization_fraction(stage)
    for event_cfg in (
      self._event_cfg,
      self._table_event_cfg,
      self._robot_event_cfg,
    ):
      event_cfg.params["curriculum_stage"] = stage

    active_range = (-0.5 * fraction, 0.5 * fraction)
    self._robot_event_cfg.params["position_range"] = active_range
    if "palm_joint_ranges" in self._robot_event_cfg.params:
      self._robot_event_cfg.params["palm_joint_ranges"] = {
        "palm_translation_x": (-0.1 * fraction, 0.2 * fraction),
        "palm_translation_y": (-0.2 * fraction, 0.2 * fraction),
        "palm_rotation_x": (-0.5 * fraction, 0.5 * fraction),
        "palm_rotation_y": (-0.5 * fraction, 0.5 * fraction),
        "palm_rotation_z": (-0.5 * fraction, 0.5 * fraction),
      }
      self._robot_event_cfg.params["palm_height_range"] = (
        0.3 - 0.1 * fraction,
        0.3 + 0.1 * fraction,
      )
    self._tendon_action_cfg.reset_target_range = (
      -0.05 * fraction,
      0.05 * fraction,
    )

    target_range = self._command.cfg.target_position_range
    x_range, y_range, z_range = self._random_target_ranges
    target_range.x = _scale_range_about_midpoint(x_range, fraction)
    target_range.y = _scale_range_about_midpoint(y_range, fraction)
    target_range.z = _scale_range_about_midpoint(z_range, fraction)

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

  def _stage_is_complete(self) -> bool:
    threshold = self._promotion_thresholds[self._stage]
    minimum = self._minimum_episodes[self._stage]
    return self._history_count >= minimum and self._success_rate() >= threshold


def _scale_range_about_midpoint(
  value_range: tuple[float, float], fraction: float
) -> tuple[float, float]:
  midpoint = 0.5 * (value_range[0] + value_range[1])
  half_width = 0.5 * (value_range[1] - value_range[0]) * fraction
  return midpoint - half_width, midpoint + half_width


def _stage_values(value, count: int, cast_type):
  if isinstance(value, (int, float)):
    return tuple(cast_type(value) for _ in range(count))
  values = tuple(cast_type(item) for item in value)
  if len(values) != count:
    raise ValueError(f"Expected {count} curriculum stage values, got {len(values)}.")
  return values
