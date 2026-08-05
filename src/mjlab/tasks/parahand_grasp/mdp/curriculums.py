from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.tasks.parahand_grasp.mdp.actions import RelativeTendonLengthActionCfg
from mjlab.tasks.parahand_grasp.mdp.consts import (
  ACTUATOR_EFFORT_FACTOR_RANGE,
  ACTUATOR_GAIN_FACTOR_RANGE,
  BOX_SCALE_RANGE,
  CAPSULE_SCALE_RANGE,
  GRAVITY_MAGNITUDE_FACTOR_RANGE,
  GRAVITY_TILT_MAX_RAD,
  JOINT_DAMPING_FACTOR_RANGE,
  OBJECT_COM_OFFSET_MAX_M,
  OBJECT_DENSITY_FACTOR_RANGE,
  OBJECT_FRICTION_FACTOR_RANGE,
  PALM_TRACKING_LAST_STAGE,
  POINT_CLOUD_NOISE_STD_MAX_M,
  PRIMITIVE_DATASET_STAGE,
  PRIMITIVE_OBJECTS,
  SPHERE_SCALE_RANGE,
  TABLE_FRICTION_FACTOR_RANGE,
  primitive_gravity_fraction,
  primitive_randomization_fraction,
  primitive_shape_probabilities,
)
from mjlab.utils.noise import GaussianNoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.managers.observation_manager import ObservationTermCfg

_SHAPE_NAMES = ("capsule", "box", "sphere")
_SIZE_DIMENSIONS = (
  ("capsule", (0,), "radius", CAPSULE_SCALE_RANGE),
  ("capsule", (1,), "half_length", CAPSULE_SCALE_RANGE),
  ("box", (0, 1, 2), "half_extent", BOX_SCALE_RANGE),
  ("sphere", (0,), "radius", SPHERE_SCALE_RANGE),
)


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
    self._gravity_event_cfg = env.event_manager.get_term_cfg(
      cfg.params["gravity_event_name"]
    )
    self._physics_event_cfg = env.event_manager.get_term_cfg(
      cfg.params["physics_event_name"]
    )
    command = env.command_manager.get_term(cfg.params["command_name"])
    if not isinstance(command, LiftingCommand):
      raise TypeError(
        "object_lesson_curriculum requires a LiftingCommand, "
        f"got {type(command).__name__}."
      )
    self._command = command
    self._object: Entity = env.scene[cfg.params["object_name"]]
    tendon_action = env.action_manager.get_term(cfg.params["tendon_action_name"])
    if not isinstance(tendon_action.cfg, RelativeTendonLengthActionCfg):
      raise TypeError("The tendon curriculum requires RelativeTendonLengthActionCfg.")
    self._tendon_action_cfg = tendon_action.cfg
    self._point_cloud_observation_cfg = self._get_point_cloud_observation_cfg(env)
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
    self._stale_episode_count = 0
    self._stage = 0
    self._window_size = self._window_sizes[0]
    self._episode_stages = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

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
    # Stage 7 uses a separate mesh environment. Primitive rehearsal remains at
    # the fully randomized Stage 6 settings with independent Palm resets.
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
    gravity_event_name: str,
    physics_event_name: str,
    command_name: str,
    tendon_action_name: str,
    object_name: str,
    promotion_threshold: float,
    success_threshold: float,
    success_window_size: int,
    min_completed_episodes: int,
  ) -> dict[str, float]:
    del (
      event_name,
      table_event_name,
      robot_event_name,
      gravity_event_name,
      physics_event_name,
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
    self._completed_episodes += len(completed_env_ids)
    if len(completed_env_ids) > 0:
      current_stage_mask = self._episode_stages[completed_env_ids] == self._stage
      self._stale_episode_count += int((~current_stage_mask).sum().item())
      completed_env_ids = completed_env_ids[current_stage_mask]

    if len(completed_env_ids) > 0:
      position_error = torch.linalg.vector_norm(
        self._object.data.root_link_pos_w[completed_env_ids]
        - self._command.target_pos[completed_env_ids],
        dim=-1,
      )
      self._append_successes(position_error < self._success_threshold)
      if self._stage < self._FINAL_STAGE and self._stage_is_complete():
        self.set_stage(self._stage + 1)

    # These envs are about to reset using the current event configuration.
    self._episode_stages[env_ids] = self._stage

    return self._state(env)

  def _state(self, env: ManagerBasedRlEnv) -> dict[str, float]:
    state = {
      "stage": float(self._stage),
      "success_rate": self._success_rate(),
      "completed_episodes": float(self._completed_episodes),
      "window_count": float(self._history_count),
      "stale_episode_count": float(self._stale_episode_count),
    }
    primitive_stage = min(self._stage, PRIMITIVE_DATASET_STAGE - 1)
    state["palm_tracking"] = self._scalar(
      float(primitive_stage <= PALM_TRACKING_LAST_STAGE), env
    )
    point_cloud_noise_std = (
      POINT_CLOUD_NOISE_STD_MAX_M * primitive_randomization_fraction(primitive_stage)
    )
    state["point_cloud/noise_std_mm"] = self._scalar(
      point_cloud_noise_std * 1000.0, env
    )

    probabilities = primitive_shape_probabilities(primitive_stage)
    for shape_name, probability in zip(_SHAPE_NAMES, probabilities, strict=True):
      state[f"shape/configured_percent/{shape_name}"] = self._scalar(
        probability * 100.0, env
      )

    self._add_size_config_metrics(state, primitive_stage, env)
    self._add_reset_config_metrics(state, primitive_stage, env)
    self._add_physics_config_metrics(state, primitive_stage, env)
    return state

  def _clear_success_history(self) -> None:
    self._success_history.zero_()
    self._history_count = 0
    self._write_index = 0
    self._stale_episode_count = 0

  def _set_randomization_stage(self, stage: int) -> None:
    fraction = primitive_randomization_fraction(stage)
    for event_cfg in (
      self._event_cfg,
      self._table_event_cfg,
      self._robot_event_cfg,
      self._gravity_event_cfg,
      self._physics_event_cfg,
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
    if self._point_cloud_observation_cfg is not None:
      noise_std = POINT_CLOUD_NOISE_STD_MAX_M * fraction
      self._point_cloud_observation_cfg.noise = (
        GaussianNoiseCfg(mean=0.0, std=noise_std) if noise_std > 0.0 else None
      )

  @staticmethod
  def _get_point_cloud_observation_cfg(
    env: ManagerBasedRlEnv,
  ) -> ObservationTermCfg | None:
    observation_manager = getattr(env, "observation_manager", None)
    if observation_manager is None:
      return None
    term_cfg = observation_manager.get_term_cfg("actor", "object_point_cloud_b")
    if term_cfg.noise is not None and not isinstance(term_cfg.noise, GaussianNoiseCfg):
      raise TypeError("The actor point-cloud curriculum requires GaussianNoiseCfg.")
    return term_cfg

  def _append_successes(self, successes: torch.Tensor) -> None:
    successes = successes.flatten()
    num_values = len(successes)
    if num_values == 0:
      return
    if num_values >= self._window_size:
      self._success_history[: self._window_size].copy_(successes[-self._window_size :])
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

  def _add_size_config_metrics(
    self,
    state: dict[str, float],
    stage: int,
    env: ManagerBasedRlEnv,
  ) -> None:
    fraction = primitive_randomization_fraction(stage)
    for shape_name, dimensions, dimension_name, scale_range in _SIZE_DIMENSIONS:
      shape_id = _SHAPE_NAMES.index(shape_name)
      nominal_size = PRIMITIVE_OBJECTS[shape_id].size[dimensions[0]]
      minimum = nominal_size * (scale_range[0] - 1.0) * fraction
      maximum = nominal_size * (scale_range[1] - 1.0) * fraction
      prefix = f"size/{shape_name}/{dimension_name}"
      state[f"{prefix}/configured_deviation_min_m"] = self._scalar(minimum, env)
      state[f"{prefix}/configured_deviation_max_m"] = self._scalar(maximum, env)

  def _add_reset_config_metrics(
    self,
    state: dict[str, float],
    stage: int,
    env: ManagerBasedRlEnv,
  ) -> None:
    fraction = primitive_randomization_fraction(stage)
    position_noise = self._event_cfg.params["position_noise"]
    for axis, configured_max in zip(("x", "y"), position_noise, strict=True):
      state[f"reset/object_xy/configured_abs_max_m/{axis}"] = self._scalar(
        abs(float(configured_max)) * fraction, env
      )

    yaw_ranges = {
      "capsule": self._event_cfg.params["capsule_yaw_range"],
      "box_sphere": self._event_cfg.params["box_yaw_range"],
    }
    for shape_name, yaw_range in yaw_ranges.items():
      yaw_abs_max = max(abs(float(yaw_range[0])), abs(float(yaw_range[1])))
      state[f"reset/object_yaw/configured_abs_max_rad/{shape_name}"] = self._scalar(
        yaw_abs_max * fraction, env
      )

    table_range = self._table_event_cfg.params["height_range"]
    table_midpoint = 0.5 * (float(table_range[0]) + float(table_range[1]))
    table_half_width = 0.5 * (float(table_range[1]) - float(table_range[0]))
    state["reset/table_height/configured_m/min"] = self._scalar(
      table_midpoint - table_half_width * fraction, env
    )
    state["reset/table_height/configured_m/max"] = self._scalar(
      table_midpoint + table_half_width * fraction, env
    )

    joint_range = self._robot_event_cfg.params["position_range"]
    state["reset/joint_offset/configured_rad/min"] = self._scalar(
      float(joint_range[0]), env
    )
    state["reset/joint_offset/configured_rad/max"] = self._scalar(
      float(joint_range[1]), env
    )
    tendon_range = self._tendon_action_cfg.reset_target_range
    state["reset/tendon_offset/configured_m/min"] = self._scalar(
      float(tendon_range[0]), env
    )
    state["reset/tendon_offset/configured_m/max"] = self._scalar(
      float(tendon_range[1]), env
    )

    if stage > PALM_TRACKING_LAST_STAGE:
      self._add_home_palm_config_metrics(state, env)

  def _add_home_palm_config_metrics(
    self,
    state: dict[str, float],
    env: ManagerBasedRlEnv,
  ) -> None:
    palm_ranges = self._robot_event_cfg.params.get("palm_joint_ranges")
    if not isinstance(palm_ranges, dict):
      return
    metric_names = {
      "palm_translation_x": "translation_x_offset_m",
      "palm_translation_y": "translation_y_offset_m",
      "palm_rotation_x": "rotation_x_offset_rad",
      "palm_rotation_y": "rotation_y_offset_rad",
      "palm_rotation_z": "rotation_z_offset_rad",
    }
    for joint_name, metric_name in metric_names.items():
      value_range = palm_ranges[joint_name]
      prefix = f"reset/palm_home_random/{metric_name}/configured"
      state[f"{prefix}/min"] = self._scalar(float(value_range[0]), env)
      state[f"{prefix}/max"] = self._scalar(float(value_range[1]), env)
    height_range = self._robot_event_cfg.params["palm_height_range"]
    prefix = "reset/palm_home_random/translation_z_position_m/configured"
    state[f"{prefix}/min"] = self._scalar(float(height_range[0]), env)
    state[f"{prefix}/max"] = self._scalar(float(height_range[1]), env)

  def _add_physics_config_metrics(
    self,
    state: dict[str, float],
    stage: int,
    env: ManagerBasedRlEnv,
  ) -> None:
    fraction = primitive_randomization_fraction(stage)
    configured_ranges = {
      "object_density/configured_factor": OBJECT_DENSITY_FACTOR_RANGE,
      "object_friction/configured_factor": OBJECT_FRICTION_FACTOR_RANGE,
      "table_friction/configured_factor": TABLE_FRICTION_FACTOR_RANGE,
      "joint_damping/configured_factor": JOINT_DAMPING_FACTOR_RANGE,
      "actuator_gain/configured_factor": ACTUATOR_GAIN_FACTOR_RANGE,
      "actuator_effort/configured_factor": ACTUATOR_EFFORT_FACTOR_RANGE,
    }
    for metric_name, full_range in configured_ranges.items():
      value_range = self._interpolate_scale_range(full_range, fraction)
      state[f"physics/{metric_name}/min"] = self._scalar(value_range[0], env)
      state[f"physics/{metric_name}/max"] = self._scalar(value_range[1], env)

    state["physics/object_com/configured_abs_max_m"] = self._scalar(
      OBJECT_COM_OFFSET_MAX_M * fraction, env
    )
    gravity = self._physics_event_cfg.params.get("gravity", (0.0, 0.0, -9.81))
    base_gravity_magnitude = math.sqrt(sum(float(value) ** 2 for value in gravity))
    gravity_magnitude = base_gravity_magnitude * primitive_gravity_fraction(stage)
    gravity_factor_range = self._interpolate_scale_range(
      GRAVITY_MAGNITUDE_FACTOR_RANGE, fraction
    )
    state["physics/gravity/configured_magnitude_mps2/current"] = self._scalar(
      gravity_magnitude, env
    )
    state["physics/gravity/configured_magnitude_mps2/min"] = self._scalar(
      gravity_magnitude * gravity_factor_range[0], env
    )
    state["physics/gravity/configured_magnitude_mps2/max"] = self._scalar(
      gravity_magnitude * gravity_factor_range[1], env
    )
    state["physics/gravity/configured_tilt_abs_max_rad"] = self._scalar(
      GRAVITY_TILT_MAX_RAD * fraction, env
    )

  @staticmethod
  def _interpolate_scale_range(
    value_range: tuple[float, float], fraction: float
  ) -> tuple[float, float]:
    return (
      1.0 + fraction * (value_range[0] - 1.0),
      1.0 + fraction * (value_range[1] - 1.0),
    )

  @staticmethod
  def _scalar(value: float | int, env: ManagerBasedRlEnv) -> float:
    del env
    return float(value)

  def _success_rate(self) -> float:
    if self._history_count == 0:
      return 0.0
    history = self._success_history[: self._history_count]
    return float(history.float().mean().item())

  def _stage_is_complete(self) -> bool:
    threshold = self._promotion_thresholds[self._stage]
    minimum = self._minimum_episodes[self._stage]
    return self._history_count >= minimum and self._success_rate() >= threshold


def _stage_values(value, count: int, cast_type):
  if isinstance(value, (int, float)):
    return tuple(cast_type(value) for _ in range(count))
  values = tuple(cast_type(item) for item in value)
  if len(values) != count:
    raise ValueError(f"Expected {count} curriculum stage values, got {len(values)}.")
  return values
