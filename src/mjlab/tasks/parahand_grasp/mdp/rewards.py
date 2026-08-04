from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.tasks.parahand_grasp.mdp.observations import object_point_cloud_b

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_NON_THUMB_CONTACT_LIKELIHOODS = (0.96, 0.92, 0.79, 0.39)
_CONTACT_WEIGHT_FLOOR = 0.2
_CONTACT_WEIGHT_POWER = 2.0
_FINGERTIP_CONTACT_NAMES = (
  "thumb_tac",
  "index_tac",
  "middle_tac",
  "ring_tac",
  "little_tac",
)


class object_point_cloud_debug_visualizer:
  """Draw the actor's sampled object point cloud using debug spheres."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    observation_group = cfg.params["observation_group"]
    observation_term = cfg.params["observation_term"]
    term_cfg = env.observation_manager.get_term_cfg(
      observation_group,
      observation_term,
    )
    if not isinstance(term_cfg.func, object_point_cloud_b):
      raise TypeError(
        f"Observation '{observation_group}/{observation_term}' must use "
        "object_point_cloud_b."
      )

    self._env = env
    self._point_cloud = term_cfg.func
    self._radius = float(cfg.params["radius"])
    self._color = tuple(cfg.params["color"])
    self._debug_vis_enabled = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    observation_group: str,
    observation_term: str,
    radius: float,
    color: tuple[float, float, float, float],
  ) -> torch.Tensor:
    del observation_group, observation_term, radius, color
    return torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    del env_ids

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled:
      return

    points_w = self._point_cloud.latest_points_w
    if points_w is None:
      return

    env_indices = list(visualizer.get_env_indices(self._env.num_envs))
    if not env_indices:
      return

    selected_points = points_w[env_indices].detach().cpu().numpy()
    for env_idx, points in zip(env_indices, selected_points, strict=True):
      for point_idx, point in enumerate(points):
        visualizer.add_sphere(
          center=point,
          radius=self._radius,
          color=self._color,
          label=f"object_point_cloud_{env_idx}_{point_idx}",
        )


def action_l2(env: ManagerBasedRlEnv, action_names: tuple[str, ...]) -> torch.Tensor:
  """Squared action magnitude, clamped to match the playground reward."""
  action, _ = _action_pair(env, action_names)
  return torch.sum(torch.square(action), dim=-1).clamp_max(1000.0)


def action_rate_l2(
  env: ManagerBasedRlEnv, action_names: tuple[str, ...]
) -> torch.Tensor:
  """Squared action change, clamped to match the playground reward."""
  action, previous_action = _action_pair(env, action_names)
  action_delta = action - previous_action
  return torch.sum(torch.square(action_delta), dim=-1).clamp_max(1000.0)


def _action_pair(
  env: ManagerBasedRlEnv, action_names: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
  action_terms = [env.action_manager.get_term(name) for name in action_names]
  actions = torch.cat([term.raw_action for term in action_terms], dim=-1)
  previous_actions = []
  for name, term in zip(action_names, action_terms, strict=True):
    previous_action = getattr(term, "prev_raw_action", None)
    if not isinstance(previous_action, torch.Tensor):
      raise TypeError(f"Action term '{name}' does not expose prev_raw_action.")
    previous_actions.append(previous_action)
  return actions, torch.cat(previous_actions, dim=-1)


def fingers_to_object(
  env: ManagerBasedRlEnv,
  std: float,
  object_cfg: SceneEntityCfg,
  fingertip_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward the mean fingertip distance to the object."""
  robot: Entity = env.scene[fingertip_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  fingertip_pos_w = robot.data.site_pos_w[:, fingertip_cfg.site_ids]
  distance = (
    torch.linalg.vector_norm(
      fingertip_pos_w - obj.data.root_link_pos_w[:, None, :], dim=-1
    )
    .max(dim=-1)
    .values
  )
  return 1.0 - torch.tanh(distance / std)


def smooth_contact_score(
  force_magnitude: torch.Tensor,
  threshold: float,
  temperature: float,
  core_both_weight: float = 0.5,
  auxiliary_bonus: float = 0.5,
) -> torch.Tensor:
  """Return a structured thumb, core-finger, and auxiliary-finger contact score."""
  if temperature <= 0.0:
    raise ValueError(f"Contact temperature must be positive, got {temperature}.")
  if not 0.0 <= core_both_weight <= 1.0:
    raise ValueError(f"Core-both weight must be in [0, 1], got {core_both_weight}.")
  if not 0.0 <= auxiliary_bonus <= 1.0:
    raise ValueError(f"Auxiliary bonus must be in [0, 1], got {auxiliary_bonus}.")
  if force_magnitude.shape[-1] != 5:
    raise ValueError(
      "Structured contact scoring requires thumb, index, middle, ring, "
      f"and little fingertip forces, got {force_magnitude.shape[-1]}."
    )

  finger_scores = torch.sigmoid((force_magnitude - threshold) / temperature)
  thumb_score, index_score, middle_score, ring_score, little_score = (
    finger_scores.unbind(dim=-1)
  )

  contact_likelihoods = finger_scores.new_tensor(_NON_THUMB_CONTACT_LIKELIHOODS)
  finger_weights = _CONTACT_WEIGHT_FLOOR + (1.0 - _CONTACT_WEIGHT_FLOOR) * torch.pow(
    contact_likelihoods / contact_likelihoods[0], _CONTACT_WEIGHT_POWER
  )
  index_weight, middle_weight, ring_weight, little_weight = finger_weights

  core_any_score = 1.0 - (1.0 - index_score) * (1.0 - middle_score)
  core_scores = torch.stack((index_score, middle_score), dim=-1)
  core_weights = torch.stack((index_weight, middle_weight))
  core_both_score = torch.exp(
    (
      torch.log(core_scores.clamp_min(torch.finfo(core_scores.dtype).eps))
      * core_weights
    ).sum(dim=-1)
    / core_weights.sum()
  )
  core_score = (
    1.0 - core_both_weight
  ) * core_any_score + core_both_weight * core_both_score

  auxiliary_score = (ring_weight * ring_score + little_weight * little_score) / (
    ring_weight + little_weight
  )
  base_gate = thumb_score * core_score
  return base_gate * (1.0 + auxiliary_bonus * (1.0 - base_gate) * auxiliary_score)


def contact_score(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
  temperature: float,
  core_both_weight: float = 0.5,
  auxiliary_bonus: float = 0.5,
) -> torch.Tensor:
  """Score structured fingertip object contact with a smooth force gate."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  force_magnitude = torch.linalg.vector_norm(force, dim=-1)
  primary_names = sensor.primary_names
  try:
    fingertip_indices = [primary_names.index(name) for name in _FINGERTIP_CONTACT_NAMES]
  except ValueError as exc:
    raise ValueError(
      f"Contact sensor '{sensor_name}' must contain {_FINGERTIP_CONTACT_NAMES}; "
      f"got {primary_names}."
    ) from exc
  return smooth_contact_score(
    force_magnitude[:, fingertip_indices],
    threshold,
    temperature,
    core_both_weight,
    auxiliary_bonus,
  )


def position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  sensor_name: str,
  std: float,
  contact_threshold: float,
  contact_temperature: float,
  contact_core_both_weight: float = 0.5,
  contact_auxiliary_bonus: float = 0.5,
) -> torch.Tensor:
  """Track the target position while maintaining a smooth contact gate."""
  obj: Entity = env.scene[object_cfg.name]
  target_position = _target_position(env, command_name)
  distance = torch.linalg.vector_norm(
    obj.data.root_link_pos_w - target_position, dim=-1
  )
  contact_gate = contact_score(
    env,
    sensor_name,
    contact_threshold,
    contact_temperature,
    contact_core_both_weight,
    contact_auxiliary_bonus,
  )
  return (1.0 - torch.tanh(distance / std)) * contact_gate


class object_lift:
  """Reward reset-relative vertical progress while maintaining contact."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    object_cfg = cfg.params["object_cfg"]
    if not isinstance(object_cfg, SceneEntityCfg):
      raise TypeError("object_lift object_cfg must be a SceneEntityCfg.")
    self._object: Entity = env.scene[object_cfg.name]
    self._initial_height = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initial_height[env_ids] = self._object_height()[env_ids]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_cfg: SceneEntityCfg,
    sensor_name: str,
    contact_threshold: float,
    contact_temperature: float,
    contact_core_both_weight: float = 0.5,
    contact_auxiliary_bonus: float = 0.5,
  ) -> torch.Tensor:
    del object_cfg
    object_height = self._object_height()
    target_height = _target_position(env, command_name)[:, 2]
    lift_height = (object_height - self._initial_height).clamp_min(0.0)
    target_lift_height = (target_height - self._initial_height).clamp_min(1.0e-3)
    lift_progress = (lift_height / target_lift_height).clamp(0.0, 1.0)
    contact_gate = contact_score(
      env,
      sensor_name,
      contact_threshold,
      contact_temperature,
      contact_core_both_weight,
      contact_auxiliary_bonus,
    )
    return lift_progress * contact_gate

  def _object_height(self) -> torch.Tensor:
    q_adr = self._object.data.indexing.free_joint_q_adr
    return self._object.data.data.qpos[:, q_adr[2]]


def good_finger_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
  temperature: float,
  core_both_weight: float = 0.5,
  auxiliary_bonus: float = 0.5,
) -> torch.Tensor:
  """Reward structured smooth fingertip object contact."""
  return contact_score(
    env,
    sensor_name,
    threshold,
    temperature,
    core_both_weight,
    auxiliary_bonus,
  )


def success(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  pos_std: float,
) -> torch.Tensor:
  """Position-only success shaping reward."""
  obj: Entity = env.scene[object_cfg.name]
  target_position = _target_position(env, command_name)
  distance = torch.linalg.vector_norm(
    obj.data.root_link_pos_w - target_position, dim=-1
  )
  return torch.square(1.0 - torch.tanh(distance / pos_std))


def final_position_success(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  threshold: float,
) -> torch.Tensor:
  """Return the binary final-pose success criterion used by the curriculum."""
  obj: Entity = env.scene[object_cfg.name]
  target_position = _target_position(env, command_name)
  distance = torch.linalg.vector_norm(
    obj.data.root_link_pos_w - target_position,
    dim=-1,
  )
  return (distance < threshold).to(dtype=torch.float32)


def _target_position(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  return command.target_pos
