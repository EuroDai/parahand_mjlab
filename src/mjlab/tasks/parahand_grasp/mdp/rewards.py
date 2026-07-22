from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
  distance = torch.linalg.vector_norm(
    fingertip_pos_w - obj.data.root_link_pos_w[:, None, :], dim=-1
  ).mean(dim=-1)
  return 1.0 - torch.tanh(distance / std)


def contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  """Require thumb contact and contact from at least one other finger."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  force_magnitude = torch.linalg.vector_norm(force, dim=-1)
  thumb_contact = force_magnitude[:, 0] > threshold
  other_contact = (force_magnitude[:, 1:] > threshold).any(dim=-1)
  return thumb_contact & other_contact


def position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  sensor_name: str,
  std: float,
  contact_threshold: float,
) -> torch.Tensor:
  """Track the target position while maintaining the Isaac-style contact gate."""
  obj: Entity = env.scene[object_cfg.name]
  target_position = _target_position(env, command_name)
  distance = torch.linalg.vector_norm(
    obj.data.root_link_pos_w - target_position, dim=-1
  )
  contact_gate = contacts(env, sensor_name, contact_threshold).float()
  return (1.0 - torch.tanh(distance / std)) * contact_gate


def good_finger_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
) -> torch.Tensor:
  """Reward thumb-plus-one-finger object contact."""
  return contacts(env, sensor_name, threshold).float()


def success(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_cfg: SceneEntityCfg,
  pos_std: float,
) -> torch.Tensor:
  """Isaac-style position-only success shaping reward."""
  obj: Entity = env.scene[object_cfg.name]
  target_position = _target_position(env, command_name)
  distance = torch.linalg.vector_norm(
    obj.data.root_link_pos_w - target_position, dim=-1
  )
  return torch.square(1.0 - torch.tanh(distance / pos_std))


def _target_position(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  return command.target_pos
