from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .consts import PRIMITIVE_OBJECTS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class reset_variant_object_pose:
  """Randomize pose while keeping each world's assigned object variant fixed."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    object_name = cfg.params["object_name"]
    self.object: Entity = env.scene[object_name]
    variant_ids = env.sim.world_to_variant.get(object_name)
    if variant_ids is None:
      raise ValueError(f"Entity '{object_name}' must use VariantEntityCfg.")
    self.variant_ids = variant_ids.to(device=env.device, dtype=torch.long)
    self._floor_offsets = torch.tensor(
      [obj.floor_offset for obj in PRIMITIVE_OBJECTS], device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    object_name: str,
    position_center: tuple[float, float],
    position_noise: tuple[float, float],
    yaw_range: tuple[float, float],
  ) -> None:
    del object_name
    num_resets = len(env_ids)
    xy_center = torch.tensor(position_center, device=env.device)
    xy_noise = torch.tensor(position_noise, device=env.device)
    active_position = torch.zeros(num_resets, 3, device=env.device)
    active_position[:, :2] = (
      xy_center + (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * xy_noise
    )
    active_position[:, 2] = self._floor_offsets[self.variant_ids[env_ids]]
    active_position += env.scene.env_origins[env_ids]

    yaw = torch.empty(num_resets, device=env.device).uniform_(*yaw_range)
    zeros = torch.zeros(num_resets, device=env.device)
    active_orientation = quat_from_euler_xyz(zeros, zeros, yaw)
    self.object.write_root_link_pose_to_sim(
      torch.cat((active_position, active_orientation), dim=-1), env_ids=env_ids
    )
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(num_resets, 6, device=env.device), env_ids=env_ids
    )
