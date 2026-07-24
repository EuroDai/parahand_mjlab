from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.entity.variants import VARIANT_DEPENDENT_FIELDS
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .consts import NOMINAL_BOX_OBJECT_NAME, PRIMITIVE_OBJECTS

_VARIANT_MODEL_FIELDS = (
  "geom_dataid",
  "geom_matid",
  *VARIANT_DEPENDENT_FIELDS,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def reset_joints_by_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  curriculum_event_name: str,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Reset selected joints around their home keyframe as lessons advance."""
  curriculum_event_cfg = env.event_manager.get_term_cfg(curriculum_event_name)
  curriculum_stage = int(curriculum_event_cfg.params["curriculum_stage"])
  active_position_range = (0.0, 0.0) if curriculum_stage == 0 else position_range
  reset_joints_by_offset(
    env,
    env_ids,
    position_range=active_position_range,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )


class reset_variant_object_pose:
  """Apply the object lesson and randomize the active object's pose."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    object_name = cfg.params["object_name"]
    self.object: Entity = env.scene[object_name]
    variant_ids = env.sim.world_to_variant.get(object_name)
    if variant_ids is None:
      raise ValueError(f"Entity '{object_name}' must use VariantEntityCfg.")
    self.variant_ids = variant_ids
    self._assigned_variant_ids = variant_ids.clone()
    metadata = self.object.variant_metadata
    if metadata is None:
      raise ValueError(f"Entity '{object_name}' has no variant metadata.")
    try:
      self._nominal_box_variant_id = metadata.variant_names.index(
        NOMINAL_BOX_OBJECT_NAME
      )
    except ValueError as exc:
      raise ValueError(
        f"Entity '{object_name}' must define variant '{NOMINAL_BOX_OBJECT_NAME}'."
      ) from exc

    nominal_worlds = (
      self._assigned_variant_ids == self._nominal_box_variant_id
    ).nonzero()
    if len(nominal_worlds) == 0:
      raise ValueError(
        f"At least one world must be assigned '{NOMINAL_BOX_OBJECT_NAME}'."
      )
    nominal_world = int(nominal_worlds[0].item())
    self._assigned_model_fields = {
      field: (
        env.sim.get_default_field(field)
        if field in VARIANT_DEPENDENT_FIELDS
        else getattr(env.sim.model, field).clone()
      )
      for field in _VARIANT_MODEL_FIELDS
    }
    self._nominal_box_model_fields = {
      field: values[nominal_world].clone()
      for field, values in self._assigned_model_fields.items()
    }
    self._applied_stage = torch.full(
      (env.num_envs,), -1, dtype=torch.int8, device=env.device
    )
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
    curriculum_stage: int,
  ) -> None:
    del object_name
    self._apply_curriculum_stage(env, env_ids, curriculum_stage)
    num_resets = len(env_ids)
    xy_center = torch.tensor(position_center, device=env.device)
    xy_noise = (
      torch.zeros_like(xy_center)
      if curriculum_stage == 0
      else torch.tensor(position_noise, device=env.device)
    )
    active_position = torch.zeros(num_resets, 3, device=env.device)
    active_position[:, :2] = (
      xy_center + (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * xy_noise
    )
    active_position[:, 2] = self._floor_offsets[self.variant_ids[env_ids]]
    active_position += env.scene.env_origins[env_ids]

    yaw = torch.zeros(num_resets, device=env.device)
    if curriculum_stage > 0:
      yaw.uniform_(*yaw_range)
    zeros = torch.zeros(num_resets, device=env.device)
    active_orientation = quat_from_euler_xyz(zeros, zeros, yaw)
    self.object.write_root_link_pose_to_sim(
      torch.cat((active_position, active_orientation), dim=-1), env_ids=env_ids
    )
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(num_resets, 6, device=env.device), env_ids=env_ids
    )

  def _apply_curriculum_stage(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    curriculum_stage: int,
  ) -> None:
    if curriculum_stage not in (0, 1, 2):
      raise ValueError(f"Unsupported object curriculum stage: {curriculum_stage}.")
    changed_env_ids = env_ids[self._applied_stage[env_ids] != curriculum_stage]
    if len(changed_env_ids) == 0:
      return

    for field in _VARIANT_MODEL_FIELDS:
      model_field = getattr(env.sim.model, field)
      if curriculum_stage == 0:
        model_field[changed_env_ids] = self._nominal_box_model_fields[field]
      else:
        model_field[changed_env_ids] = self._assigned_model_fields[field][
          changed_env_ids
        ]

    if curriculum_stage == 0:
      self.variant_ids[changed_env_ids] = self._nominal_box_variant_id
    else:
      self.variant_ids[changed_env_ids] = self._assigned_variant_ids[changed_env_ids]
    self._applied_stage[changed_env_ids] = curriculum_stage
