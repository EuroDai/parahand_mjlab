from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

from ._table import get_table_heights
from .consts import (
  BOX_SPHERE_SCALE_RANGE,
  CAPSULE_SCALE_RANGE,
  PRIMITIVE_DATASET_STAGE,
  PRIMITIVE_OBJECTS,
  primitive_randomization_fraction,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

_DENSITY = 500.0
_INACTIVE_SIZE = 1.0e-6
_CAPSULE = 0
_BOX = 1
_SPHERE = 2
_SIZE_REFRESH_INTERVALS = {2: 16, 3: 8, 4: 4, 5: 2}
_CACHED_DERIVED_FIELDS = (
  "body_subtreemass",
  "dof_invweight0",
  "body_invweight0",
  "tendon_length0",
  "tendon_invweight0",
  "actuator_acc0",
)


def reset_joints_by_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  curriculum_stage: int,
) -> None:
  """Reset active joints using the range selected by the curriculum."""
  del curriculum_stage
  reset_joints_by_offset(
    env,
    env_ids,
    position_range=position_range,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )


class reset_table_height:
  """Randomize a mocap tabletop and optionally move the robot base with it."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    table_name = cfg.params["table_name"]
    self.table: Entity = env.scene[table_name]
    if not self.table.is_fixed_base or not self.table.is_mocap:
      raise ValueError("The randomized table must be a fixed-base mocap entity.")
    height_range = cfg.params["height_range"]
    midpoint = 0.5 * (float(height_range[0]) + float(height_range[1]))
    self.heights = torch.full(
      (env.num_envs,),
      midpoint,
      device=env.device,
      dtype=torch.float32,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    table_name: str,
    height_range: tuple[float, float],
    robot_name: str | None,
    robot_base_follows_table: bool,
    curriculum_stage: int,
  ) -> None:
    del table_name
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
      return
    fraction = primitive_randomization_fraction(curriculum_stage)
    midpoint = 0.5 * (height_range[0] + height_range[1])
    half_width = 0.5 * (height_range[1] - height_range[0]) * fraction
    self.heights[env_ids] = torch.empty(len(env_ids), device=env.device).uniform_(
      midpoint - half_width, midpoint + half_width
    )
    self._write_mocap_at_height(env, self.table, env_ids, self.heights[env_ids])

    if robot_base_follows_table:
      if robot_name is None:
        raise ValueError("robot_name is required when robot_base_follows_table=True.")
      robot: Entity = env.scene[robot_name]
      if not robot.is_fixed_base or not robot.is_mocap:
        raise ValueError("The table-following robot must be a fixed-base mocap entity.")
      self._write_mocap_at_height(env, robot, env_ids, self.heights[env_ids])

  @staticmethod
  def _write_mocap_at_height(
    env: ManagerBasedRlEnv,
    entity: Entity,
    env_ids: torch.Tensor,
    heights: torch.Tensor,
  ) -> None:
    default_root_state = entity.data.default_root_state
    assert default_root_state is not None
    pose = default_root_state[env_ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + heights
    entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)


class reset_joints_above_table:
  """Reset joints near home and sample the ParaHand palm height above the table."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    joint_ids, _ = self._asset.find_joints(
      (cfg.params["palm_height_joint_name"],),
      preserve_order=True,
    )
    if len(joint_ids) != 1:
      raise ValueError("Expected exactly one palm height joint.")
    self._palm_height_joint_id = torch.tensor(
      joint_ids,
      device=env.device,
      dtype=torch.long,
    )
    palm_joint_ranges = cfg.params["palm_joint_ranges"]
    palm_joint_ids, palm_joint_names = self._asset.find_joints(
      tuple(palm_joint_ranges), preserve_order=True
    )
    if tuple(palm_joint_names) != tuple(palm_joint_ranges):
      raise ValueError("Could not resolve all configured palm curriculum joints.")
    self._palm_joint_ids = torch.tensor(
      palm_joint_ids, device=env.device, dtype=torch.long
    )
    self._palm_joint_names = tuple(palm_joint_names)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    palm_height_joint_name: str,
    palm_height_range: tuple[float, float],
    palm_joint_ranges: dict[str, tuple[float, float]],
    curriculum_stage: int,
  ) -> None:
    del palm_height_joint_name, curriculum_stage
    reset_joints_by_offset(
      env,
      env_ids,
      position_range=position_range,
      velocity_range=velocity_range,
      asset_cfg=asset_cfg,
    )
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    default_position = self._asset.data.default_joint_pos[
      env_ids[:, None], self._palm_joint_ids[None, :]
    ]
    offsets = torch.stack(
      [
        torch.empty(len(env_ids), device=env.device).uniform_(*palm_joint_ranges[name])
        for name in self._palm_joint_names
      ],
      dim=-1,
    )
    palm_limits = self._asset.data.soft_joint_pos_limits[
      env_ids[:, None], self._palm_joint_ids[None, :]
    ]
    palm_position = (default_position + offsets).clamp(
      palm_limits[..., 0], palm_limits[..., 1]
    )
    self._asset.write_joint_position_to_sim(
      palm_position,
      joint_ids=self._palm_joint_ids,
      env_ids=env_ids,
    )
    self._asset.data.joint_pos_target[
      env_ids[:, None], self._palm_joint_ids[None, :]
    ] = palm_position
    palm_height = torch.empty(len(env_ids), 1, device=env.device).uniform_(
      *palm_height_range
    )
    palm_height_limits = self._asset.data.soft_joint_pos_limits[
      env_ids[:, None], self._palm_height_joint_id[None, :]
    ]
    palm_height.clamp_(
      palm_height_limits[..., 0],
      palm_height_limits[..., 1],
    )
    self._asset.write_joint_state_to_sim(
      palm_height,
      torch.zeros_like(palm_height),
      joint_ids=self._palm_height_joint_id,
      env_ids=env_ids,
    )
    self._asset.data.joint_pos_target[
      env_ids[:, None], self._palm_height_joint_id[None, :]
    ] = palm_height


class reset_primitive_object_pose:
  """Randomize an analytic primitive and reset its pose.

  The historical class name is retained so existing task configuration imports do
  not break. Unlike the old implementation, this uses three real primitive geom
  slots and samples continuous dimensions at every reset.
  """

  model_fields = (
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "geom_pos",
    "geom_rgba",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
  )
  recompute = RecomputeLevel.none

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    object_name = cfg.params["object_name"]
    self.object: Entity = env.scene[object_name]
    if self.object.variant_metadata is not None:
      raise ValueError("Primitive object reset requires a non-variant EntityCfg.")
    if len(self.object.indexing.geom_ids) != len(PRIMITIVE_OBJECTS):
      raise ValueError(
        f"Expected {len(PRIMITIVE_OBJECTS)} primitive object geoms, "
        f"got {len(self.object.indexing.geom_ids)}."
      )

    self._geom_ids = self.object.indexing.geom_ids.to(dtype=torch.long)
    self._body_id = int(self.object.indexing.body_ids[0].item())
    self._base_sizes = torch.tensor(
      [obj.size for obj in PRIMITIVE_OBJECTS],
      device=env.device,
      dtype=torch.float32,
    )
    self._colors = torch.tensor(
      [obj.rgba for obj in PRIMITIVE_OBJECTS],
      device=env.device,
      dtype=torch.float32,
    )
    self.shape_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.sizes = self._base_sizes[0].expand(env.num_envs, -1).clone()
    self._slot_sizes = self._base_sizes[None, :, :].expand(env.num_envs, -1, -1).clone()
    self._slot_cache_stage = torch.full(
      (env.num_envs,), -1, dtype=torch.int8, device=env.device
    )
    self._derived_cache = {
      name: torch.empty(
        env.num_envs,
        len(PRIMITIVE_OBJECTS),
        *getattr(env.sim.model, name).shape[1:],
        device=env.device,
        dtype=getattr(env.sim.model, name).dtype,
      )
      for name in _CACHED_DERIVED_FIELDS
    }
    self._random_stage = -1
    self._random_reset_count = 0
    self._applied_stage = torch.full(
      (env.num_envs,), -1, dtype=torch.int8, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    object_name: str,
    position_center: tuple[float, float],
    position_noise: tuple[float, float],
    capsule_roll_range: tuple[float, float],
    box_yaw_range: tuple[float, float],
    curriculum_stage: int,
    table_height_event_name: str,
    table_clearance: float,
  ) -> None:
    del object_name
    if not 0 <= curriculum_stage < PRIMITIVE_DATASET_STAGE:
      raise ValueError(f"Unsupported object curriculum stage: {curriculum_stage}.")
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    self._apply_curriculum_stage(env, env_ids, curriculum_stage)

    num_resets = len(env_ids)
    xy_center = torch.tensor(position_center, device=env.device)
    fraction = primitive_randomization_fraction(curriculum_stage)
    xy_noise = torch.tensor(position_noise, device=env.device) * fraction
    roll = torch.zeros(num_resets, device=env.device)
    yaw = torch.zeros_like(roll)
    capsule = self.shape_ids[env_ids] == _CAPSULE
    box = self.shape_ids[env_ids] == _BOX
    if curriculum_stage >= 2:
      if capsule.any():
        roll[capsule] = torch.empty(
          int(capsule.sum().item()), device=env.device
        ).uniform_(capsule_roll_range[0] * fraction, capsule_roll_range[1] * fraction)
      if box.any():
        yaw[box] = torch.empty(int(box.sum().item()), device=env.device).uniform_(
          box_yaw_range[0] * fraction, box_yaw_range[1] * fraction
        )
    zeros = torch.zeros_like(roll)
    active_orientation = quat_from_euler_xyz(roll, zeros, yaw)

    active_position = torch.zeros(num_resets, 3, device=env.device)
    active_position[:, :2] = (
      xy_center + (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * xy_noise
    )
    table_heights = get_table_heights(env, table_height_event_name)
    floor_offsets = self._floor_offsets(env_ids)
    if capsule.any():
      capsule_sizes = self.sizes[env_ids[capsule]]
      floor_offsets[capsule] = (
        capsule_sizes[:, 0] + capsule_sizes[:, 1] * torch.sin(roll[capsule]).abs()
      )
    active_position[:, 2] = (
      table_heights[env_ids] + floor_offsets + float(table_clearance)
    )
    active_position += env.scene.env_origins[env_ids]
    self.object.write_root_link_pose_to_sim(
      torch.cat((active_position, active_orientation), dim=-1), env_ids=env_ids
    )
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(num_resets, 6, device=env.device), env_ids=env_ids
    )

  def _apply_curriculum_stage(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, stage: int
  ) -> None:
    if stage <= 1:
      update_env_ids = env_ids[self._applied_stage[env_ids] != stage]
      if len(update_env_ids) == 0:
        return
      self._sample_primitives(update_env_ids, stage)
      self._write_primitive_model(env, update_env_ids)
      self._applied_stage[update_env_ids] = stage
      env.sim.recompute_constants(RecomputeLevel.set_const)
      return

    if stage != self._random_stage:
      self._random_stage = stage
      self._random_reset_count = 0
    self.shape_ids[env_ids] = torch.randint(
      len(PRIMITIVE_OBJECTS), (len(env_ids),), device=env.device
    )
    refresh_interval = _SIZE_REFRESH_INTERVALS[stage]
    periodic_refresh = self._random_reset_count % refresh_interval == 0
    invalid = self._slot_cache_stage[env_ids] != stage
    refresh_env_ids = env_ids if periodic_refresh else env_ids[invalid]
    if len(refresh_env_ids) > 0:
      self._refresh_size_slots(env, refresh_env_ids, stage)
    self._activate_cached_primitives(env, env_ids)
    self._applied_stage[env_ids] = stage
    self._random_reset_count += 1

  def _refresh_size_slots(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stage: int,
  ) -> None:
    selected_shape_ids = self.shape_ids[env_ids].clone()
    for shape_id in range(len(PRIMITIVE_OBJECTS)):
      self.shape_ids[env_ids] = shape_id
      self.sizes[env_ids] = self._sample_sizes(
        torch.full_like(env_ids, shape_id),
        stage,
      )
      self._slot_sizes[env_ids, shape_id] = self.sizes[env_ids]
      self._write_primitive_model(env, env_ids)
      env.sim.recompute_constants(RecomputeLevel.set_const)
      for name, cache in self._derived_cache.items():
        cache[env_ids, shape_id] = getattr(env.sim.model, name)[env_ids]
    self.shape_ids[env_ids] = selected_shape_ids
    self._slot_cache_stage[env_ids] = stage

  def _activate_cached_primitives(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
  ) -> None:
    shape_ids = self.shape_ids[env_ids]
    self.sizes[env_ids] = self._slot_sizes[env_ids, shape_ids]
    self._write_primitive_model(env, env_ids)
    for name, cache in self._derived_cache.items():
      getattr(env.sim.model, name)[env_ids] = cache[env_ids, shape_ids]

  def _sample_primitives(self, env_ids: torch.Tensor, stage: int) -> None:
    count = len(env_ids)
    if stage == 0:
      self.shape_ids[env_ids] = _CAPSULE
      self.sizes[env_ids] = self._base_sizes[_CAPSULE]
      return
    if stage == 1:
      shape_ids = env_ids.remainder(len(PRIMITIVE_OBJECTS))
      self.shape_ids[env_ids] = shape_ids
      self.sizes[env_ids] = self._base_sizes[shape_ids]
      return

    shape_ids = torch.randint(len(PRIMITIVE_OBJECTS), (count,), device=env_ids.device)
    self.shape_ids[env_ids] = shape_ids
    self.sizes[env_ids] = self._sample_sizes(shape_ids, stage)

  def _sample_sizes(
    self,
    shape_ids: torch.Tensor,
    stage: int,
  ) -> torch.Tensor:
    count = len(shape_ids)
    fraction = primitive_randomization_fraction(stage)
    minimum = torch.full((count, 1), BOX_SPHERE_SCALE_RANGE[0], device=shape_ids.device)
    maximum = torch.full((count, 1), BOX_SPHERE_SCALE_RANGE[1], device=shape_ids.device)
    capsule = shape_ids == _CAPSULE
    minimum[capsule] = CAPSULE_SCALE_RANGE[0]
    maximum[capsule] = CAPSULE_SCALE_RANGE[1]
    minimum = 1.0 + fraction * (minimum - 1.0)
    maximum = 1.0 + fraction * (maximum - 1.0)
    scales = (
      torch.rand(count, 3, device=shape_ids.device) * (maximum - minimum) + minimum
    )
    sizes = self._base_sizes[shape_ids] * scales
    # MuJoCo ignores unused primitive size components, but keeping them at zero
    # makes the physical parameterization explicit.
    sizes[shape_ids == _SPHERE, 1:] = 0.0
    sizes[shape_ids == _CAPSULE, 2] = 0.0
    return sizes

  def _write_primitive_model(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> None:
    count = len(env_ids)
    geom_ids = self._geom_ids
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    active = (
      torch.arange(len(PRIMITIVE_OBJECTS), device=env.device)[None, :]
      == self.shape_ids[env_ids, None]
    )

    all_sizes = torch.full(
      (count, len(PRIMITIVE_OBJECTS), 3),
      _INACTIVE_SIZE,
      device=env.device,
      dtype=self._base_sizes.dtype,
    )
    all_sizes[torch.arange(count, device=env.device), self.shape_ids[env_ids]] = (
      self.sizes[env_ids]
    )
    env.sim.model.geom_size[env_grid, geom_grid] = all_sizes

    env.sim.model.geom_pos[env_grid, geom_grid] = 0.0

    colors = self._colors[None, :, :].expand(count, -1, -1).clone()
    colors[..., 3] = active.to(colors.dtype)
    env.sim.model.geom_rgba[env_grid, geom_grid] = colors
    self._write_bounds(env, env_grid, geom_grid, all_sizes)
    self._write_inertia(env, env_ids)

  def _write_bounds(
    self,
    env: ManagerBasedRlEnv,
    env_grid: torch.Tensor,
    geom_grid: torch.Tensor,
    sizes: torch.Tensor,
  ) -> None:
    radius = sizes[:, _CAPSULE, 0]
    half_length = sizes[:, _CAPSULE, 1]
    rbound = torch.stack(
      (
        radius + half_length,
        torch.linalg.vector_norm(sizes[:, _BOX], dim=-1),
        sizes[:, _SPHERE, 0],
      ),
      dim=-1,
    )
    half_extents = sizes.clone()
    half_extents[:, _CAPSULE] = torch.stack(
      (radius, radius, radius + half_length), dim=-1
    )
    half_extents[:, _SPHERE] = sizes[:, _SPHERE, :1].expand(-1, 3)
    env.sim.model.geom_rbound[env_grid, geom_grid] = rbound
    env.sim.model.geom_aabb[env_grid, geom_grid, 0] = 0.0
    env.sim.model.geom_aabb[env_grid, geom_grid, 1] = half_extents

  def _write_inertia(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    shape_ids = self.shape_ids[env_ids]
    sizes = self.sizes[env_ids]
    mass = torch.empty(len(env_ids), device=env.device)
    inertia = torch.empty(len(env_ids), 3, device=env.device)

    box = shape_ids == _BOX
    if box.any():
      half = sizes[box]
      box_mass = _DENSITY * 8.0 * half.prod(dim=-1)
      mass[box] = box_mass
      inertia[box] = (box_mass[:, None] / 3.0) * torch.stack(
        (
          half[:, 1].square() + half[:, 2].square(),
          half[:, 0].square() + half[:, 2].square(),
          half[:, 0].square() + half[:, 1].square(),
        ),
        dim=-1,
      )

    sphere = shape_ids == _SPHERE
    if sphere.any():
      radius = sizes[sphere, 0]
      sphere_mass = _DENSITY * (4.0 / 3.0) * math.pi * radius.pow(3)
      mass[sphere] = sphere_mass
      sphere_inertia = 0.4 * sphere_mass * radius.square()
      inertia[sphere] = sphere_inertia[:, None].expand(-1, 3)

    capsule = shape_ids == _CAPSULE
    if capsule.any():
      radius = sizes[capsule, 0]
      half_length = sizes[capsule, 1]
      cylinder_mass = _DENSITY * math.pi * radius.square() * (2.0 * half_length)
      cap_mass = _DENSITY * (4.0 / 3.0) * math.pi * radius.pow(3)
      capsule_mass = cylinder_mass + cap_mass
      axial = 0.5 * cylinder_mass * radius.square() + 0.4 * cap_mass * radius.square()
      transverse = cylinder_mass * (
        3.0 * radius.square() + 4.0 * half_length.square()
      ) / 12.0 + cap_mass * (
        0.4 * radius.square() + (half_length + 0.375 * radius).square()
      )
      mass[capsule] = capsule_mass
      # The capsule geom is rotated so its long axis is body x.
      inertia[capsule] = torch.stack((axial, transverse, transverse), dim=-1)

    body_ids = torch.full_like(env_ids, self._body_id)
    env.sim.model.body_mass[env_ids, body_ids] = mass
    env.sim.model.body_inertia[env_ids, body_ids] = inertia
    env.sim.model.body_ipos[env_ids, body_ids] = 0.0
    identity = torch.zeros(len(env_ids), 4, device=env.device)
    identity[:, 0] = 1.0
    env.sim.model.body_iquat[env_ids, body_ids] = identity

  def _floor_offsets(self, env_ids: torch.Tensor) -> torch.Tensor:
    shape_ids = self.shape_ids[env_ids]
    sizes = self.sizes[env_ids]
    offsets = sizes[:, 0].clone()
    box = shape_ids == _BOX
    offsets[box] = sizes[box, 2]
    return offsets


# Backward-compatible import name for downstream task configurations.
reset_variant_object_pose = reset_primitive_object_pose


class reset_mesh_object_pose:
  """Reset fixed mesh variants using per-variant floor offsets."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    object_name = cfg.params["object_name"]
    self.object: Entity = env.scene[object_name]
    if self.object.variant_metadata is None:
      raise ValueError("Mesh object reset requires a VariantEntityCfg.")
    variant_ids = env.sim.world_to_variant.get(object_name)
    if variant_ids is None:
      raise ValueError(f"No mesh variant assignment found for '{object_name}'.")
    self._variant_ids = variant_ids.to(device=env.device, dtype=torch.long)
    offsets = cfg.params["floor_offsets"]
    if len(offsets) != len(self.object.variant_metadata.variant_names):
      raise ValueError(
        f"Expected {len(self.object.variant_metadata.variant_names)} floor offsets, "
        f"got {len(offsets)}."
      )
    self._floor_offsets = torch.tensor(
      offsets,
      device=env.device,
      dtype=torch.float32,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    object_name: str,
    position_center: tuple[float, float],
    position_noise: tuple[float, float],
    yaw_range: tuple[float, float],
    floor_offsets: tuple[float, ...],
  ) -> None:
    del object_name, floor_offsets
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_resets = len(env_ids)
    xy_center = torch.tensor(position_center, device=env.device)
    xy_noise = torch.tensor(position_noise, device=env.device)
    position = torch.zeros(num_resets, 3, device=env.device)
    position[:, :2] = (
      xy_center + (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * xy_noise
    )
    position[:, 2] = self._floor_offsets[self._variant_ids[env_ids]]
    position += env.scene.env_origins[env_ids]

    yaw = torch.empty(num_resets, device=env.device).uniform_(*yaw_range)
    zeros = torch.zeros(num_resets, device=env.device)
    orientation = quat_from_euler_xyz(zeros, zeros, yaw)
    self.object.write_root_link_pose_to_sim(
      torch.cat((position, orientation), dim=-1),
      env_ids=env_ids,
    )
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(num_resets, 6, device=env.device),
      env_ids=env_ids,
    )


class reset_dropped_mesh_object_pose:
  """Reset mesh variants above the floor with random SO(3) orientation."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    object_name = cfg.params["object_name"]
    self.object: Entity = env.scene[object_name]
    metadata = self.object.variant_metadata
    if metadata is None:
      raise ValueError("Dropped mesh reset requires a VariantEntityCfg.")
    variant_ids = env.sim.world_to_variant.get(object_name)
    if variant_ids is None:
      raise ValueError(f"No mesh variant assignment found for '{object_name}'.")
    self._variant_ids = variant_ids.to(device=env.device, dtype=torch.long)

    point_paths = cfg.params["variant_point_cloud_paths"]
    point_scales = cfg.params["variant_point_cloud_scales"]
    if len(point_paths) != len(metadata.variant_names) or len(point_scales) != len(
      metadata.variant_names
    ):
      raise ValueError(
        "Dropped mesh point-cloud paths and scales must align with mesh variants."
      )
    point_clouds = []
    for point_path, scale in zip(point_paths, point_scales, strict=True):
      points = np.load(Path(point_path), allow_pickle=False)
      if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
          f"Expected an (N, 3) point cloud at {point_path}, got {points.shape}."
        )
      if not np.isfinite(points).all():
        raise ValueError(f"Point cloud contains non-finite values: {point_path}.")
      point_clouds.append(
        torch.as_tensor(
          points,
          device=env.device,
          dtype=torch.float32,
        )
        * float(scale)
      )
    point_counts = {len(points) for points in point_clouds}
    if len(point_counts) != 1:
      raise ValueError("All dropped mesh point clouds must have the same size.")
    self._points_local = torch.stack(point_clouds)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    object_name: str,
    position_center: tuple[float, float],
    position_noise: tuple[float, float],
    drop_height_range: tuple[float, float],
    clearance: float,
    table_height_event_name: str,
    variant_point_cloud_paths: tuple[str, ...],
    variant_point_cloud_scales: tuple[float, ...],
  ) -> None:
    del object_name, variant_point_cloud_paths, variant_point_cloud_scales
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_resets = len(env_ids)
    if num_resets == 0:
      return

    orientation = torch.randn(num_resets, 4, device=env.device)
    orientation /= torch.linalg.vector_norm(
      orientation, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    # q and -q encode the same rotation. Canonicalizing the sign makes recorded
    # reset states easier to compare without changing the SO(3) distribution.
    orientation *= torch.where(
      orientation[:, :1] < 0.0,
      -torch.ones_like(orientation[:, :1]),
      torch.ones_like(orientation[:, :1]),
    )

    variant_points = self._points_local[self._variant_ids[env_ids]]
    point_orientation = orientation[:, None, :].expand(-1, variant_points.shape[1], -1)
    rotated_points = quat_apply(point_orientation, variant_points)
    support_height = -rotated_points[..., 2].amin(dim=1)

    xy_center = torch.tensor(position_center, device=env.device)
    xy_noise = torch.tensor(position_noise, device=env.device)
    position = torch.zeros(num_resets, 3, device=env.device)
    position[:, :2] = (
      xy_center + (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * xy_noise
    )
    drop_height = torch.empty(num_resets, device=env.device).uniform_(
      *drop_height_range
    )
    table_heights = get_table_heights(env, table_height_event_name)
    position[:, 2] = (
      table_heights[env_ids] + support_height + float(clearance) + drop_height
    )
    position += env.scene.env_origins[env_ids]

    self.object.write_root_link_pose_to_sim(
      torch.cat((position, orientation), dim=-1),
      env_ids=env_ids,
    )
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(num_resets, 6, device=env.device),
      env_ids=env_ids,
    )
