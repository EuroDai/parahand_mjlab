from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from .consts import OBJECT_SCALE_RANGE, PRIMITIVE_OBJECTS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

_DENSITY = 500.0
_INACTIVE_SIZE = 1.0e-6
_CAPSULE = 0
_BOX = 1
_SPHERE = 2


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
    yaw_range: tuple[float, float],
    curriculum_stage: int,
  ) -> None:
    del object_name
    if curriculum_stage not in (0, 1, 2):
      raise ValueError(f"Unsupported object curriculum stage: {curriculum_stage}.")
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    self._apply_curriculum_stage(env, env_ids, curriculum_stage)

    num_resets = len(env_ids)
    xy_center = torch.tensor(position_center, device=env.device)
    xy_noise = torch.tensor(position_noise, device=env.device)
    active_position = torch.zeros(num_resets, 3, device=env.device)
    active_position[:, :2] = (
      xy_center + (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * xy_noise
    )
    active_position[:, 2] = self._floor_offsets(env_ids)
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
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, stage: int
  ) -> None:
    changed_env_ids = env_ids[self._applied_stage[env_ids] != stage]
    if len(changed_env_ids) == 0:
      return
    # Stage 2 changes only point-cloud sampling. Preserve the stage-1 physical
    # object so the expensive model-constant recomputation is not repeated.
    if stage == 2:
      self._applied_stage[changed_env_ids] = stage
      return

    self._sample_primitives(changed_env_ids, stage)
    self._write_primitive_model(env, changed_env_ids)
    self._applied_stage[changed_env_ids] = stage
    env.sim.recompute_constants(RecomputeLevel.set_const)

  def _sample_primitives(self, env_ids: torch.Tensor, stage: int) -> None:
    count = len(env_ids)
    if stage == 0:
      self.shape_ids[env_ids] = _CAPSULE
      self.sizes[env_ids] = self._base_sizes[_CAPSULE]
      return

    shape_ids = torch.randint(len(PRIMITIVE_OBJECTS), (count,), device=env_ids.device)
    scales = torch.empty(count, 3, device=env_ids.device).uniform_(*OBJECT_SCALE_RANGE)
    sizes = self._base_sizes[shape_ids] * scales
    # MuJoCo ignores unused primitive size components, but keeping them at zero
    # makes the physical parameterization explicit.
    sizes[shape_ids == _SPHERE, 1:] = 0.0
    sizes[shape_ids == _CAPSULE, 2] = 0.0
    self.shape_ids[env_ids] = shape_ids
    self.sizes[env_ids] = sizes

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
