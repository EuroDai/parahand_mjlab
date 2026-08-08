from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_euler_xyz,
)

from ._table import get_table_heights
from .actions import RelativeTendonLengthAction, apply_mcp_1_constraints
from .consts import (
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
  ORIENTATION_RANDOMIZATION_STAGE,
  PALM_TRACKING_LAST_STAGE,
  PRIMITIVE_DATASET_STAGE,
  PRIMITIVE_OBJECTS,
  SPHERE_SCALE_RANGE,
  TABLE_FRICTION_FACTOR_RANGE,
  primitive_gravity_fraction,
  primitive_randomization_fraction,
  primitive_shape_probabilities,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

_DENSITY = 500.0
_INACTIVE_SIZE = 1.0e-6
_CAPSULE = 0
_BOX = 1
_SPHERE = 2
_SIZE_REFRESH_INTERVALS = {3: 16, 4: 4, 5: 2, 6: 2}
_PALM_JOINT_NAMES = (
  "palm_translation_x",
  "palm_translation_y",
  "palm_translation_z",
  "palm_rotation_x",
  "palm_rotation_y",
  "palm_rotation_z",
)
_BOX_SPHERE_PALM_REFERENCE = (-0.12, -0.005, 0.168, 0.23, 0.0, 0.0)
_CAPSULE_PALM_REFERENCE = (-0.122, -0.005, 0.15, 0.23, 0.0, 0.0)
_BOX_SPHERE_OBJECT_REFERENCE_HEIGHT = 0.03
_CAPSULE_OBJECT_REFERENCE_HEIGHT = 0.02
_BOX_SPHERE_HAND_JOINT_REFERENCE = {
  "thumb_cmc_1": 0.0,
  "thumb_cmc_2": 0.22,
  "thumb_mcp": 0.23,
  "thumb_ip": 0.63,
  "index_mcp_1": 0.0,
  "index_mcp_2": 0.7,
  "index_pip": 0.47,
  "index_dip": 0.36,
  "middle_mcp_1": 0.0,
  "middle_mcp_2": 0.7,
  "middle_pip": 0.47,
  "middle_dip": 0.36,
  "ring_mcp_1": 0.0,
  "ring_mcp_2": 0.7,
  "ring_pip": 0.47,
  "ring_dip": 0.36,
  "little_mcp_1": 0.0,
  "little_mcp_2": 0.7,
  "little_pip": 0.47,
  "little_dip": 0.36,
}
_CAPSULE_HAND_JOINT_REFERENCE = {
  "thumb_cmc_1": 0.0,
  "thumb_cmc_2": 0.22,
  "thumb_mcp": 0.32,
  "thumb_ip": 0.63,
  "index_mcp_1": 0.0,
  "index_mcp_2": 0.95,
  "index_pip": 0.45,
  "index_dip": 0.39,
  "middle_mcp_1": 0.0,
  "middle_mcp_2": 0.95,
  "middle_pip": 0.45,
  "middle_dip": 0.39,
  "ring_mcp_1": 0.0,
  "ring_mcp_2": 0.95,
  "ring_pip": 0.45,
  "ring_dip": 0.39,
  "little_mcp_1": 0.0,
  "little_mcp_2": 0.95,
  "little_pip": 0.45,
  "little_dip": 0.39,
}
_PASSIVE_FINGER_JOINT_NAMES = (
  "index_pip",
  "index_dip",
  "middle_pip",
  "middle_dip",
  "ring_pip",
  "ring_dip",
  "little_pip",
  "little_dip",
)
_TRACKING_TENDON_REFERENCE = 0.021
_CACHED_DERIVED_FIELDS = (
  "body_subtreemass",
  "dof_invweight0",
  "body_invweight0",
  "tendon_length0",
  "tendon_invweight0",
  "actuator_acc0",
)


def _matrix_from_intrinsic_xyz(euler_xyz: torch.Tensor) -> torch.Tensor:
  """Convert intrinsic XYZ Euler angles to rotation matrices."""
  roll, pitch, yaw = euler_xyz.unbind(dim=-1)
  zero = torch.zeros_like(roll)
  one = torch.ones_like(roll)
  cr, sr = torch.cos(roll), torch.sin(roll)
  cp, sp = torch.cos(pitch), torch.sin(pitch)
  cy, sy = torch.cos(yaw), torch.sin(yaw)
  roll_matrix = torch.stack(
    (one, zero, zero, zero, cr, -sr, zero, sr, cr),
    dim=-1,
  ).reshape(-1, 3, 3)
  pitch_matrix = torch.stack(
    (cp, zero, sp, zero, one, zero, -sp, zero, cp),
    dim=-1,
  ).reshape(-1, 3, 3)
  yaw_matrix = torch.stack(
    (cy, -sy, zero, sy, cy, zero, zero, zero, one),
    dim=-1,
  ).reshape(-1, 3, 3)
  return roll_matrix @ pitch_matrix @ yaw_matrix


def _intrinsic_xyz_from_matrix(rotation: torch.Tensor) -> torch.Tensor:
  """Convert rotation matrices to intrinsic XYZ Euler angles."""
  pitch = torch.asin(rotation[:, 0, 2].clamp(-1.0, 1.0))
  roll = torch.atan2(-rotation[:, 1, 2], rotation[:, 2, 2])
  yaw = torch.atan2(-rotation[:, 0, 1], rotation[:, 0, 0])
  return torch.stack((roll, pitch, yaw), dim=-1)


class reset_joints_by_curriculum:
  """Reset active joints using the curriculum-selected range."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def __call__(
    self,
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


def reset_gravity_by_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  gravity: tuple[float, float, float],
  curriculum_stage: int,
) -> None:
  """Scale per-environment gravity for the current primitive lesson."""
  env_ids = env_ids.to(device=env.device, dtype=torch.long)
  gravity_w = torch.tensor(
    gravity,
    device=env.device,
    dtype=env.sim.model.opt.gravity.dtype,
  )
  env.sim.model.opt.gravity[env_ids] = gravity_w * primitive_gravity_fraction(
    curriculum_stage
  )


def _curriculum_scale_range(
  value_range: tuple[float, float], fraction: float
) -> tuple[float, float]:
  """Interpolate a multiplicative randomization range from the identity."""
  return (
    1.0 + fraction * (value_range[0] - 1.0),
    1.0 + fraction * (value_range[1] - 1.0),
  )


class randomize_teacher_physics:
  """Apply curriculum-scaled physical domain randomization at episode reset."""

  model_fields = (
    "body_mass",
    "body_ipos",
    "body_inertia",
    "geom_friction",
    "dof_damping",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "jnt_actfrcrange",
    "tendon_actfrcrange",
    *_CACHED_DERIVED_FIELDS,
  )
  recompute = RecomputeLevel.set_const

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    object_cfg = cfg.params["object_cfg"]
    self._object: Entity = env.scene[object_cfg.name]
    self._object_body_id = int(self._object.indexing.body_ids[0].item())
    self._object_is_variant = self._object.variant_metadata is not None

    model = env.sim.model
    self._nominal_object_mass = model.body_mass[:, self._object_body_id].clone()
    self._nominal_object_ipos = model.body_ipos[:, self._object_body_id].clone()
    self._nominal_object_inertia = model.body_inertia[:, self._object_body_id].clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    table_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    gravity: tuple[float, float, float],
    curriculum_stage: int,
  ) -> None:
    """Randomize physical parameters without accumulating across resets."""
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    fraction = primitive_randomization_fraction(curriculum_stage)
    model = env.sim.model

    if self._object_is_variant:
      base_mass = self._nominal_object_mass[env_ids]
      base_ipos = self._nominal_object_ipos[env_ids]
      base_inertia = self._nominal_object_inertia[env_ids]
    else:
      # The primitive reset immediately before this event writes the shape-specific
      # nominal mass, COM, and inertia for the current sampled dimensions.
      base_mass = model.body_mass[env_ids, self._object_body_id].clone()
      base_ipos = model.body_ipos[env_ids, self._object_body_id].clone()
      base_inertia = model.body_inertia[env_ids, self._object_body_id].clone()

    density_range = _curriculum_scale_range(OBJECT_DENSITY_FACTOR_RANGE, fraction)
    density_factor = torch.empty(
      len(env_ids), device=env.device, dtype=base_mass.dtype
    ).uniform_(*density_range)
    com_offset = torch.empty(
      len(env_ids), 3, device=env.device, dtype=base_ipos.dtype
    ).uniform_(
      -OBJECT_COM_OFFSET_MAX_M * fraction,
      OBJECT_COM_OFFSET_MAX_M * fraction,
    )
    model.body_mass[env_ids, self._object_body_id] = base_mass * density_factor
    model.body_inertia[env_ids, self._object_body_id] = (
      base_inertia * density_factor[:, None]
    )
    model.body_ipos[env_ids, self._object_body_id] = base_ipos + com_offset

    dr.geom_friction(
      env,
      env_ids,
      ranges=_curriculum_scale_range(OBJECT_FRICTION_FACTOR_RANGE, fraction),
      asset_cfg=object_cfg,
      operation="scale",
      axes=[0, 1, 2],
      shared_random=True,
    )
    dr.geom_friction(
      env,
      env_ids,
      ranges=_curriculum_scale_range(TABLE_FRICTION_FACTOR_RANGE, fraction),
      asset_cfg=table_cfg,
      operation="scale",
      axes=[0, 1, 2],
      shared_random=True,
    )
    dr.joint_damping(
      env,
      env_ids,
      ranges=_curriculum_scale_range(JOINT_DAMPING_FACTOR_RANGE, fraction),
      asset_cfg=robot_cfg,
      operation="scale",
      shared_random=True,
    )
    gain_range = _curriculum_scale_range(ACTUATOR_GAIN_FACTOR_RANGE, fraction)
    dr.pd_gains(
      env,
      env_ids,
      kp_range=gain_range,
      kd_range=gain_range,
      asset_cfg=robot_cfg,
      operation="scale",
    )
    dr.effort_limits(
      env,
      env_ids,
      effort_limit_range=_curriculum_scale_range(
        ACTUATOR_EFFORT_FACTOR_RANGE, fraction
      ),
      asset_cfg=robot_cfg,
      operation="scale",
    )

    base_gravity = torch.tensor(
      gravity,
      device=env.device,
      dtype=model.opt.gravity.dtype,
    )
    gravity_fraction = primitive_gravity_fraction(curriculum_stage)
    magnitude_factor = torch.empty(
      len(env_ids), device=env.device, dtype=base_gravity.dtype
    ).uniform_(*_curriculum_scale_range(GRAVITY_MAGNITUDE_FACTOR_RANGE, fraction))
    tilt_radius = (
      torch.rand(len(env_ids), device=env.device, dtype=base_gravity.dtype).sqrt()
      * GRAVITY_TILT_MAX_RAD
      * fraction
    )
    tilt_azimuth = (
      torch.rand(len(env_ids), device=env.device, dtype=base_gravity.dtype)
      * 2.0
      * math.pi
    )
    tilt = torch.stack(
      (
        tilt_radius * torch.cos(tilt_azimuth),
        tilt_radius * torch.sin(tilt_azimuth),
      ),
      dim=-1,
    )
    zero = torch.zeros(len(env_ids), device=env.device, dtype=base_gravity.dtype)
    gravity_quat = quat_from_euler_xyz(tilt[:, 0], tilt[:, 1], zero)
    gravity_w = (
      base_gravity.expand(len(env_ids), -1)
      * gravity_fraction
      * magnitude_factor[:, None]
    )
    gravity_w = quat_apply(gravity_quat, gravity_w)
    model.opt.gravity[env_ids] = gravity_w


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
  """Reset ParaHand joints with object-relative or independently random Palm poses."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.joint_names is None:
      raise ValueError("ParaHand reset requires explicit active joint names.")
    active_joint_ids, active_joint_names = self._asset.find_joints(
      tuple(asset_cfg.joint_names),
      preserve_order=True,
    )
    if tuple(active_joint_names) != tuple(asset_cfg.joint_names):
      raise ValueError("Could not resolve all configured active ParaHand joints.")
    self._active_joint_ids = torch.tensor(
      active_joint_ids, device=env.device, dtype=torch.long
    )
    self._active_joint_names = tuple(active_joint_names)
    self._mcp_1_active_columns = (
      self._active_joint_names.index("index_mcp_1"),
      self._active_joint_names.index("middle_mcp_1"),
      self._active_joint_names.index("ring_mcp_1"),
      self._active_joint_names.index("little_mcp_1"),
    )
    passive_joint_ids, passive_joint_names = self._asset.find_joints(
      _PASSIVE_FINGER_JOINT_NAMES,
      preserve_order=True,
    )
    if tuple(passive_joint_names) != _PASSIVE_FINGER_JOINT_NAMES:
      raise ValueError("Could not resolve all passive ParaHand finger joints.")
    self._passive_joint_ids = torch.tensor(
      passive_joint_ids, device=env.device, dtype=torch.long
    )
    self._box_sphere_passive_reference = torch.tensor(
      [_BOX_SPHERE_HAND_JOINT_REFERENCE[name] for name in _PASSIVE_FINGER_JOINT_NAMES],
      device=env.device,
      dtype=torch.float32,
    )
    self._capsule_passive_reference = torch.tensor(
      [_CAPSULE_HAND_JOINT_REFERENCE[name] for name in _PASSIVE_FINGER_JOINT_NAMES],
      device=env.device,
      dtype=torch.float32,
    )
    palm_joint_ids, palm_joint_names = self._asset.find_joints(
      _PALM_JOINT_NAMES, preserve_order=True
    )
    if tuple(palm_joint_names) != _PALM_JOINT_NAMES:
      raise ValueError("Could not resolve all ParaHand Palm joints.")
    self._palm_joint_ids = torch.tensor(
      palm_joint_ids, device=env.device, dtype=torch.long
    )
    palm_body_ids, palm_body_names = self._asset.find_bodies(
      ("palm",), preserve_order=True
    )
    if tuple(palm_body_names) != ("palm",):
      raise ValueError("Could not resolve the ParaHand Palm body.")
    palm_body_id = int(self._asset.indexing.body_ids[palm_body_ids[0]].item())
    palm_base_quat = torch.tensor(
      env.sim.mj_model.body_quat[palm_body_id],
      device=env.device,
      dtype=torch.float32,
    )
    self._palm_base_rotation = matrix_from_quat(palm_base_quat[None])[0]
    box_sphere_active_reference = torch.full(
      (len(self._active_joint_names),),
      torch.nan,
      device=env.device,
      dtype=torch.float32,
    )
    capsule_active_reference = box_sphere_active_reference.clone()
    for name, value in _BOX_SPHERE_HAND_JOINT_REFERENCE.items():
      if name in self._active_joint_names:
        box_sphere_active_reference[self._active_joint_names.index(name)] = value
    for name, value in _CAPSULE_HAND_JOINT_REFERENCE.items():
      if name in self._active_joint_names:
        capsule_active_reference[self._active_joint_names.index(name)] = value
    self._box_sphere_active_reference = box_sphere_active_reference
    self._capsule_active_reference = capsule_active_reference

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
    object_pose_event_name: str,
    tendon_action_name: str,
  ) -> None:
    del asset_cfg
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    tracking = curriculum_stage <= PALM_TRACKING_LAST_STAGE
    object_reset = (
      self._object_reset_term(env, object_pose_event_name) if tracking else None
    )
    shape_ids = (
      object_reset.shape_ids[env_ids]
      if object_reset is not None
      else torch.full_like(env_ids, -1)
    )

    default_joint_pos = self._asset.data.default_joint_pos
    default_joint_vel = self._asset.data.default_joint_vel
    soft_joint_pos_limits = self._asset.data.soft_joint_pos_limits
    assert default_joint_pos is not None
    assert default_joint_vel is not None
    assert soft_joint_pos_limits is not None

    joint_position = default_joint_pos[
      env_ids[:, None], self._active_joint_ids[None, :]
    ].clone()
    if tracking:
      reference_mask = torch.isfinite(self._box_sphere_active_reference)
      joint_position = torch.where(
        reference_mask,
        self._box_sphere_active_reference,
        joint_position,
      )
      capsule = shape_ids == _CAPSULE
      if capsule.any():
        capsule_reference_mask = torch.isfinite(self._capsule_active_reference)
        joint_position[capsule] = torch.where(
          capsule_reference_mask,
          self._capsule_active_reference,
          joint_position[capsule],
        )
    joint_position += torch.empty_like(joint_position).uniform_(*position_range)
    joint_limits = soft_joint_pos_limits[
      env_ids[:, None], self._active_joint_ids[None, :]
    ]
    joint_position.clamp_(joint_limits[..., 0], joint_limits[..., 1])
    apply_mcp_1_constraints(
      joint_position,
      joint_limits,
      self._mcp_1_active_columns,
    )
    joint_velocity = default_joint_vel[
      env_ids[:, None], self._active_joint_ids[None, :]
    ].clone()
    joint_velocity += torch.empty_like(joint_velocity).uniform_(*velocity_range)
    self._asset.write_joint_state_to_sim(
      joint_position,
      joint_velocity,
      joint_ids=self._active_joint_ids,
      env_ids=env_ids,
    )
    self._asset.data.joint_pos_target[
      env_ids[:, None], self._active_joint_ids[None, :]
    ] = joint_position
    if tracking:
      passive_position = self._box_sphere_passive_reference.expand(
        len(env_ids), -1
      ).clone()
      passive_position[shape_ids == _CAPSULE] = self._capsule_passive_reference
      passive_limits = soft_joint_pos_limits[
        env_ids[:, None], self._passive_joint_ids[None, :]
      ]
      passive_position.clamp_(passive_limits[..., 0], passive_limits[..., 1])
      self._asset.write_joint_state_to_sim(
        passive_position,
        torch.zeros_like(passive_position),
        joint_ids=self._passive_joint_ids,
        env_ids=env_ids,
      )

    tendon_action = env.action_manager.get_term(tendon_action_name)
    if not isinstance(tendon_action, RelativeTendonLengthAction):
      raise TypeError(
        f"Action '{tendon_action_name}' must be RelativeTendonLengthAction."
      )
    tendon_centers = torch.full(
      (len(env_ids),), torch.nan, device=env.device, dtype=torch.float32
    )
    if tracking:
      tendon_centers.fill_(_TRACKING_TENDON_REFERENCE)
    tendon_action.set_reset_target_center(env_ids, tendon_centers)

    if tracking:
      assert object_reset is not None
      palm_position = self._tracking_palm_position(
        env,
        env_ids,
        shape_ids,
        object_reset.latest_root_pose[env_ids],
        palm_height_joint_name,
      )
    else:
      palm_position = self._random_palm_position(
        env_ids,
        palm_height_joint_name,
        palm_height_range,
        palm_joint_ranges,
      )
    palm_limits = soft_joint_pos_limits[env_ids[:, None], self._palm_joint_ids[None, :]]
    palm_position.clamp_(palm_limits[..., 0], palm_limits[..., 1])
    self._asset.write_joint_state_to_sim(
      palm_position,
      torch.zeros_like(palm_position),
      joint_ids=self._palm_joint_ids,
      env_ids=env_ids,
    )
    self._asset.data.joint_pos_target[
      env_ids[:, None], self._palm_joint_ids[None, :]
    ] = palm_position

  @staticmethod
  def _object_reset_term(
    env: ManagerBasedRlEnv,
    object_pose_event_name: str,
  ) -> reset_primitive_object_pose:
    object_event_cfg = env.event_manager.get_term_cfg(object_pose_event_name)
    if not isinstance(object_event_cfg.func, reset_primitive_object_pose):
      raise TypeError(
        f"Event '{object_pose_event_name}' must use reset_primitive_object_pose."
      )
    return object_event_cfg.func

  def _tracking_palm_position(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    shape_ids: torch.Tensor,
    object_pose_w: torch.Tensor,
    palm_height_joint_name: str,
  ) -> torch.Tensor:
    del palm_height_joint_name
    capsule = shape_ids == _CAPSULE
    palm_reference = (
      object_pose_w.new_tensor(_BOX_SPHERE_PALM_REFERENCE)
      .expand(len(env_ids), -1)
      .clone()
    )
    palm_reference[capsule] = object_pose_w.new_tensor(_CAPSULE_PALM_REFERENCE)
    object_reference_height = object_pose_w.new_full(
      (len(env_ids),), _BOX_SPHERE_OBJECT_REFERENCE_HEIGHT
    )
    object_reference_height[capsule] = _CAPSULE_OBJECT_REFERENCE_HEIGHT

    relative_translation = palm_reference[:, :3].clone()
    relative_translation[:, 2] -= object_reference_height
    rotated_translation = quat_apply(object_pose_w[:, 3:7], relative_translation)

    table_heights = get_table_heights(env, "reset_table_height")
    object_position_b = object_pose_w[:, :3] - env.scene.env_origins[env_ids]
    object_position_b[:, 2] -= table_heights[env_ids]
    palm_translation = object_position_b + rotated_translation

    object_rotation = matrix_from_quat(object_pose_w[:, 3:7])
    palm_base_rotation = self._palm_base_rotation.to(object_pose_w).expand(
      len(env_ids), -1, -1
    )
    palm_reference_rotation = _matrix_from_intrinsic_xyz(palm_reference[:, 3:6])
    palm_joint_rotation = (
      palm_base_rotation.transpose(-1, -2)
      @ object_rotation
      @ palm_base_rotation
      @ palm_reference_rotation
    )
    palm_rotation = _intrinsic_xyz_from_matrix(palm_joint_rotation)
    return torch.cat((palm_translation, palm_rotation), dim=-1)

  def _random_palm_position(
    self,
    env_ids: torch.Tensor,
    palm_height_joint_name: str,
    palm_height_range: tuple[float, float],
    palm_joint_ranges: dict[str, tuple[float, float]],
  ) -> torch.Tensor:
    default_position = self._asset.data.default_joint_pos[
      env_ids[:, None], self._palm_joint_ids[None, :]
    ]
    offsets = torch.stack(
      [
        torch.empty(len(env_ids), device=default_position.device).uniform_(
          *(
            palm_height_range
            if name == palm_height_joint_name
            else palm_joint_ranges[name]
          )
        )
        for name in _PALM_JOINT_NAMES
      ],
      dim=-1,
    )
    height_column = _PALM_JOINT_NAMES.index(palm_height_joint_name)
    offsets[:, height_column] -= default_position[:, height_column]
    return default_position + offsets


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
    self.latest_root_pose = torch.zeros(
      env.num_envs, 7, device=env.device, dtype=torch.float32
    )
    self.latest_root_pose[:, 3] = 1.0
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
    capsule_yaw_range: tuple[float, float],
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
    yaw = torch.zeros(num_resets, device=env.device)
    capsule = self.shape_ids[env_ids] == _CAPSULE
    box = self.shape_ids[env_ids] == _BOX
    sphere = self.shape_ids[env_ids] == _SPHERE
    if curriculum_stage >= ORIENTATION_RANDOMIZATION_STAGE:
      if capsule.any():
        yaw[capsule] = torch.empty(
          int(capsule.sum().item()), device=env.device
        ).uniform_(capsule_yaw_range[0] * fraction, capsule_yaw_range[1] * fraction)
      yaw_shape = box | sphere
      if yaw_shape.any():
        yaw[yaw_shape] = torch.empty(
          int(yaw_shape.sum().item()), device=env.device
        ).uniform_(box_yaw_range[0] * fraction, box_yaw_range[1] * fraction)
    zeros = torch.zeros_like(yaw)
    active_orientation = quat_from_euler_xyz(zeros, zeros, yaw)

    active_position = torch.zeros(num_resets, 3, device=env.device)
    active_position[:, :2] = (
      xy_center + (torch.rand(num_resets, 2, device=env.device) * 2.0 - 1.0) * xy_noise
    )
    table_heights = get_table_heights(env, table_height_event_name)
    floor_offsets = self._floor_offsets(env_ids)
    active_position[:, 2] = (
      table_heights[env_ids] + floor_offsets + float(table_clearance)
    )
    active_position += env.scene.env_origins[env_ids]
    active_pose = torch.cat((active_position, active_orientation), dim=-1)
    self.latest_root_pose[env_ids] = active_pose
    self.object.write_root_link_pose_to_sim(active_pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(num_resets, 6, device=env.device), env_ids=env_ids
    )

  def _apply_curriculum_stage(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, stage: int
  ) -> None:
    if stage == 0:
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
    self.shape_ids[env_ids] = self._sample_shape_ids(len(env_ids), stage, env.device)
    fraction = primitive_randomization_fraction(stage)
    periodic_refresh = False
    if fraction > 0.0:
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
    shape_ids = tuple(
      shape_id
      for shape_id, probability in enumerate(primitive_shape_probabilities(stage))
      if probability > 0.0
    )
    for shape_id in shape_ids:
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
      self.shape_ids[env_ids] = _BOX
      self.sizes[env_ids] = self._base_sizes[_BOX]
      return

    shape_ids = self._sample_shape_ids(count, stage, env_ids.device)
    self.shape_ids[env_ids] = shape_ids
    self.sizes[env_ids] = self._sample_sizes(shape_ids, stage)

  @staticmethod
  def _sample_shape_ids(
    count: int,
    stage: int,
    device: torch.device | str,
  ) -> torch.Tensor:
    probabilities = torch.tensor(
      primitive_shape_probabilities(stage),
      device=device,
      dtype=torch.float32,
    )
    return torch.multinomial(probabilities, count, replacement=True)

  def _sample_sizes(
    self,
    shape_ids: torch.Tensor,
    stage: int,
  ) -> torch.Tensor:
    count = len(shape_ids)
    fraction = primitive_randomization_fraction(stage)
    minimum = torch.full((count, 1), BOX_SCALE_RANGE[0], device=shape_ids.device)
    maximum = torch.full((count, 1), BOX_SCALE_RANGE[1], device=shape_ids.device)
    capsule = shape_ids == _CAPSULE
    sphere = shape_ids == _SPHERE
    minimum[capsule] = CAPSULE_SCALE_RANGE[0]
    maximum[capsule] = CAPSULE_SCALE_RANGE[1]
    minimum[sphere] = SPHERE_SCALE_RANGE[0]
    maximum[sphere] = SPHERE_SCALE_RANGE[1]
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
  """Reset randomly oriented mesh variants at a height above the tabletop."""

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
