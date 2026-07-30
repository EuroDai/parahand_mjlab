from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.tasks.parahand_grasp.mdp.consts import PRIMITIVE_OBJECTS
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_position(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected absolute joint positions."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def body_quaternion_b(
  env: ManagerBasedRlEnv,
  body_asset_cfg: SceneEntityCfg,
  base_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected body quaternions in the robot base frame."""
  body_asset: Entity = env.scene[body_asset_cfg.name]
  base_asset: Entity = env.scene[base_asset_cfg.name]
  body_quat_w = body_asset.data.body_link_quat_w[:, body_asset_cfg.body_ids]
  num_bodies = body_quat_w.shape[1]
  base_quat_w = base_asset.data.root_link_quat_w[:, None, :].expand(-1, num_bodies, -1)
  value = quat_mul(quat_inv(base_quat_w), body_quat_w)
  return value.reshape(env.num_envs, -1)


def site_position_b(
  env: ManagerBasedRlEnv,
  site_asset_cfg: SceneEntityCfg,
  base_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected site positions in the robot base frame."""
  site_asset: Entity = env.scene[site_asset_cfg.name]
  base_asset: Entity = env.scene[base_asset_cfg.name]
  site_pos_w = site_asset.data.site_pos_w[:, site_asset_cfg.site_ids]
  num_sites = site_pos_w.shape[1]
  base_pos_w = base_asset.data.root_link_pos_w[:, None, :]
  base_quat_w_inv = quat_inv(base_asset.data.root_link_quat_w)[:, None, :].expand(
    -1, num_sites, -1
  )
  value = quat_apply(base_quat_w_inv, site_pos_w - base_pos_w)
  return value.reshape(env.num_envs, -1)


def object_position_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """机器人坐标系下的物体位置"""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  object_pos_rel_w = obj.data.root_link_pos_w - robot.data.root_link_pos_w
  return quat_apply(quat_inv(robot.data.root_link_quat_w), object_pos_rel_w)


def object_to_palm_position_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  palm_site_cfg: SceneEntityCfg,
  base_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vector from the palm site to the object center in robot-base axes."""
  obj: Entity = env.scene[object_name]
  palm_asset: Entity = env.scene[palm_site_cfg.name]
  base_asset: Entity = env.scene[base_asset_cfg.name]
  palm_pos_w = palm_asset.data.site_pos_w[:, palm_site_cfg.site_ids]
  if palm_pos_w.shape[1] != 1:
    raise ValueError("object_to_palm_position_b requires exactly one palm site.")
  difference_w = obj.data.root_link_pos_w - palm_pos_w[:, 0]
  return quat_apply(quat_inv(base_asset.data.root_link_quat_w), difference_w)


def object_quaternion_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Object orientation quaternion in the robot base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  return quat_mul(quat_inv(robot.data.root_link_quat_w), obj.data.root_link_quat_w)


def target_position_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """机器人坐标系下的目标位置"""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  robot: Entity = env.scene[asset_cfg.name]
  target_pos_rel_w = command.target_pos - robot.data.root_link_pos_w
  return quat_apply(quat_inv(robot.data.root_link_quat_w), target_pos_rel_w)


def contact_force_b(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return signed net fingertip-contact forces in the robot base frame."""
  sensor: ContactSensor = env.scene[sensor_name]
  robot: Entity = env.scene[asset_cfg.name]
  force_w = sensor.data.force
  assert force_w is not None
  num_fingertips = force_w.shape[1]
  base_quat_w_inv = quat_inv(robot.data.root_link_quat_w)[:, None, :].expand(
    -1, num_fingertips, -1
  )
  force_b = quat_apply(base_quat_w_inv, force_w)
  return force_b.flatten(start_dim=1)


def contact_force_magnitude(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  fingertip_name: str | None = None,
) -> torch.Tensor:
  """Return net contact-force magnitudes for all fingertips or one named fingertip."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  force_magnitude = torch.linalg.vector_norm(force, dim=-1)
  if fingertip_name is None:
    return force_magnitude
  try:
    fingertip_index = sensor.primary_names.index(fingertip_name)
  except ValueError as exc:
    raise ValueError(
      f"Fingertip '{fingertip_name}' is not a primary of contact sensor "
      f"'{sensor_name}'. Available primaries: {sensor.primary_names}."
    ) from exc
  return force_magnitude[:, fingertip_index]


def tendon_length(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected tendon lengths."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.tendon_len[:, asset_cfg.tendon_ids]


def last_actions(
  env: ManagerBasedRlEnv,
  action_names: tuple[str, ...],
) -> torch.Tensor:
  """Concatenate clipped actions from the selected action terms."""
  return torch.cat(
    [env.action_manager.get_term(name).raw_action for name in action_names], dim=-1
  )


class object_point_cloud_b:
  """Fixed object surface points expressed in the robot base frame.

  The local surface cloud is built once from the scene's initial object shape.
  Subsequent observations apply only the object's rigid rotation and translation.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._pool_size = int(cfg.params.get("pool_size", 256))
    self._sample_size = int(cfg.params.get("sample_size", 256))
    if self._sample_size != self._pool_size:
      raise ValueError(
        "Fixed point clouds require sample_size to equal pool_size, "
        f"got sample_size={self._sample_size} and pool_size={self._pool_size}."
      )

    object_name = cfg.params.get("object_name")
    if not isinstance(object_name, str):
      raise TypeError("object_point_cloud_b requires an 'object_name'.")
    curriculum_event_name = cfg.params.get("curriculum_event_name")
    if not isinstance(curriculum_event_name, str):
      raise TypeError("object_point_cloud_b requires a 'curriculum_event_name'.")
    self._curriculum_event_cfg = env.event_manager.get_term_cfg(curriculum_event_name)
    self._cache_for_visualization = bool(
      cfg.params.get("cache_for_visualization", False)
    )
    self._latest_points_w: torch.Tensor | None = None
    obj: Entity = env.scene[object_name]
    reset_term = self._curriculum_event_cfg.func
    self._primitive_reset = (
      reset_term
      if hasattr(reset_term, "shape_ids") and hasattr(reset_term, "sizes")
      else None
    )
    metadata = obj.variant_metadata
    if self._primitive_reset is not None:
      self._variant_ids = None
      self._points_local = torch.empty(
        env.num_envs, self._pool_size, 3, device=env.device
      )
      self._refresh_primitive_points(
        torch.arange(env.num_envs, device=env.device, dtype=torch.long)
      )
    else:
      variant_ids = env.sim.world_to_variant.get(object_name)
      if variant_ids is None or metadata is None:
        raise ValueError(
          f"Entity '{object_name}' must use analytic primitive reset state "
          "or VariantEntityCfg mesh assets."
        )
      self._variant_ids = variant_ids.to(device=env.device, dtype=torch.long)
      point_paths = cfg.params.get("variant_point_cloud_paths")
      point_scales = cfg.params.get("variant_point_cloud_scales")
      if point_paths is None and point_scales is None:
        point_pools = []
        for variant_idx in range(len(metadata.variant_names)):
          source_model = metadata.variant_source_specs[variant_idx].compile()
          point_pools.append(
            _sample_model_surface_points(source_model, self._pool_size, env.device)
          )
      elif point_paths is None or point_scales is None:
        raise ValueError(
          "variant_point_cloud_paths and variant_point_cloud_scales must be "
          "provided together."
        )
      else:
        if len(point_paths) != len(metadata.variant_names) or len(point_scales) != len(
          metadata.variant_names
        ):
          raise ValueError(
            "Preprocessed point-cloud paths and scales must align with mesh variants."
          )
        point_pools = [
          _load_preprocessed_point_cloud(
            Path(path),
            float(scale),
            self._pool_size,
            env.device,
          )
          for path, scale in zip(point_paths, point_scales, strict=True)
        ]
      self._points_local = torch.stack(point_pools)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    ref_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    pool_size: int = 256,
    sample_size: int = 256,
    flatten: bool = True,
    curriculum_event_name: str = "reset_object_pose",
    cache_for_visualization: bool = False,
    variant_point_cloud_paths: tuple[str, ...] | None = None,
    variant_point_cloud_scales: tuple[float, ...] | None = None,
  ) -> torch.Tensor:
    del (
      curriculum_event_name,
      variant_point_cloud_paths,
      variant_point_cloud_scales,
    )
    if pool_size != self._pool_size or sample_size != self._sample_size:
      raise ValueError(
        "object_point_cloud_b pool_size and sample_size cannot change after "
        "initialization."
      )
    if cache_for_visualization != self._cache_for_visualization:
      raise ValueError(
        "object_point_cloud_b cache_for_visualization cannot change after "
        "initialization."
      )

    obj: Entity = env.scene[object_name]
    ref_asset: Entity = env.scene[ref_asset_cfg.name]

    points_local = (
      self._points_local
      if self._variant_ids is None
      else self._points_local[self._variant_ids]
    )

    object_quat_w = obj.data.root_link_quat_w[:, None, :].expand(-1, sample_size, -1)
    points_w = (
      quat_apply(object_quat_w, points_local) + obj.data.root_link_pos_w[:, None, :]
    )
    if self._cache_for_visualization:
      self._latest_points_w = points_w
    ref_quat_w_inv = quat_inv(ref_asset.data.root_link_quat_w)[:, None, :].expand(
      -1, sample_size, -1
    )
    points_b = quat_apply(
      ref_quat_w_inv,
      points_w - ref_asset.data.root_link_pos_w[:, None, :],
    )
    return points_b.reshape(env.num_envs, -1) if flatten else points_b

  @property
  def latest_points_w(self) -> torch.Tensor | None:
    """Latest sampled surface points in world coordinates for play visualization."""
    return self._latest_points_w

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if getattr(self, "_primitive_reset", None) is not None:
      refresh_ids = torch.arange(
        self._points_local.shape[0],
        device=self._points_local.device,
        dtype=torch.long,
      )[env_ids]
      self._refresh_primitive_points(refresh_ids)

  def _refresh_primitive_points(self, env_ids: torch.Tensor) -> None:
    primitive_reset = getattr(self, "_primitive_reset", None)
    if primitive_reset is None or len(env_ids) == 0:
      return
    shape_ids = primitive_reset.shape_ids[env_ids]
    sizes = primitive_reset.sizes[env_ids]
    for shape_id, primitive in enumerate(PRIMITIVE_OBJECTS):
      mask = shape_ids == shape_id
      if not mask.any():
        continue
      points = _sample_primitive_surface_points(
        primitive.geom_type.value,
        sizes[mask],
        self._pool_size,
      )
      geom_quat = torch.tensor(
        primitive.geom_quat,
        device=points.device,
        dtype=points.dtype,
      ).expand(points.shape[0], self._pool_size, -1)
      self._points_local[env_ids[mask]] = quat_apply(geom_quat, points)


def _sample_primitive_surface_points(
  geom_type: int,
  geom_sizes: torch.Tensor,
  num_points: int,
) -> torch.Tensor:
  if geom_type == mujoco.mjtGeom.mjGEOM_BOX.value:
    return _sample_box_surface_points(geom_sizes, num_points)
  if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE.value:
    return _sample_sphere_surface_points(geom_sizes, num_points)
  if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE.value:
    return _sample_capsule_surface_points(geom_sizes, num_points)
  if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER.value:
    return _sample_cylinder_surface_points(geom_sizes, num_points)
  geom_name = mujoco.mjtGeom(geom_type).name
  raise ValueError(f"Unsupported point-cloud geom type: {geom_name}")


def _load_preprocessed_point_cloud(
  path: Path,
  scale: float,
  num_points: int,
  device: str,
) -> torch.Tensor:
  points = np.load(path, allow_pickle=False)
  if points.ndim != 2 or points.shape[1] != 3 or len(points) < num_points:
    raise ValueError(
      f"Expected at least {num_points} preprocessed 3D points in {path}, "
      f"got shape {points.shape}."
    )
  if not np.isfinite(points).all():
    raise ValueError(f"Preprocessed point cloud contains non-finite values: {path}")
  if len(points) > num_points:
    # Preserve the coverage of the uniformly preprocessed source cloud without
    # introducing per-frame or per-reset randomness.
    indices = (
      (np.arange(num_points, dtype=np.float64) + 0.5) * (len(points) / num_points)
    ).astype(np.int64)
    points = points[indices]
  return (
    torch.as_tensor(
      points,
      dtype=torch.float32,
      device=device,
    )
    * scale
  )


def _sample_model_surface_points(
  model: mujoco.MjModel,
  num_points: int,
  device: str,
) -> torch.Tensor:
  geom_ids = [
    geom_id
    for geom_id in range(model.ngeom)
    if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0
  ]
  if not geom_ids:
    geom_ids = list(range(model.ngeom))
  if not geom_ids:
    raise ValueError("Object point-cloud source model contains no geoms.")

  areas = torch.tensor(
    [_geom_surface_area(model, geom_id) for geom_id in geom_ids],
    dtype=torch.float32,
    device=device,
  )
  if not torch.isfinite(areas).all() or areas.sum() <= 0.0:
    raise ValueError("Object point-cloud source model has invalid surface area.")

  geom_cdf = torch.cumsum(areas / areas.sum(), dim=0)
  geom_choices = torch.searchsorted(
    geom_cdf, _stratified_unit(num_points, device, areas.dtype)
  ).clamp_max(len(geom_ids) - 1)
  points = torch.empty(num_points, 3, dtype=torch.float32, device=device)
  for choice, geom_id in enumerate(geom_ids):
    mask = geom_choices == choice
    count = int(mask.sum().item())
    if count == 0:
      continue
    geom_points = _sample_geom_surface_points(model, geom_id, count, device)
    geom_quat = torch.tensor(
      model.geom_quat[geom_id], dtype=torch.float32, device=device
    ).expand(count, -1)
    geom_pos = torch.tensor(model.geom_pos[geom_id], dtype=torch.float32, device=device)
    points[mask] = quat_apply(geom_quat, geom_points) + geom_pos
  return points


def _geom_surface_area(model: mujoco.MjModel, geom_id: int) -> float:
  geom_type = mujoco.mjtGeom(model.geom_type[geom_id])
  size = model.geom_size[geom_id]
  if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
    return float(8.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]))
  if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
    return float(4.0 * torch.pi * size[0] ** 2)
  if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return float(4.0 * torch.pi * size[0] * (size[1] + size[0]))
  if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
    return float(2.0 * torch.pi * size[0] * (2.0 * size[1] + size[0]))
  if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
    _, _, face_areas = _mesh_triangles(model, geom_id, "cpu")
    return float(face_areas.sum().item())
  raise ValueError(f"Unsupported point-cloud geom type: {geom_type.name}")


def _sample_geom_surface_points(
  model: mujoco.MjModel,
  geom_id: int,
  num_points: int,
  device: str,
) -> torch.Tensor:
  geom_type = mujoco.mjtGeom(model.geom_type[geom_id])
  if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
    vertices, faces, face_areas = _mesh_triangles(model, geom_id, device)
    face_cdf = torch.cumsum(face_areas / face_areas.sum(), dim=0)
    unit = _stratified_unit(num_points, device, face_areas.dtype)
    face_ids = torch.searchsorted(face_cdf, unit).clamp_max(len(face_areas) - 1)
    triangles = vertices[faces[face_ids]]
    sqrt_u = torch.sqrt(_quasi_unit(num_points, device, face_areas.dtype)[:, None])
    point_ids = torch.arange(num_points, device=device, dtype=face_areas.dtype) + 0.5
    barycentric_v = torch.remainder(point_ids * 0.7548776662466927, 1.0)[:, None]
    return (
      (1.0 - sqrt_u) * triangles[:, 0]
      + sqrt_u * (1.0 - barycentric_v) * triangles[:, 1]
      + sqrt_u * barycentric_v * triangles[:, 2]
    )

  geom_sizes = torch.tensor(
    model.geom_size[geom_id], dtype=torch.float32, device=device
  ).reshape(1, 3)
  return _sample_primitive_surface_points(
    geom_type.value, geom_sizes, num_points
  ).squeeze(0)


def _mesh_triangles(
  model: mujoco.MjModel,
  geom_id: int,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  mesh_id = int(model.geom_dataid[geom_id])
  if mesh_id < 0:
    raise ValueError(f"Mesh geom {geom_id} has no mesh data.")
  vert_adr = int(model.mesh_vertadr[mesh_id])
  vert_num = int(model.mesh_vertnum[mesh_id])
  face_adr = int(model.mesh_faceadr[mesh_id])
  face_num = int(model.mesh_facenum[mesh_id])
  vertices = torch.tensor(
    model.mesh_vert[vert_adr : vert_adr + vert_num],
    dtype=torch.float32,
    device=device,
  )
  faces = torch.tensor(
    model.mesh_face[face_adr : face_adr + face_num],
    dtype=torch.long,
    device=device,
  )
  triangles = vertices[faces]
  face_areas = 0.5 * torch.linalg.vector_norm(
    torch.linalg.cross(
      triangles[:, 1] - triangles[:, 0],
      triangles[:, 2] - triangles[:, 0],
    ),
    dim=-1,
  )
  if face_areas.sum() <= 0.0:
    raise ValueError(f"Mesh geom {geom_id} has no non-degenerate faces.")
  return vertices, faces, face_areas


def _sample_box_surface_points(
  half_sizes: torch.Tensor,
  num_points: int,
) -> torch.Tensor:
  num_envs = half_sizes.shape[0]
  axis_weights = torch.stack(
    (
      half_sizes[:, 1] * half_sizes[:, 2],
      half_sizes[:, 0] * half_sizes[:, 2],
      half_sizes[:, 0] * half_sizes[:, 1],
    ),
    dim=-1,
  )
  face_weights = axis_weights[:, :, None].expand(-1, -1, 2).reshape(num_envs, 6)
  face_cdf = torch.cumsum(
    face_weights / face_weights.sum(dim=-1, keepdim=True), dim=-1
  )
  unit = _stratified_unit(num_points, half_sizes.device, half_sizes.dtype)
  face_ids = (unit[None, :, None] > face_cdf[:, None, :]).sum(dim=-1).clamp_max(5)
  face_membership = torch.nn.functional.one_hot(face_ids, num_classes=6)
  local_ids = (
    torch.cumsum(face_membership, dim=1).gather(2, face_ids.unsqueeze(-1)).squeeze(-1)
    - 1
  )
  face_counts = face_membership.sum(dim=1).gather(1, face_ids).clamp_min(1)
  first_coord = (local_ids.to(half_sizes.dtype) + 0.5) / face_counts
  second_coord = torch.remainder(
    _radical_inverse_base2(local_ids, half_sizes.dtype) + 0.5 / face_counts,
    1.0,
  )
  face_axes = torch.div(face_ids, 2, rounding_mode="floor")
  face_signs = face_ids.remainder(2).to(half_sizes.dtype) * 2.0 - 1.0

  points = torch.zeros(
    num_envs,
    num_points,
    3,
    device=half_sizes.device,
    dtype=half_sizes.dtype,
  )
  for axis in range(3):
    first_tangent = (axis + 1) % 3
    second_tangent = (axis + 2) % 3
    mask = face_axes == axis
    points[:, :, axis] = torch.where(
      mask, face_signs * half_sizes[:, axis, None], points[:, :, axis]
    )
    points[:, :, first_tangent] = torch.where(
      mask,
      (first_coord * 2.0 - 1.0) * half_sizes[:, first_tangent, None],
      points[:, :, first_tangent],
    )
    points[:, :, second_tangent] = torch.where(
      mask,
      (second_coord * 2.0 - 1.0) * half_sizes[:, second_tangent, None],
      points[:, :, second_tangent],
    )
  return points


def _radical_inverse_base2(
  indices: torch.Tensor,
  dtype: torch.dtype,
) -> torch.Tensor:
  """Return the base-two radical inverse used by a Hammersley point set."""
  remaining = indices
  result = torch.zeros_like(indices, dtype=dtype)
  factor = 0.5
  while torch.any(remaining > 0):
    result += remaining.remainder(2).to(dtype) * factor
    remaining = torch.div(remaining, 2, rounding_mode="floor")
    factor *= 0.5
  return result


def _sample_sphere_surface_points(
  geom_sizes: torch.Tensor,
  num_points: int,
) -> torch.Tensor:
  unit = _stratified_unit(num_points, geom_sizes.device, geom_sizes.dtype)
  z = 1.0 - 2.0 * unit
  theta = _golden_angles(num_points, geom_sizes.device, geom_sizes.dtype)
  radial = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
  directions = torch.stack(
    (radial * torch.cos(theta), radial * torch.sin(theta), z), dim=-1
  )
  return directions * geom_sizes[:, None, :1]


def _sample_capsule_surface_points(
  geom_sizes: torch.Tensor,
  num_points: int,
) -> torch.Tensor:
  radius = geom_sizes[:, 0:1]
  half_length = geom_sizes[:, 1:2]
  unit = _stratified_unit(num_points, geom_sizes.device, geom_sizes.dtype)
  theta = _golden_angles(num_points, geom_sizes.device, geom_sizes.dtype)
  side_prob = half_length / (half_length + radius).clamp_min(1.0e-6)
  on_side = unit[None, :] < side_prob

  side_x = radius * torch.cos(theta)
  side_y = radius * torch.sin(theta)
  side_unit = (unit[None, :] / side_prob.clamp_min(1.0e-6)).clamp_max(1.0)
  side_z = (side_unit * 2.0 - 1.0) * half_length

  cap_unit = ((unit[None, :] - side_prob) / (1.0 - side_prob).clamp_min(1.0e-6)).clamp(
    0.0, 1.0
  )
  cap_z_unit = 1.0 - 2.0 * cap_unit
  cap_radial = radius * torch.sqrt((1.0 - cap_z_unit.square()).clamp_min(0.0))
  cap_sign = torch.where(cap_z_unit < 0.0, -1.0, 1.0)
  cap_x = cap_radial * torch.cos(theta)
  cap_y = cap_radial * torch.sin(theta)
  cap_z = cap_sign * half_length + radius * cap_z_unit

  return torch.stack(
    (
      torch.where(on_side, side_x, cap_x),
      torch.where(on_side, side_y, cap_y),
      torch.where(on_side, side_z, cap_z),
    ),
    dim=-1,
  )


def _sample_cylinder_surface_points(
  geom_sizes: torch.Tensor,
  num_points: int,
) -> torch.Tensor:
  radius = geom_sizes[:, 0:1]
  half_length = geom_sizes[:, 1:2]
  unit = _stratified_unit(num_points, geom_sizes.device, geom_sizes.dtype)
  theta = _golden_angles(num_points, geom_sizes.device, geom_sizes.dtype)
  side_prob = (2.0 * half_length) / (2.0 * half_length + radius).clamp_min(1.0e-6)
  on_side = unit[None, :] < side_prob

  side_x = radius * torch.cos(theta)
  side_y = radius * torch.sin(theta)
  side_unit = (unit[None, :] / side_prob.clamp_min(1.0e-6)).clamp_max(1.0)
  side_z = (side_unit * 2.0 - 1.0) * half_length

  cap_unit = ((unit[None, :] - side_prob) / (1.0 - side_prob).clamp_min(1.0e-6)).clamp(
    0.0, 1.0
  )
  cap_radial = radius * torch.sqrt(torch.remainder(cap_unit * 2.0, 1.0))
  cap_sign = torch.where(cap_unit < 0.5, -1.0, 1.0)
  cap_x = cap_radial * torch.cos(theta)
  cap_y = cap_radial * torch.sin(theta)
  cap_z = cap_sign * half_length

  return torch.stack(
    (
      torch.where(on_side, side_x, cap_x),
      torch.where(on_side, side_y, cap_y),
      torch.where(on_side, side_z, cap_z),
    ),
    dim=-1,
  )


def _stratified_unit(
  num_points: int,
  device: str | torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  return (torch.arange(num_points, device=device, dtype=dtype) + 0.5) / num_points


def _quasi_unit(
  num_points: int,
  device: str | torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  point_ids = torch.arange(num_points, device=device, dtype=dtype) + 0.5
  return torch.remainder(point_ids * 0.6180339887498949, 1.0)


def _golden_angles(
  num_points: int,
  device: str | torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  point_ids = torch.arange(num_points, device=device, dtype=dtype)
  return point_ids * 2.399963229728653


def camera_rgb(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """RGB observation in CNN-compatible format (B, C, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  rgb_data = sensor.data.rgb  # (B, H, W, 3)
  assert rgb_data is not None, f"Camera '{sensor_name}' has no RGB data"
  rgb_data = rgb_data.permute(0, 3, 1, 2)  # (B, 3, H, W)
  return rgb_data.float() / 255.0


def camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
) -> torch.Tensor:
  """Depth observation in CNN-compatible format (B, 1, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth  # (B, H, W, 1)
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)  # (B, 1, H, W)
  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  return torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)


def camera_segmentation(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Per-pixel typed segmentation in (B, 2, H, W) format."""
  sensor: CameraSensor = env.scene[sensor_name]
  seg_data = sensor.data.segmentation  # (B, H, W, 2)
  assert seg_data is not None, f"Camera '{sensor_name}' has no segmentation data"
  return seg_data.permute(0, 3, 1, 2)  # (B, 2, H, W)
