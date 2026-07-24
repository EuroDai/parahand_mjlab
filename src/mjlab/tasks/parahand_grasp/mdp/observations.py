from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
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


def contact_force_magnitude(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Magnitude of each primary's net contact force."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return torch.linalg.vector_norm(force, dim=-1)


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
  """Sampled object surface points expressed in the robot base frame."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._pool_size = int(cfg.params.get("pool_size", 256))
    self._sample_size = int(cfg.params.get("sample_size", 64))
    if self._sample_size > self._pool_size:
      raise ValueError(
        f"sample_size ({self._sample_size}) must not exceed "
        f"pool_size ({self._pool_size})."
      )

    object_name = cfg.params.get("object_name")
    if not isinstance(object_name, str):
      raise TypeError("object_point_cloud_b requires an 'object_name'.")
    curriculum_event_name = cfg.params.get("curriculum_event_name")
    if not isinstance(curriculum_event_name, str):
      raise TypeError("object_point_cloud_b requires a 'curriculum_event_name'.")
    self._curriculum_event_cfg = env.event_manager.get_term_cfg(curriculum_event_name)
    self._dynamic_sampling_stage = int(cfg.params.get("dynamic_sampling_stage", 2))
    self._cache_for_visualization = bool(
      cfg.params.get("cache_for_visualization", False)
    )
    self._latest_points_w: torch.Tensor | None = None
    obj: Entity = env.scene[object_name]
    variant_ids = env.sim.world_to_variant.get(object_name)
    if variant_ids is None:
      raise ValueError(f"Entity '{object_name}' must use VariantEntityCfg.")
    self._variant_ids = variant_ids.to(device=env.device, dtype=torch.long)

    metadata = obj.variant_metadata
    if metadata is None:
      raise ValueError(f"Entity '{object_name}' has no variant metadata.")
    primitive_by_name = {primitive.name: primitive for primitive in PRIMITIVE_OBJECTS}
    point_pools = []
    for variant_idx, variant_name in enumerate(metadata.variant_names):
      primitive = primitive_by_name.get(variant_name)
      if primitive is not None:
        geom_size = torch.tensor(primitive.size, device=env.device).reshape(1, 3)
        points = _sample_primitive_surface_points(
          primitive.geom_type.value,
          geom_size,
          self._pool_size,
        ).squeeze(0)
        geom_quat = torch.tensor(
          primitive.geom_quat,
          device=env.device,
          dtype=points.dtype,
        ).expand(self._pool_size, -1)
        point_pools.append(quat_apply(geom_quat, points))
      else:
        source_model = metadata.variant_source_specs[variant_idx].compile()
        point_pools.append(
          _sample_model_surface_points(source_model, self._pool_size, env.device)
        )
    self._points_local = torch.stack(point_pools)
    self._fixed_sample_ids = torch.arange(
      self._sample_size, device=env.device, dtype=torch.long
    ).expand(env.num_envs, -1)
    self._cached_sample_ids = self._fixed_sample_ids.clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    ref_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    pool_size: int = 256,
    sample_size: int = 64,
    flatten: bool = True,
    curriculum_event_name: str = "reset_object_pose",
    dynamic_sampling_stage: int = 2,
    cache_for_visualization: bool = False,
  ) -> torch.Tensor:
    del curriculum_event_name
    if pool_size != self._pool_size or sample_size != self._sample_size:
      raise ValueError(
        "object_point_cloud_b pool_size and sample_size cannot change after "
        "initialization."
      )
    if dynamic_sampling_stage != self._dynamic_sampling_stage:
      raise ValueError(
        "object_point_cloud_b dynamic_sampling_stage cannot change after "
        "initialization."
      )
    if cache_for_visualization != self._cache_for_visualization:
      raise ValueError(
        "object_point_cloud_b cache_for_visualization cannot change after "
        "initialization."
      )

    obj: Entity = env.scene[object_name]
    ref_asset: Entity = env.scene[ref_asset_cfg.name]

    curriculum_stage = int(self._curriculum_event_cfg.params["curriculum_stage"])
    if curriculum_stage >= self._dynamic_sampling_stage:
      sample_ids = self._draw_sample_ids(env.num_envs, env.device)
    else:
      sample_ids = self._cached_sample_ids
    points_local = torch.gather(
      self._points_local[self._variant_ids],
      dim=1,
      index=sample_ids.unsqueeze(-1).expand(-1, -1, 3),
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
    curriculum_stage = int(self._curriculum_event_cfg.params["curriculum_stage"])
    if curriculum_stage == 0:
      self._cached_sample_ids[env_ids] = self._fixed_sample_ids[env_ids]
    else:
      num_envs = self._cached_sample_ids[env_ids].shape[0]
      self._cached_sample_ids[env_ids] = self._draw_sample_ids(
        num_envs, self._cached_sample_ids.device
      )

  def _draw_sample_ids(
    self,
    num_envs: int,
    device: str | torch.device,
  ) -> torch.Tensor:
    random_scores = torch.rand(
      num_envs,
      self._pool_size,
      dtype=self._points_local.dtype,
      device=device,
    )
    return random_scores.topk(self._sample_size, dim=1).indices


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

  geom_choices = torch.multinomial(areas, num_points, replacement=True)
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
    face_ids = torch.multinomial(face_areas, num_points, replacement=True)
    triangles = vertices[faces[face_ids]]
    barycentric = torch.rand(num_points, 2, device=device)
    sqrt_u = torch.sqrt(barycentric[:, :1])
    return (
      (1.0 - sqrt_u) * triangles[:, 0]
      + sqrt_u * (1.0 - barycentric[:, 1:]) * triangles[:, 1]
      + sqrt_u * barycentric[:, 1:] * triangles[:, 2]
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
  face_weights = torch.stack(
    (
      half_sizes[:, 1] * half_sizes[:, 2],
      half_sizes[:, 0] * half_sizes[:, 2],
      half_sizes[:, 0] * half_sizes[:, 1],
    ),
    dim=-1,
  )
  face_axes = torch.multinomial(face_weights, num_points, replacement=True)
  points = (
    torch.rand(num_envs, num_points, 3, device=half_sizes.device) * 2.0 - 1.0
  ) * half_sizes[:, None, :]
  face_signs = torch.where(
    torch.rand(num_envs, num_points, device=half_sizes.device) < 0.5,
    -1.0,
    1.0,
  )
  face_extents = torch.gather(half_sizes, dim=1, index=face_axes)
  return points.scatter(
    dim=2,
    index=face_axes.unsqueeze(-1),
    src=(face_signs * face_extents).unsqueeze(-1),
  )


def _sample_sphere_surface_points(
  geom_sizes: torch.Tensor,
  num_points: int,
) -> torch.Tensor:
  directions = torch.randn(geom_sizes.shape[0], num_points, 3, device=geom_sizes.device)
  directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
  return directions * geom_sizes[:, None, :1]


def _sample_capsule_surface_points(
  geom_sizes: torch.Tensor,
  num_points: int,
) -> torch.Tensor:
  radius = geom_sizes[:, 0:1]
  half_length = geom_sizes[:, 1:2]
  theta = torch.rand(geom_sizes.shape[0], num_points, device=geom_sizes.device) * (
    2.0 * torch.pi
  )
  side_prob = half_length / (half_length + radius).clamp_min(1.0e-6)
  on_side = torch.rand_like(theta) < side_prob

  side_x = radius * torch.cos(theta)
  side_y = radius * torch.sin(theta)
  side_z = (torch.rand_like(theta) * 2.0 - 1.0) * half_length

  cap_z_unit = torch.rand_like(theta)
  cap_radial = radius * torch.sqrt((1.0 - cap_z_unit.square()).clamp_min(0.0))
  cap_sign = torch.where(torch.rand_like(theta) < 0.5, -1.0, 1.0)
  cap_x = cap_radial * torch.cos(theta)
  cap_y = cap_radial * torch.sin(theta)
  cap_z = cap_sign * (half_length + radius * cap_z_unit)

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
  theta = torch.rand(geom_sizes.shape[0], num_points, device=geom_sizes.device) * (
    2.0 * torch.pi
  )
  side_prob = (2.0 * half_length) / (2.0 * half_length + radius).clamp_min(1.0e-6)
  on_side = torch.rand_like(theta) < side_prob

  side_x = radius * torch.cos(theta)
  side_y = radius * torch.sin(theta)
  side_z = (torch.rand_like(theta) * 2.0 - 1.0) * half_length

  cap_radial = radius * torch.sqrt(torch.rand_like(theta))
  cap_sign = torch.where(torch.rand_like(theta) < 0.5, -1.0, 1.0)
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
