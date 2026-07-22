from __future__ import annotations

from functools import partial

import mujoco
import numpy as np
import trimesh

from mjlab.asset_zoo.robots import get_parahand_robot_cfg
from mjlab.entity import EntityCfg, VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.parahand_grasp.grasp_object_env_cfg import (
  make_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.consts import PRIMITIVE_OBJECTS, PrimitiveObject


def _make_variant_collision_mesh(obj: PrimitiveObject) -> trimesh.Trimesh:
  """Build the mesh proxy required for mixed-shape batched simulation."""
  if obj.geom_type == mujoco.mjtGeom.mjGEOM_BOX:
    return trimesh.creation.box(extents=tuple(2.0 * value for value in obj.size))
  if obj.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
    return trimesh.creation.icosphere(subdivisions=2, radius=obj.size[0])
  if obj.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return trimesh.creation.capsule(
      height=2.0 * obj.size[1], radius=obj.size[0], count=(12, 12)
    )
  raise ValueError(f"Unsupported primitive type: {obj.geom_type.name}")


def get_object_spec(obj: PrimitiveObject) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  mesh = _make_variant_collision_mesh(obj)
  mesh.apply_translation(-mesh.bounding_box.centroid)
  spec.add_mesh(
    name="object_mesh",
    uservert=np.asarray(mesh.vertices, dtype=np.float64).reshape(-1).tolist(),
    userface=np.asarray(mesh.faces, dtype=np.int32).reshape(-1).tolist(),
  )
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_freejoint")
  body.add_geom(
    name="object_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="object_mesh",
    quat=obj.geom_quat,
    density=500.0,
    rgba=obj.rgba,
    friction=(1.0, 0.1, 0.002),
    condim=4,
    solref=(0.02, 1.0),
    solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    contype=2_097_152,
    conaffinity=2_097_151,
  )
  body.add_site(name="object_center", pos=(0.0, 0.0, 0.0))
  return spec


def get_object_cfg() -> VariantEntityCfg:
  return VariantEntityCfg(
    variants={obj.name: partial(get_object_spec, obj=obj) for obj in PRIMITIVE_OBJECTS},
    init_state=EntityCfg.InitialStateCfg(
      pos=(-0.55, 0.0, 0.03),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={},
      joint_vel={},
    ),
  )


def parahand_grasp_object_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_grasp_object_env_cfg()
  cfg.scene.entities = {
    "robot": get_parahand_robot_cfg(),
    "object": get_object_cfg(),
  }
  cfg.scene.num_envs = 1 if play else 4096

  cfg.viewer.body_name = "palm"

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg
