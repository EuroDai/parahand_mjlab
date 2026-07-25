from __future__ import annotations

from copy import deepcopy
from functools import partial

import mujoco
import numpy as np
import trimesh

from mjlab.asset_zoo.robots import (
  PARAHAND_ONLY_ACTION_SCALE,
  get_parahand_only_robot_cfg,
  get_parahand_robot_cfg,
)
from mjlab.entity import EntityCfg, VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.tasks.parahand_grasp import mdp as parahand_mdp
from mjlab.tasks.parahand_grasp.grasp_object_env_cfg import (
  make_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.actions import (
  RelativeTendonLengthActionCfg,
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


def get_object_cfg(
  init_pos: tuple[float, float, float] = (-0.55, 0.0, 0.03),
) -> VariantEntityCfg:
  return VariantEntityCfg(
    variants={obj.name: partial(get_object_spec, obj=obj) for obj in PRIMITIVE_OBJECTS},
    init_state=EntityCfg.InitialStateCfg(
      pos=init_pos,
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={},
      joint_vel={},
    ),
  )


def _enable_object_point_cloud_debug_vis(cfg: ManagerBasedRlEnvCfg) -> None:
  """Enable task-local point-cloud visualization for play environments."""
  actor_group = cfg.observations["actor"]
  point_cloud_cfg = deepcopy(actor_group.terms["object_point_cloud_b"])
  point_cloud_cfg.params["cache_for_visualization"] = True
  actor_group.terms["object_point_cloud_b"] = point_cloud_cfg
  cfg.rewards["object_point_cloud_debug"] = RewardTermCfg(
    func=parahand_mdp.object_point_cloud_debug_visualizer,
    weight=0.0,
    params={
      "observation_group": "actor",
      "observation_term": "object_point_cloud_b",
      "radius": 0.003,
      "color": (0.0, 0.8, 1.0, 0.9),
    },
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
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    _enable_object_point_cloud_debug_vis(cfg)

  return cfg


def parahand_only_grasp_object_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_grasp_object_env_cfg()
  cfg.scene.entities = {
    "robot": get_parahand_only_robot_cfg(),
    "object": get_object_cfg(init_pos=(0.3, 0.1, 0.03)),
  }
  cfg.scene.num_envs = 1 if play else 4096

  joint_actuator_names = tuple(
    name
    for name in PARAHAND_ONLY_ACTION_SCALE
    if name not in ("index_tendon", "middle_tendon", "ring_tendon", "little_tendon")
  )
  cfg.actions["joint_pos"] = parahand_mdp.ParaHandRelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=joint_actuator_names,
    scale={name: PARAHAND_ONLY_ACTION_SCALE[name] for name in joint_actuator_names},
    preserve_order=True,
    raw_action_limit=1.0,
    coupled_finger_actuator_names=(
      "index_mcp_1",
      "middle_mcp_1",
      "ring_mcp_1",
      "little_mcp_1",
    ),
  )
  tendon_action = cfg.actions["tendon_length"]
  assert isinstance(tendon_action, RelativeTendonLengthActionCfg)
  tendon_action.scale = PARAHAND_ONLY_ACTION_SCALE["index_tendon"]

  reset_joints_cfg = cfg.events["reset_robot_joints"].params["asset_cfg"]
  reset_joints_cfg.joint_names = joint_actuator_names
  reset_joints_cfg.preserve_order = True
  cfg.events["reset_object_pose"].params["position_center"] = (0, 0)

  command_cfg = cfg.commands["object_pose"]
  assert isinstance(command_cfg, LiftingCommandCfg)
  target_range = command_cfg.target_position_range
  target_range.x = (-0.1, 0.1)
  target_range.y = (-0.1, 0.1)
  target_range.z = (0.35, 0.55)

  fingertip_body_names = (
    "thumb_dp",
    "index_dp",
    "middle_dp",
    "ring_dp",
    "little_dp",
  )
  for group_cfg in cfg.observations.values():
    fingertip_quat_cfg = group_cfg.terms["fingertip_quat_b"]
    fingertip_quat_cfg.params["body_asset_cfg"].body_names = fingertip_body_names
    fingertip_quat_cfg.params["body_asset_cfg"].preserve_order = True

  cfg.viewer.body_name = "palm"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    _enable_object_point_cloud_debug_vis(cfg)

  return cfg
