from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import mujoco

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
from mjlab.tasks.parahand_grasp.mdp.consts import PRIMITIVE_OBJECTS


def get_object_spec() -> mujoco.MjSpec:
  """Build the analytic primitive slots used by the grasp curriculum.

  All three geoms share one rigid body. The reset event moves inactive slots away
  from the body and continuously randomizes the active slot's dimensions.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_freejoint")
  for obj in PRIMITIVE_OBJECTS:
    body.add_geom(
      name=f"{obj.name}_geom",
      type=obj.geom_type,
      size=obj.size,
      quat=obj.geom_quat,
      density=500.0,
      rgba=obj.rgba,
      friction=(1.0, 0.002, 0.0001),
      condim=4,
      solref=(0.01, 1.0),
      solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
      contype=2_097_152,
      conaffinity=2_097_151,
    )
  body.add_site(name="object_center", pos=(0.0, 0.0, 0.0))
  return spec


def get_table_spec() -> mujoco.MjSpec:
  """Build the 0.8 m square analytic table used by all curriculum lessons."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="tabletop",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, -0.025),
    size=(0.4, 0.4, 0.025),
    rgba=(0.42, 0.30, 0.20, 1.0),
    friction=(1.0, 0.005, 0.0001),
    condim=4,
    solref=(0.01, 1.0),
    solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    contype=1,
    conaffinity=2_359_296,
  )
  return spec


def get_table_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_table_spec,
    # Keep the table out of the initial host-side contact solve. The first reset
    # moves it to its sampled 0.6--1.0 m height before policy observations.
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
  )


def get_object_cfg(
  init_pos: tuple[float, float, float] = (-0.55, 0.0, 0.03),
) -> EntityCfg:
  return EntityCfg(
    spec_fn=get_object_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=init_pos,
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={},
      joint_vel={},
    ),
  )


def get_mesh_object_cfg(
  variants: dict[str, Callable[[], mujoco.MjSpec]],
  init_pos: tuple[float, float, float] = (-0.55, 0.0, 0.03),
) -> VariantEntityCfg:
  """Build a mesh-variant object for future datasets such as YCB.

  Mesh entities keep the triangle-surface point-cloud path, while the built-in
  grasp curriculum uses :func:`get_object_cfg` and analytic primitives.
  """
  return VariantEntityCfg(
    variants=variants,
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
    "table": get_table_cfg(),
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
    "table": get_table_cfg(),
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
  cfg.events["reset_robot_joints"].func = parahand_mdp.reset_joints_above_table
  cfg.events["reset_robot_joints"].params.update(
    {
      "palm_height_joint_name": "palm_translation_z",
      "palm_height_range": (0.2, 0.4),
      "object_pose_event_name": "reset_object_pose",
      "tendon_action_name": "tendon_length",
      "palm_joint_ranges": {
        "palm_translation_x": (-0.1, 0.2),
        "palm_translation_y": (-0.2, 0.2),
        "palm_rotation_x": (-0.5, 0.5),
        "palm_rotation_y": (-0.5, 0.5),
        "palm_rotation_z": (-0.5, 0.5),
      },
    }
  )
  del cfg.events["reset_base"]
  cfg.events["reset_table_height"].params.update(
    {
      "robot_name": "robot",
      "robot_base_follows_table": True,
    }
  )
  cfg.events["reset_object_pose"].params["position_center"] = (0, 0)
  cfg.events = {
    name: cfg.events[name]
    for name in (
      "reset_gravity",
      "reset_table_height",
      "reset_object_pose",
      "reset_robot_joints",
      "reset_teacher_physics",
    )
  }

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
