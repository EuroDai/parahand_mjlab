import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.parahand_grasp import mdp as parahand_mdp
from mjlab.tasks.parahand_grasp.mdp.commands import (
  TableRelativeLiftingCommandCfg as ObjectPoseCommandCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

_FINGERTIP_CONTACT_SENSOR_NAME = "fingertip_object_contact"
_FINGERTIP_SITE_NAMES = (
  "thumb_tip",
  "index_tip",
  "middle_tip",
  "ring_tip",
  "little_tip",
)
_FINGERTIP_CONTACT_GEOM_NAMES = (
  "thumb_tac",
  "index_tac",
  "middle_tac",
  "ring_tac",
  "little_tac",
)
_ARM_ACTUATOR_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
_HAND_ACTUATOR_NAMES = (
  "thumb_cmc_1",
  "thumb_cmc_2",
  "thumb_mcp",
  "thumb_ip",
  "index_mcp_1",
  "index_mcp_2",
  "middle_mcp_1",
  "middle_mcp_2",
  "ring_mcp_1",
  "ring_mcp_2",
  "little_mcp_1",
  "little_mcp_2",
)
_TENDON_NAMES = (
  "index_tendon",
  "middle_tendon",
  "ring_tendon",
  "little_tendon",
)
_ACTION_TERM_NAMES = ("joint_pos", "tendon_length")
_ARM_ACTION_SCALE = 0.01
_HAND_ACTION_SCALE = 0.03
_TENDON_ACTION_SCALE = 0.005
_CONTACT_THRESHOLD = 0.5
_CONTACT_TEMPERATURE = 0.1
_TABLE_HEIGHT_EVENT_NAME = "reset_table_height"


def make_grasp_object_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base object-grasping task configuration."""

  actor_terms = {
    "object_position_b": ObservationTermCfg(
      func=parahand_mdp.object_position_b,
      params={"object_name": "object"},
      noise=Unoise(n_min=-0.003, n_max=0.003),
      clip=(-2.0, 2.0),
    ),
    "target_position_b": ObservationTermCfg(
      func=parahand_mdp.target_position_b,
      params={"command_name": "object_pose"},
      clip=(-2.0, 2.0),
    ),
    "object_point_cloud_b": ObservationTermCfg(
      func=parahand_mdp.object_point_cloud_b,
      params={
        "object_name": "object",
        "pool_size": 256,
        "sample_size": 256,
        "flatten": True,
        "curriculum_event_name": "reset_object_pose",
      },
      clip=(-2.0, 2.0),
    ),
    "object_to_palm_position_b": ObservationTermCfg(
      func=parahand_mdp.object_to_palm_position_b,
      params={
        "object_name": "object",
        "palm_site_cfg": SceneEntityCfg(
          "robot",
          site_names=("inner_palm_5",),
        ),
      },
      noise=Unoise(n_min=-0.003, n_max=0.003),
      clip=(-2.0, 2.0),
    ),
    "object_quaternion_b": ObservationTermCfg(
      func=parahand_mdp.object_quaternion_b,
      params={"object_name": "object"},
      noise=Unoise(n_min=-0.002, n_max=0.002),
      clip=(-1.0, 1.0),
    ),
    "joint_pos": ObservationTermCfg(
      func=parahand_mdp.joint_position,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "fingertip_pos_b": ObservationTermCfg(
      func=parahand_mdp.site_position_b,
      params={
        "site_asset_cfg": SceneEntityCfg(
          "robot",
          site_names=_FINGERTIP_SITE_NAMES,
          preserve_order=True,
        ),
      },
      noise=Unoise(n_min=-0.002, n_max=0.002),
      clip=(-2.0, 2.0),
    ),
    "fingertip_quat_b": ObservationTermCfg(
      func=parahand_mdp.body_quaternion_b,
      params={
        "body_asset_cfg": SceneEntityCfg(
          "robot",
          body_names=(
            "thumb_dp",
            "index_dp",
            "middle_dp",
            "ring_dp",
            "little_dp",
          ),
          preserve_order=True,
        ),
      },
      noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "palm_pos_b": ObservationTermCfg(
      func=parahand_mdp.site_position_b,
      params={
        "site_asset_cfg": SceneEntityCfg(
          "robot",
          site_names=("inner_palm_5",),
        ),
      },
      noise=Unoise(n_min=-0.002, n_max=0.002),
      clip=(-2.0, 2.0),
    ),
    "palm_quat_b": ObservationTermCfg(
      func=parahand_mdp.body_quaternion_b,
      params={
        "body_asset_cfg": SceneEntityCfg("robot", body_names=("palm",)),
      },
      noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "contact_force": ObservationTermCfg(
      func=parahand_mdp.contact_force_b,
      params={"sensor_name": _FINGERTIP_CONTACT_SENSOR_NAME},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      clip=(-20.0, 20.0),
    ),
    "tendon_state": ObservationTermCfg(
      func=parahand_mdp.tendon_length,
      params={
        "asset_cfg": SceneEntityCfg("robot", tendon_names=_TENDON_NAMES),
      },
      noise=Unoise(n_min=-0.001, n_max=0.001),
    ),
    "actions": ObservationTermCfg(
      func=parahand_mdp.last_actions,
      params={"action_names": _ACTION_TERM_NAMES},
    ),
  }

  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(
      actor_terms,
      enable_corruption=True,
      history_length=5,
      flatten_history_dim=False,
    ),
    "critic": ObservationGroupCfg(
      critic_terms,
      enable_corruption=False,
      history_length=5,
      flatten_history_dim=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": parahand_mdp.ParaHandRelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=_ARM_ACTUATOR_NAMES + _HAND_ACTUATOR_NAMES,
      scale={
        **dict.fromkeys(_ARM_ACTUATOR_NAMES, _ARM_ACTION_SCALE),
        **dict.fromkeys(_HAND_ACTUATOR_NAMES, _HAND_ACTION_SCALE),
      },
      preserve_order=True,
      raw_action_limit=1.0,
    ),
    "tendon_length": parahand_mdp.RelativeTendonLengthActionCfg(
      entity_name="robot",
      actuator_names=_TENDON_NAMES,
      scale=_TENDON_ACTION_SCALE,
      preserve_order=True,
      raw_action_limit=1.0,
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "object_pose": ObjectPoseCommandCfg(
      entity_name="object",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      difficulty="dynamic",
      target_position_range=ObjectPoseCommandCfg.TargetPositionRangeCfg(
        x=(-0.6, -0.4),
        y=(-0.2, 0.2),
        z=(0.35, 0.55),
      ),
      object_pose_range=None,
    )
  }

  events = {
    "reset_table_height": EventTermCfg(
      func=parahand_mdp.reset_table_height,
      mode="reset",
      params={
        "table_name": "table",
        "height_range": (0.6, 1.0),
        "robot_name": None,
        "robot_base_follows_table": False,
        "curriculum_stage": 0,
      },
    ),
    # For positioning the base of the robot at env_origins.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=parahand_mdp.reset_joints_by_curriculum,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "curriculum_stage": 0,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=_ARM_ACTUATOR_NAMES + _HAND_ACTUATOR_NAMES,
          preserve_order=True,
        ),
      },
    ),
    "reset_object_pose": EventTermCfg(
      func=parahand_mdp.reset_primitive_object_pose,
      mode="reset",
      params={
        "object_name": "object",
        "position_center": (0.0, 0.0),
        "position_noise": (0.1, 0.1),
        "capsule_roll_range": (-0.5, 0.5),
        "box_yaw_range": (-0.5 * math.pi, 0.5 * math.pi),
        "curriculum_stage": 0,
        "table_height_event_name": _TABLE_HEIGHT_EVENT_NAME,
        "table_clearance": 0.003,
      },
    ),
  }

  fingertip_contact_sensor_cfg = ContactSensorCfg(
    name=_FINGERTIP_CONTACT_SENSOR_NAME,
    primary=ContactMatch(
      mode="geom",
      pattern=_FINGERTIP_CONTACT_GEOM_NAMES,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="object", entity="object"),
    fields=("force",),
    reduce="netforce",
  )

  metrics = {
    f"{geom_name.removesuffix('_tac')}_contact_force_last": MetricsTermCfg(
      func=parahand_mdp.contact_force_magnitude,
      params={
        "sensor_name": _FINGERTIP_CONTACT_SENSOR_NAME,
        "fingertip_name": geom_name,
      },
      reduce="last",
    )
    for geom_name in _FINGERTIP_CONTACT_GEOM_NAMES
  }

  rewards = {
    "action_l2": RewardTermCfg(
      func=parahand_mdp.action_l2,
      weight=-0.005,
      params={"action_names": _ACTION_TERM_NAMES},
    ),
    "action_rate_l2": RewardTermCfg(
      func=parahand_mdp.action_rate_l2,
      weight=-0.005,
      params={"action_names": _ACTION_TERM_NAMES},
    ),
    "fingers_to_object": RewardTermCfg(
      func=parahand_mdp.fingers_to_object,
      weight=1.0,
      params={
        "std": 0.15,
        "object_cfg": SceneEntityCfg("object"),
        "fingertip_cfg": SceneEntityCfg(
          "robot",
          site_names=_FINGERTIP_SITE_NAMES,
          preserve_order=True,
        ),
      },
    ),
    "object_lift": RewardTermCfg(
      func=parahand_mdp.object_lift,
      weight=2.0,
      params={
        "command_name": "object_pose",
        "object_cfg": SceneEntityCfg("object"),
        "sensor_name": _FINGERTIP_CONTACT_SENSOR_NAME,
        "contact_threshold": _CONTACT_THRESHOLD,
        "contact_temperature": _CONTACT_TEMPERATURE,
      },
    ),
    "position_tracking": RewardTermCfg(
      func=parahand_mdp.position_tracking,
      weight=2.0,
      params={
        "command_name": "object_pose",
        "object_cfg": SceneEntityCfg("object"),
        "sensor_name": _FINGERTIP_CONTACT_SENSOR_NAME,
        "std": 0.25,
        "contact_threshold": _CONTACT_THRESHOLD,
        "contact_temperature": _CONTACT_TEMPERATURE,
      },
    ),
    "good_finger_contact": RewardTermCfg(
      func=parahand_mdp.good_finger_contact,
      weight=0.5,
      params={
        "sensor_name": _FINGERTIP_CONTACT_SENSOR_NAME,
        "threshold": _CONTACT_THRESHOLD,
        "temperature": _CONTACT_TEMPERATURE,
      },
    ),
    "success": RewardTermCfg(
      func=parahand_mdp.success,
      weight=10.0,
      params={
        "command_name": "object_pose",
        "object_cfg": SceneEntityCfg("object"),
        "pos_std": 0.1,
      },
    ),
    "early_termination": RewardTermCfg(
      func=mdp.is_terminated,
      weight=-1.0,
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "object_out_of_bounds": TerminationTermCfg(
      func=parahand_mdp.object_out_of_bounds,
      params={
        "object_name": "object",
        "x_bounds": (-0.4, 0.4),
        "y_bounds": (-0.4, 0.4),
        "table_height_event_name": _TABLE_HEIGHT_EVENT_NAME,
        "z_lower_offset": -0.05,
        "z_upper_offset": 2.0,
      },
    ),
    "abnormal_robot": TerminationTermCfg(
      func=parahand_mdp.abnormal_robot,
      params={"max_abs_qvel": 500.0},
    ),
  }

  curriculum = {
    "object_lesson": CurriculumTermCfg(
      func=parahand_mdp.object_lesson_curriculum,
      params={
        "event_name": "reset_object_pose",
        "table_event_name": "reset_table_height",
        "robot_event_name": "reset_robot_joints",
        "command_name": "object_pose",
        "tendon_action_name": "tendon_length",
        "object_name": "object",
        "promotion_threshold": (0.85, 0.80, 0.75, 0.72, 0.72, 0.70),
        "success_threshold": 0.05,
        "success_window_size": 4096,
        "min_completed_episodes": 1024,
      },
    )
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
      sensors=(fingertip_contact_sensor_cfg,),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=256,
      njmax=2048,
      nan_guard=NanGuardCfg(enabled=True),
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=10,
    episode_length_s=20,
  )
