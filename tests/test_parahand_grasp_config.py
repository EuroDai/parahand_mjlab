import math
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.parahand_fr3.parahand_constants import (
  HAND_ACTUATOR_NAMES as PARAHAND_FR3_HAND_ACTUATOR_NAMES,
)
from mjlab.asset_zoo.robots.parahand_fr3.parahand_constants import (
  get_spec as get_parahand_fr3_spec,
)
from mjlab.asset_zoo.robots.parahand_only.parahand_only_constants import (
  HAND_ACTUATOR_NAMES as PARAHAND_ONLY_HAND_ACTUATOR_NAMES,
)
from mjlab.asset_zoo.robots.parahand_only.parahand_only_constants import (
  PALM_ROTATION_ACTUATOR_NAMES,
  PALM_TRANSLATION_ACTUATOR_NAMES,
  PARAHAND_ONLY_ACTION_SCALE,
  PARAHAND_ONLY_XML,
  TENDON_ACTUATOR_NAMES,
  get_parahand_only_robot_cfg,
)
from mjlab.asset_zoo.robots.parahand_only.parahand_only_constants import (
  get_spec as get_parahand_only_spec,
)
from mjlab.entity import EntityCfg, VariantEntityCfg
from mjlab.scripts.play import _apply_curriculum_stage_override
from mjlab.tasks.manipulation.mdp import LiftingCommand, LiftingCommandCfg
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  get_mesh_object_cfg,
  get_object_spec,
  get_table_spec,
  parahand_grasp_object_env_cfg,
  parahand_only_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.config.parahand.rl_cfg import (
  parahand_grasp_object_ppo_runner_cfg,
  parahand_only_grasp_object_ppo_runner_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.actions import (
  ParaHandRelativeJointPositionActionCfg,
  RelativeTendonLengthActionCfg,
)
from mjlab.tasks.parahand_grasp.mdp.consts import (
  BOX_SCALE_RANGE,
  CAPSULE_SCALE_RANGE,
  FIRST_LESSON_OBJECT_NAME,
  ORIGINAL_PALM_STAGE,
  PALM_TRACKING_LAST_STAGE,
  PRIMITIVE_DATASET_STAGE,
  PRIMITIVE_GRAVITY_FRACTIONS,
  PRIMITIVE_OBJECTS,
  PRIMITIVE_RANDOMIZATION_FRACTIONS,
  SPHERE_SCALE_RANGE,
)
from mjlab.tasks.parahand_grasp.mdp.curriculums import object_lesson_curriculum
from mjlab.tasks.parahand_grasp.mdp.events import (
  reset_dropped_mesh_object_pose,
  reset_gravity_by_curriculum,
  reset_joints_above_table,
  reset_primitive_object_pose,
)
from mjlab.tasks.parahand_grasp.mdp.observations import (
  _sample_primitive_surface_points,
  object_point_cloud_b,
  object_quaternion_b,
  object_to_palm_position_b,
)
from mjlab.tasks.parahand_grasp.mdp.terminations import object_out_of_bounds
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_from_euler_xyz


def test_object_spec_contains_all_curriculum_primitives():
  model = get_object_spec().compile()

  assert model.ngeom == len(PRIMITIVE_OBJECTS)
  assert tuple(mujoco.mjtGeom(model.geom_type[i]) for i in range(model.ngeom)) == tuple(
    obj.geom_type for obj in PRIMITIVE_OBJECTS
  )
  assert all(model.geom_dataid[i] == -1 for i in range(model.ngeom))


def test_mesh_object_factory_keeps_future_ycb_variant_path():
  cfg = get_mesh_object_cfg({"example_ycb_object": get_object_spec})

  assert isinstance(cfg, VariantEntityCfg)
  assert tuple(cfg.variants) == ("example_ycb_object",)


def test_table_matches_workspace_bounds():
  model = get_table_spec().compile()

  assert model.ngeom == 1
  assert mujoco.mjtGeom(model.geom_type[0]) == mujoco.mjtGeom.mjGEOM_BOX
  assert tuple(model.geom_size[0]) == pytest.approx((0.4, 0.4, 0.025))
  assert tuple(model.geom_pos[0]) == pytest.approx((0.0, 0.0, -0.025))


def test_object_curriculum_uses_continuously_randomized_primitives():
  cfg = parahand_grasp_object_env_cfg()

  command_cfg = cfg.commands["object_pose"]
  reset_cfg = cfg.events["reset_object_pose"]
  curriculum_cfg = cfg.curriculum["object_lesson"]
  object_cfg = cfg.scene.entities["object"]

  assert isinstance(object_cfg, EntityCfg)
  assert FIRST_LESSON_OBJECT_NAME == "object_box"
  assert len(PRIMITIVE_OBJECTS) == 3
  assert CAPSULE_SCALE_RANGE == (0.5, 2.0)
  assert BOX_SCALE_RANGE == (0.5, 1.0)
  assert SPHERE_SCALE_RANGE == (0.5, 1.5)
  assert PRIMITIVE_DATASET_STAGE == 6
  assert PRIMITIVE_RANDOMIZATION_FRACTIONS == (
    0.0,
    0.1,
    0.1,
    0.5,
    1.0,
    1.0,
  )
  assert PRIMITIVE_GRAVITY_FRACTIONS == (0.0, 0.5, 0.5, 1.0, 1.0, 1.0)
  assert isinstance(command_cfg, LiftingCommandCfg)
  assert command_cfg.object_pose_range is None
  assert "geom_size" in reset_cfg.func.model_fields
  assert "body_inertia" in reset_cfg.func.model_fields
  assert all(event_cfg.mode == "reset" for event_cfg in cfg.events.values())
  assert reset_cfg.params["curriculum_stage"] == 0
  point_cloud_cfg = cfg.observations["actor"].terms["object_point_cloud_b"]
  assert point_cloud_cfg.params["curriculum_event_name"] == "reset_object_pose"
  assert point_cloud_cfg.params["pool_size"] == 256
  assert point_cloud_cfg.params["sample_size"] == 256
  assert "dynamic_sampling_stage" not in point_cloud_cfg.params
  runner_cfg = parahand_grasp_object_ppo_runner_cfg()
  assert runner_cfg.actor.pointnet_cfg is not None
  assert runner_cfg.critic.pointnet_cfg is not None
  assert runner_cfg.actor.pointnet_cfg["point_cloud_points"] == 256
  assert runner_cfg.critic.pointnet_cfg["point_cloud_points"] == 256
  assert runner_cfg.actor.hidden_dims == (1024, 1024, 512, 512)
  assert runner_cfg.critic.hidden_dims == (1024, 1024, 512, 512)
  assert runner_cfg.stage2_enabled is True
  assert runner_cfg.unseen_eval_interval == 300
  assert runner_cfg.unseen_eval_start_stage == 2
  assert runner_cfg.stage2_dataset == "dfc"
  assert runner_cfg.stage2_primitive_ratio == 0.25
  assert runner_cfg.stage2_shard_size_per_rank == 128
  assert runner_cfg.stage2_shard_update_interval == 200
  assert runner_cfg.stage2_drop_height_range == (0.10, 0.15)
  for model_cfg in (runner_cfg.actor, runner_cfg.critic):
    assert model_cfg.pointnet_cfg is not None
    assert model_cfg.pointnet_cfg["feature_dims"] == (64, 128, 256)
    assert model_cfg.pointnet_cfg["pooling"] == "max_mean"
    assert model_cfg.pointnet_cfg["history_mode"] == "latest"
    assert model_cfg.pointnet_cfg["chunk_size"] == 256
    assert model_cfg.pointnet_cfg["gradient_checkpointing"] is True
  actor_term_names = tuple(cfg.observations["actor"].terms)
  assert actor_term_names.index("object_to_palm_position_b") == (
    actor_term_names.index("object_point_cloud_b") + 1
  )
  assert actor_term_names.index("object_quaternion_b") > actor_term_names.index(
    "object_to_palm_position_b"
  )
  assert cfg.observations["actor"].terms["object_to_palm_position_b"].clip == (
    -2.0,
    2.0,
  )
  assert cfg.observations["actor"].terms["object_quaternion_b"].params == {
    "object_name": "object"
  }
  assert "object_quaternion_b" in cfg.observations["critic"].terms
  assert cfg.observations["actor"].nan_policy == "warn"
  assert cfg.observations["actor"].nan_check_per_term is True
  assert cfg.observations["critic"].nan_policy == "warn"
  assert cfg.observations["critic"].nan_check_per_term is True
  assert cfg.non_finite_reset_attempts == 2
  table_reset_cfg = cfg.events["reset_table_height"]
  assert table_reset_cfg.params["height_range"] == (0.6, 1.0)
  assert table_reset_cfg.params["robot_base_follows_table"] is False
  assert reset_cfg.params["table_height_event_name"] == "reset_table_height"
  assert reset_cfg.params["table_clearance"] == 0.003
  out_of_bounds_cfg = cfg.terminations["object_out_of_bounds"]
  assert out_of_bounds_cfg.params == {
    "object_name": "object",
    "x_bounds": (-0.4, 0.4),
    "y_bounds": (-0.4, 0.4),
    "table_height_event_name": "reset_table_height",
    "z_lower_offset": -0.05,
    "z_upper_offset": 2.0,
  }
  assert cfg.terminations["non_finite_state"].func.__name__ == "nan_detection"
  reset_joints_cfg = cfg.events["reset_robot_joints"]
  assert reset_joints_cfg.func.__name__ == "reset_joints_by_curriculum"
  assert reset_joints_cfg.params["position_range"] == (-0.05, 0.05)
  assert curriculum_cfg.params["promotion_threshold"] == (
    0.85,
    0.80,
    0.75,
    0.72,
    0.70,
    0.70,
  )
  assert curriculum_cfg.params["success_threshold"] == 0.05
  assert curriculum_cfg.params["success_window_size"] == 4096
  assert curriculum_cfg.params["min_completed_episodes"] == 1024


def test_standalone_hand_and_table_share_randomized_base_height():
  cfg = parahand_only_grasp_object_env_cfg()

  table_reset_cfg = cfg.events["reset_table_height"]
  assert table_reset_cfg.params["robot_name"] == "robot"
  assert table_reset_cfg.params["robot_base_follows_table"] is True
  joint_reset_cfg = cfg.events["reset_robot_joints"]
  assert joint_reset_cfg.func.__name__ == "reset_joints_above_table"
  assert joint_reset_cfg.params["palm_height_joint_name"] == "palm_translation_z"
  assert joint_reset_cfg.params["palm_height_range"] == (0.2, 0.4)
  assert joint_reset_cfg.params["object_pose_event_name"] == "reset_object_pose"
  assert joint_reset_cfg.params["tendon_action_name"] == "tendon_length"
  event_names = tuple(cfg.events)
  assert event_names.index("reset_object_pose") < event_names.index(
    "reset_robot_joints"
  )


def test_tracking_palm_pose_preserves_shape_reference_in_object_frame():
  reset = object.__new__(reset_joints_above_table)
  model = mujoco.MjModel.from_xml_path(str(PARAHAND_ONLY_XML))
  palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
  reset._palm_base_rotation = matrix_from_quat(
    torch.tensor(model.body_quat[palm_body_id])[None]
  )[0]
  table_heights = torch.tensor([0.6, 0.8, 1.0, 1.2])
  env = SimpleNamespace(
    scene=SimpleNamespace(env_origins=torch.zeros(4, 3)),
    event_manager=SimpleNamespace(
      get_term_cfg=lambda _name: SimpleNamespace(
        func=SimpleNamespace(heights=table_heights)
      )
    ),
  )
  zero = torch.zeros(4)
  yaw = torch.tensor([0.0, 0.0, 0.5 * math.pi, -0.5 * math.pi])
  orientation = quat_from_euler_xyz(zero, zero, yaw)
  object_pose = torch.cat(
    (
      torch.tensor(
        [
          [0.0, 0.0, 0.63],
          [0.0, 0.0, 0.82],
          [0.1, 0.2, 1.03],
          [0.1, 0.2, 1.23],
        ]
      ),
      orientation,
    ),
    dim=-1,
  )

  palm_position = reset._tracking_palm_position(
    cast(Any, env),
    torch.arange(4),
    torch.tensor([1, 0, 2, 1]),
    object_pose,
    "palm_translation_z",
  )

  torch.testing.assert_close(
    palm_position[0],
    torch.tensor([-0.12, -0.005, 0.168, 0.23, 0.0, 0.0]),
  )
  torch.testing.assert_close(
    palm_position[1],
    torch.tensor([-0.122, -0.005, 0.168, 0.23, 0.0, 0.0]),
  )
  torch.testing.assert_close(
    palm_position[2, :3],
    torch.tensor([0.105, 0.08, 0.168]),
  )
  torch.testing.assert_close(
    palm_position[3, :3],
    torch.tensor([0.095, 0.32, 0.168]),
  )

  data = mujoco.MjData(model)
  follow_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "follow_object")
  cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
  mujoco.mj_resetDataKeyframe(model, data, follow_key_id)
  mujoco.mj_forward(model, data)
  reference_relative_rotation = data.xmat[cube_body_id].reshape(3, 3).T @ data.xmat[
    palm_body_id
  ].reshape(3, 3)

  for index in (2, 3):
    data.qpos[:6] = palm_position[index].numpy()
    data.qpos[-7:-4] = (0.1, 0.2, 0.03)
    data.qpos[-4:] = orientation[index].numpy()
    mujoco.mj_forward(model, data)
    transformed_relative_rotation = data.xmat[cube_body_id].reshape(3, 3).T @ data.xmat[
      palm_body_id
    ].reshape(3, 3)
    np.testing.assert_allclose(
      transformed_relative_rotation,
      reference_relative_rotation,
      atol=1.0e-6,
    )


def test_play_curriculum_override_keeps_full_target_randomization_at_stage_one():
  cfg = parahand_only_grasp_object_env_cfg(play=True)

  _apply_curriculum_stage_override(cfg, 1)

  assert cfg.events["reset_object_pose"].params["curriculum_stage"] == 1
  assert cfg.events["reset_gravity"].params["curriculum_stage"] == 1
  assert cfg.curriculum == {}
  command_cfg = cfg.commands["object_pose"]
  assert isinstance(command_cfg, LiftingCommandCfg)
  target_range = command_cfg.target_position_range
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)
  assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.05, 0.05)


def test_primitive_curriculum_override_handles_zero_through_five():
  cfg = parahand_only_grasp_object_env_cfg(play=True)

  _apply_curriculum_stage_override(cfg, 5)
  assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.5, 0.5)

  cfg = parahand_only_grasp_object_env_cfg(play=True)
  with pytest.raises(ValueError, match="between 0 and 5"):
    _apply_curriculum_stage_override(cfg, 6)


@pytest.mark.parametrize(
  "stage,fraction",
  [(1, 0.1), (2, 0.1), (3, 0.5), (4, 1.0), (5, 1.0)],
)
def test_play_curriculum_override_scales_all_robot_ranges(stage, fraction):
  cfg = parahand_only_grasp_object_env_cfg(play=True)

  _apply_curriculum_stage_override(cfg, stage)

  robot_params = cfg.events["reset_robot_joints"].params
  assert robot_params["position_range"] == pytest.approx(
    (-0.5 * fraction, 0.5 * fraction)
  )
  expected_palm_ranges = {
    "palm_translation_x": (-0.1 * fraction, 0.2 * fraction),
    "palm_translation_y": (-0.2 * fraction, 0.2 * fraction),
    "palm_rotation_x": (-0.5 * fraction, 0.5 * fraction),
    "palm_rotation_y": (-0.5 * fraction, 0.5 * fraction),
    "palm_rotation_z": (-0.5 * fraction, 0.5 * fraction),
  }
  for name, expected_range in expected_palm_ranges.items():
    assert robot_params["palm_joint_ranges"][name] == pytest.approx(expected_range)
  assert robot_params["palm_height_range"] == pytest.approx(
    (0.3 - 0.1 * fraction, 0.3 + 0.1 * fraction)
  )
  tendon_cfg = cfg.actions["tendon_length"]
  assert isinstance(tendon_cfg, RelativeTendonLengthActionCfg)
  assert tendon_cfg.reset_target_range == pytest.approx(
    (-0.05 * fraction, 0.05 * fraction)
  )
  command_cfg = cfg.commands["object_pose"]
  assert isinstance(command_cfg, LiftingCommandCfg)
  target_range = command_cfg.target_position_range
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)


@pytest.mark.parametrize(
  "stage,fraction",
  enumerate(PRIMITIVE_GRAVITY_FRACTIONS),
)
def test_gravity_curriculum_scales_world_gravity(stage, fraction):
  gravity = torch.zeros(2, 3)
  env = SimpleNamespace(
    device="cpu",
    sim=SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(gravity=gravity))),
  )

  reset_gravity_by_curriculum(
    cast(Any, env),
    torch.tensor([1]),
    gravity=(0.0, 0.0, -9.81),
    curriculum_stage=stage,
  )

  torch.testing.assert_close(
    gravity[1],
    torch.tensor([0.0, 0.0, -9.81 * fraction]),
  )
  assert torch.all(gravity[0] == 0.0)


def test_play_curriculum_override_keeps_full_target_randomization_at_stage_zero():
  cfg = parahand_only_grasp_object_env_cfg(play=True)

  _apply_curriculum_stage_override(cfg, 0)

  command_cfg = cfg.commands["object_pose"]
  assert isinstance(command_cfg, LiftingCommandCfg)
  target_range = command_cfg.target_position_range
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)


def test_grasp_rewards_use_aligned_smooth_contact_gate():
  cfg = parahand_only_grasp_object_env_cfg()
  object_lift = cfg.rewards["object_lift"]
  position_tracking = cfg.rewards["position_tracking"]
  good_finger_contact = cfg.rewards["good_finger_contact"]
  success = cfg.rewards["success"]

  assert object_lift.weight == 2.0
  assert object_lift.params["contact_threshold"] == 0.5
  assert object_lift.params["contact_temperature"] == 0.1
  assert object_lift.params["contact_core_both_weight"] == 0.5
  assert object_lift.params["contact_auxiliary_bonus"] == 0.5
  assert position_tracking.params["contact_threshold"] == 0.5
  assert position_tracking.params["contact_temperature"] == 0.1
  assert position_tracking.params["contact_core_both_weight"] == 0.5
  assert position_tracking.params["contact_auxiliary_bonus"] == 0.5
  assert good_finger_contact.params["threshold"] == 0.5
  assert good_finger_contact.params["temperature"] == 0.1
  assert good_finger_contact.params["core_both_weight"] == 0.5
  assert good_finger_contact.params["auxiliary_bonus"] == 0.5
  assert set(success.params) == {"command_name", "object_cfg", "pos_std"}
  assert success.params["pos_std"] == 0.1


def test_object_curriculum_promotes_at_isaac_final_success_threshold():
  num_envs = 20
  target_pos = torch.zeros(num_envs, 3)
  object_pos = torch.zeros_like(target_pos)
  object_pos[:17, 0] = 0.049
  object_pos[17:, 0] = 0.051

  command = object.__new__(LiftingCommand)
  command.target_pos = target_pos
  command.cfg = LiftingCommandCfg(
    entity_name="object",
    resampling_time_range=(1.0e9, 1.0e9),
    difficulty="dynamic",
    target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
      x=(-0.1, 0.1),
      y=(-0.1, 0.1),
      z=(0.35, 0.55),
    ),
    object_pose_range=None,
  )
  event_cfgs = {
    "reset_object_pose": SimpleNamespace(params={"curriculum_stage": 0}),
    "reset_table_height": SimpleNamespace(params={"curriculum_stage": 0}),
    "reset_gravity": SimpleNamespace(params={"curriculum_stage": 0}),
    "reset_robot_joints": SimpleNamespace(
      params={"curriculum_stage": 0, "position_range": (0.0, 0.0)}
    ),
  }
  tendon_action_cfg = RelativeTendonLengthActionCfg(
    entity_name="robot",
    actuator_names=("index_tendon",),
    scale=0.005,
  )
  params = {
    "event_name": "reset_object_pose",
    "table_event_name": "reset_table_height",
    "robot_event_name": "reset_robot_joints",
    "gravity_event_name": "reset_gravity",
    "command_name": "object_pose",
    "tendon_action_name": "tendon_length",
    "object_name": "object",
    "promotion_threshold": 0.85,
    "success_threshold": 0.05,
    "success_window_size": 20,
    "min_completed_episodes": 20,
  }
  cfg = SimpleNamespace(params=params)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    event_manager=SimpleNamespace(get_term_cfg=lambda name: event_cfgs[name]),
    command_manager=SimpleNamespace(get_term=lambda _name: command),
    action_manager=SimpleNamespace(
      get_term=lambda _name: SimpleNamespace(cfg=tendon_action_cfg)
    ),
    scene={
      "object": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=object_pos),
      )
    },
  )
  curriculum = cast(Any, object_lesson_curriculum)(cfg, env)
  target_range = command.cfg.target_position_range
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)

  initial_state = curriculum(env, torch.arange(num_envs), **params)
  assert initial_state["completed_episodes"].item() == 0.0
  assert initial_state["stage"].item() == 0.0

  env.episode_length_buf.fill_(1)
  state = curriculum(env, torch.arange(num_envs), **params)

  assert state["success_rate"].item() == pytest.approx(0.85)
  assert state["completed_episodes"].item() == num_envs
  assert state["stage"].item() == 1.0
  assert event_cfgs["reset_object_pose"].params["curriculum_stage"] == 1
  assert event_cfgs["reset_gravity"].params["curriculum_stage"] == 1
  assert state["window_count"].item() == 0.0
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)

  object_pos[:18, 0] = 0.049
  object_pos[18:, 0] = 0.051
  state = curriculum(env, torch.arange(num_envs), **params)

  assert state["success_rate"].item() == pytest.approx(0.90)
  assert state["completed_episodes"].item() == 2 * num_envs
  assert state["stage"].item() == 2.0
  assert event_cfgs["reset_object_pose"].params["curriculum_stage"] == 2
  assert event_cfgs["reset_gravity"].params["curriculum_stage"] == 2
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)
  assert state["window_count"].item() == 0


def test_object_curriculum_success_rate_ignores_inactive_history_capacity():
  curriculum = object.__new__(object_lesson_curriculum)
  curriculum._success_history = torch.zeros(8, dtype=torch.bool)
  curriculum._history_count = 0
  curriculum._write_index = 0
  curriculum._completed_episodes = 0
  curriculum._window_size = 4

  curriculum._append_successes(torch.tensor([True, True, True, False]))

  assert curriculum._history_count == 4
  assert curriculum._success_rate() == pytest.approx(0.75)


def test_object_reset_samples_continuous_dimensions_bounded_by_defaults():
  num_envs = 1000
  event = object.__new__(reset_primitive_object_pose)
  event._base_sizes = torch.tensor([obj.size for obj in PRIMITIVE_OBJECTS])
  event.shape_ids = torch.zeros(num_envs, dtype=torch.long)
  event.sizes = torch.zeros(num_envs, 3)
  env_ids = torch.arange(num_envs)

  event._sample_primitives(env_ids, ORIGINAL_PALM_STAGE)

  assert set(event.shape_ids.tolist()) == {0, 1, 2}
  selected_defaults = event._base_sizes[event.shape_ids]
  positive = selected_defaults > 0
  capsule = event.shape_ids == 0
  box = event.shape_ids == 1
  sphere = event.shape_ids == 2
  capsule_dimensions = event._base_sizes[0] > 0
  assert torch.all(
    event.sizes[capsule][:, capsule_dimensions]
    >= event._base_sizes[0, capsule_dimensions] * CAPSULE_SCALE_RANGE[0]
  )
  assert torch.all(
    event.sizes[capsule][:, capsule_dimensions]
    <= event._base_sizes[0, capsule_dimensions] * CAPSULE_SCALE_RANGE[1]
  )
  assert torch.all(
    event.sizes[box][positive[box]] <= selected_defaults[box][positive[box]]
  )
  assert torch.all(
    event.sizes[box][positive[box]]
    >= selected_defaults[box][positive[box]] * BOX_SCALE_RANGE[0]
  )
  assert torch.all(
    event.sizes[sphere, 0] >= selected_defaults[sphere, 0] * SPHERE_SCALE_RANGE[0]
  )
  assert torch.all(
    event.sizes[sphere, 0] <= selected_defaults[sphere, 0] * SPHERE_SCALE_RANGE[1]
  )
  box_sizes = event.sizes[event.shape_ids == 1]
  assert torch.any(box_sizes[:, 0] != box_sizes[:, 1])
  assert torch.any(box_sizes[:, 1] != box_sizes[:, 2])


@pytest.mark.parametrize(
  "stage,fraction",
  [(2, 0.1), (3, 0.5), (4, 1.0), (5, 1.0)],
)
def test_primitive_size_curriculum_uses_configured_fraction(stage, fraction):
  num_envs = 6000
  event = object.__new__(reset_primitive_object_pose)
  event._base_sizes = torch.tensor([obj.size for obj in PRIMITIVE_OBJECTS])
  event.shape_ids = torch.zeros(num_envs, dtype=torch.long)
  event.sizes = torch.zeros(num_envs, 3)

  event._sample_primitives(torch.arange(num_envs), stage)

  for shape_id, full_range in enumerate(
    (CAPSULE_SCALE_RANGE, BOX_SCALE_RANGE, SPHERE_SCALE_RANGE)
  ):
    mask = event.shape_ids == shape_id
    dimensions = event._base_sizes[shape_id] > 0
    scales = event.sizes[mask][:, dimensions] / event._base_sizes[shape_id, dimensions]
    expected_min = 1.0 + fraction * (full_range[0] - 1.0)
    expected_max = 1.0 + fraction * (full_range[1] - 1.0)
    assert scales.min() >= expected_min
    assert scales.max() <= expected_max
    assert scales.min() < expected_min + 0.01
    assert scales.max() > expected_max - 0.01


def test_first_lesson_uses_one_fixed_nominal_box():
  event = object.__new__(reset_primitive_object_pose)
  event._base_sizes = torch.tensor([obj.size for obj in PRIMITIVE_OBJECTS])
  event.shape_ids = torch.zeros(6, dtype=torch.long)
  event.sizes = torch.zeros(6, 3)
  env_ids = torch.arange(6)

  event._sample_primitives(env_ids, 0)
  assert event.shape_ids.tolist() == [1, 1, 1, 1, 1, 1]
  torch.testing.assert_close(event.sizes, event._base_sizes[event.shape_ids])


def test_ten_percent_box_lesson_randomizes_only_box_dimensions():
  num_envs = 1000
  event = object.__new__(reset_primitive_object_pose)
  event._base_sizes = torch.tensor([obj.size for obj in PRIMITIVE_OBJECTS])
  event.shape_ids = torch.zeros(num_envs, dtype=torch.long)
  event.sizes = torch.zeros(num_envs, 3)

  event._sample_primitives(torch.arange(num_envs), 1)

  assert set(event.shape_ids.tolist()) == {1}
  scales = event.sizes / event._base_sizes[1]
  assert scales.min() >= 0.95
  assert scales.max() <= 1.0
  assert scales.min() < 0.96


def test_object_dimensions_cache_fixed_first_lesson():
  event = cast(Any, object.__new__(reset_primitive_object_pose))
  event._applied_stage = torch.full((3,), -1, dtype=torch.int8)
  event._sample_primitives = Mock()
  event._write_primitive_model = Mock()
  env = SimpleNamespace(
    sim=SimpleNamespace(recompute_constants=Mock()),
  )
  env_ids = torch.arange(3)

  event._apply_curriculum_stage(env, env_ids, 0)
  event._apply_curriculum_stage(env, env_ids, 0)

  assert event._sample_primitives.call_count == 1
  assert event._write_primitive_model.call_count == 1
  assert env.sim.recompute_constants.call_count == 1
  assert event._applied_stage.tolist() == [0, 0, 0]


def test_random_lesson_refreshes_size_slots_at_stage_interval():
  event = cast(Any, object.__new__(reset_primitive_object_pose))
  event.shape_ids = torch.zeros(3, dtype=torch.long)
  event._applied_stage = torch.full((3,), -1, dtype=torch.int8)
  event._slot_cache_stage = torch.full((3,), -1, dtype=torch.int8)
  event._random_stage = -1
  event._random_reset_count = 0

  def mark_refreshed(_env, refresh_ids, stage):
    event._slot_cache_stage[refresh_ids] = stage

  event._refresh_size_slots = Mock(side_effect=mark_refreshed)
  event._activate_cached_primitives = Mock()
  env = SimpleNamespace(device="cpu")
  env_ids = torch.arange(3)

  for _ in range(17):
    event._apply_curriculum_stage(env, env_ids, 1)

  assert event._refresh_size_slots.call_count == 2
  assert event._activate_cached_primitives.call_count == 17
  assert event._applied_stage.tolist() == [1, 1, 1]


def test_inactive_primitive_slots_are_tiny_without_displacement():
  event = cast(Any, object.__new__(reset_primitive_object_pose))
  event._geom_ids = torch.arange(3)
  event._base_sizes = torch.tensor([obj.size for obj in PRIMITIVE_OBJECTS])
  event._colors = torch.tensor([obj.rgba for obj in PRIMITIVE_OBJECTS])
  event.shape_ids = torch.tensor([1])
  event.sizes = torch.tensor([[0.02, 0.025, 0.03]])
  event._write_bounds = Mock()
  event._write_inertia = Mock()
  model = SimpleNamespace(
    geom_size=torch.zeros(1, 3, 3),
    geom_pos=torch.zeros(1, 3, 3),
    geom_rgba=torch.zeros(1, 3, 4),
  )
  env = SimpleNamespace(device="cpu", sim=SimpleNamespace(model=model))

  event._write_primitive_model(env, torch.tensor([0]))

  torch.testing.assert_close(model.geom_size[0, 1], event.sizes[0])
  assert torch.all(model.geom_size[0, (0, 2)] == 1.0e-6)
  assert torch.all(model.geom_pos == 0.0)
  assert model.geom_rgba[0, :, 3].tolist() == [0.0, 1.0, 0.0]


def test_object_pose_is_fixed_first_then_shape_specific_at_full_randomization():
  num_envs = 3
  event = cast(Any, object.__new__(reset_primitive_object_pose))
  event._apply_curriculum_stage = Mock()
  event._floor_offsets = Mock(return_value=torch.tensor([0.02, 0.025, 0.025]))
  event.shape_ids = torch.tensor([0, 1, 2])
  event.sizes = torch.tensor(
    [
      [0.02, 0.035, 0.0],
      [0.03, 0.03, 0.03],
      [0.03, 0.0, 0.0],
    ]
  )
  event.latest_root_pose = torch.zeros(num_envs, 7)
  event.object = SimpleNamespace(
    write_root_link_pose_to_sim=Mock(),
    write_root_link_velocity_to_sim=Mock(),
  )
  env = SimpleNamespace(
    device="cpu",
    scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
    event_manager=SimpleNamespace(
      get_term_cfg=lambda _name: SimpleNamespace(
        func=SimpleNamespace(heights=torch.tensor([0.6, 1.0, 0.8]))
      )
    ),
  )
  env_ids = torch.arange(num_envs)
  params = {
    "object_name": "object",
    "position_center": (0.0, 0.0),
    "position_noise": (0.05, 0.1),
    "capsule_yaw_range": (0.5, 0.5),
    "box_yaw_range": (0.5, 0.5),
    "table_height_event_name": "reset_table_height",
    "table_clearance": 0.003,
  }

  torch.manual_seed(0)
  event(env, env_ids, curriculum_stage=0, **params)
  first_lesson_pose = event.object.write_root_link_pose_to_sim.call_args.args[0]
  assert torch.all(first_lesson_pose[:, :2] == 0.0)
  torch.testing.assert_close(
    first_lesson_pose[:, 2],
    torch.tensor([0.623, 1.028, 0.828]),
  )
  torch.testing.assert_close(
    first_lesson_pose[:, 3:],
    torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(num_envs, -1),
  )

  torch.manual_seed(0)
  event(env, env_ids, curriculum_stage=PALM_TRACKING_LAST_STAGE, **params)
  final_pose = event.object.write_root_link_pose_to_sim.call_args.args[0]
  assert torch.all(final_pose[:, 0].abs() <= 0.05)
  assert torch.all(final_pose[:, 1].abs() <= 0.1)
  assert torch.any(final_pose[:, :2] != 0.0)
  # All primitives stay upright and rotate around the tabletop normal.
  assert final_pose[0, 4] == 0.0
  assert final_pose[0, 6] > 0.0
  assert final_pose[1, 4] == 0.0
  assert final_pose[1, 6] > 0.0
  assert final_pose[2, 4] == 0.0
  assert final_pose[2, 6] > 0.0
  assert final_pose[0, 2] == pytest.approx(0.6 + 0.02 + 0.003)
  torch.testing.assert_close(event.latest_root_pose, final_pose)


def test_dropped_mesh_height_is_measured_above_each_tabletop():
  num_envs = 2
  event = cast(Any, object.__new__(reset_dropped_mesh_object_pose))
  event._variant_ids = torch.zeros(num_envs, dtype=torch.long)
  event._points_local = torch.tensor(
    [[[-0.02, 0.0, -0.03], [0.02, 0.0, 0.03]]],
  )
  event.object = SimpleNamespace(
    write_root_link_pose_to_sim=Mock(),
    write_root_link_velocity_to_sim=Mock(),
  )
  table_heights = torch.tensor([0.6, 1.0])
  env = SimpleNamespace(
    device="cpu",
    scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
    event_manager=SimpleNamespace(
      get_term_cfg=lambda _name: SimpleNamespace(
        func=SimpleNamespace(heights=table_heights)
      )
    ),
  )
  env_ids = torch.arange(num_envs)

  torch.manual_seed(0)
  event(
    env,
    env_ids,
    object_name="object",
    position_center=(0.0, 0.0),
    position_noise=(0.0, 0.0),
    drop_height_range=(0.12, 0.12),
    clearance=0.003,
    table_height_event_name="reset_table_height",
    variant_point_cloud_paths=("unused",),
    variant_point_cloud_scales=(1.0,),
  )

  pose = event.object.write_root_link_pose_to_sim.call_args.args[0]
  local_points = event._points_local[event._variant_ids]
  quaternions = pose[:, None, 3:].expand(-1, local_points.shape[1], -1)
  rotated_points = quat_apply(quaternions, local_points)
  lowest_point_z = pose[:, 2] + rotated_points[..., 2].amin(dim=1)
  torch.testing.assert_close(lowest_point_z, table_heights + 0.003 + 0.12)


def test_object_to_palm_position_is_expressed_in_robot_base_frame():
  half_sqrt_two = 2.0**-0.5
  env = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          root_link_quat_w=torch.tensor([[half_sqrt_two, 0.0, 0.0, half_sqrt_two]]),
          site_pos_w=torch.tensor([[[1.0, 2.0, 3.0]]]),
        )
      ),
      "object": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[2.0, 2.0, 4.0]]))
      ),
    },
  )
  palm_cfg = SimpleNamespace(name="robot", site_ids=torch.tensor([0]))

  value = object_to_palm_position_b(
    cast(Any, env),
    object_name="object",
    palm_site_cfg=cast(Any, palm_cfg),
  )

  torch.testing.assert_close(value, torch.tensor([[0.0, -1.0, 1.0]]))


def test_object_out_of_bounds_uses_absolute_table_relative_limits():
  positions = torch.tensor(
    [
      [10.39, 20.0, 0.56],
      [10.41, 20.0, 0.80],
      [10.0, 20.0, 0.54],
      [10.0, 20.0, 3.01],
    ]
  )
  term = cast(Any, object.__new__(object_out_of_bounds))
  term._object = SimpleNamespace(
    data=SimpleNamespace(
      indexing=SimpleNamespace(free_joint_q_adr=torch.tensor([0, 1, 2])),
      data=SimpleNamespace(qpos=positions),
    )
  )
  env = SimpleNamespace(
    scene=SimpleNamespace(
      env_origins=torch.tensor([[10.0, 20.0, 0.0]]).expand(4, -1),
    ),
    event_manager=SimpleNamespace(
      get_term_cfg=lambda _name: SimpleNamespace(
        func=SimpleNamespace(heights=torch.tensor([0.6, 0.6, 0.6, 1.0]))
      )
    ),
  )

  result = term(
    cast(Any, env),
    object_name="object",
    x_bounds=(-0.4, 0.4),
    y_bounds=(-0.4, 0.4),
    table_height_event_name="reset_table_height",
    z_lower_offset=-0.05,
    z_upper_offset=2.0,
  )

  assert result.tolist() == [False, True, True, True]


def test_object_quaternion_is_expressed_in_robot_base_frame():
  half_sqrt_two = 2.0**-0.5
  env = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          root_link_quat_w=torch.tensor([[half_sqrt_two, 0.0, 0.0, half_sqrt_two]])
        )
      ),
      "object": SimpleNamespace(
        data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
      ),
    },
  )

  value = object_quaternion_b(cast(Any, env), object_name="object")
  torch.testing.assert_close(
    value,
    torch.tensor([[half_sqrt_two, 0.0, 0.0, -half_sqrt_two]]),
  )


def test_point_cloud_is_fixed_and_only_tracks_rigid_object_pose():
  term = cast(Any, object.__new__(object_point_cloud_b))
  term._pool_size = 4
  term._sample_size = 4
  term._cache_for_visualization = False
  term._latest_points_w = None
  term._curriculum_event_cfg = SimpleNamespace(params={"curriculum_stage": 0})
  term._variant_ids = torch.tensor([0])
  term._points_local = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
  )
  identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  zero_pos = torch.zeros(1, 3)
  object_entity = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=zero_pos.clone(),
      root_link_quat_w=identity_quat,
    )
  )
  robot_entity = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=zero_pos.clone(),
      root_link_quat_w=identity_quat,
    )
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene={"object": object_entity, "robot": robot_entity},
  )
  params = {
    "object_name": "object",
    "pool_size": 4,
    "sample_size": 4,
    "flatten": False,
    "curriculum_event_name": "reset_object_pose",
  }

  first = term(env, **params)
  second = term(env, **params)
  torch.testing.assert_close(first, second)

  object_entity.data.root_link_pos_w[:, 0] = 0.5
  moved = term(env, **params)
  torch.testing.assert_close(
    moved - first,
    torch.tensor([[[0.5, 0.0, 0.0]]]).expand(1, 4, 3),
  )

  third = term(env, **params)
  torch.testing.assert_close(third, moved)


def test_point_cloud_reset_refreshes_primitive_geometry_without_resampling():
  term = cast(Any, object.__new__(object_point_cloud_b))
  term._primitive_reset = SimpleNamespace()
  term._points_local = torch.zeros(2, 4, 3)
  term._refresh_primitive_points = Mock()

  term.reset(torch.tensor([0, 1]))
  torch.testing.assert_close(
    term._refresh_primitive_points.call_args.args[0], torch.tensor([0, 1])
  )


@pytest.mark.parametrize(
  "geom_type,size",
  [
    (mujoco.mjtGeom.mjGEOM_BOX, (0.03, 0.02, 0.01)),
    (mujoco.mjtGeom.mjGEOM_SPHERE, (0.03, 0.0, 0.0)),
    (mujoco.mjtGeom.mjGEOM_CAPSULE, (0.02, 0.035, 0.0)),
  ],
)
def test_analytic_point_cloud_is_deterministic(geom_type, size):
  sizes = torch.tensor([size])

  first = _sample_primitive_surface_points(geom_type.value, sizes, 1024)
  second = _sample_primitive_surface_points(geom_type.value, sizes, 1024)

  torch.testing.assert_close(first, second)
  assert first.shape == (1, 1024, 3)


def test_box_point_cloud_lies_on_surface_with_area_weighted_faces():
  half_sizes = torch.tensor([[0.03, 0.02, 0.01]])
  num_points = 600

  points = _sample_primitive_surface_points(
    mujoco.mjtGeom.mjGEOM_BOX.value, half_sizes, num_points
  )[0]

  on_faces = torch.isclose(points.abs(), half_sizes[0])
  assert torch.all(on_faces.sum(dim=-1) == 1)
  assert torch.all(points.abs() <= half_sizes[0])

  expected_axis_counts = (
    (torch.tensor([2.0, 3.0, 6.0]) * (num_points / 11.0)).round().long()
  )
  torch.testing.assert_close(on_faces.sum(dim=0), expected_axis_counts, atol=2, rtol=0)


def test_box_top_and_bottom_faces_have_two_dimensional_coverage():
  half_sizes = torch.tensor([[0.03, 0.02, 0.01]])
  points = _sample_primitive_surface_points(
    mujoco.mjtGeom.mjGEOM_BOX.value, half_sizes, 600
  )[0]

  for z in (-half_sizes[0, 2], half_sizes[0, 2]):
    face_points = points[torch.isclose(points[:, 2], z)]
    x_bins = torch.bucketize(
      face_points[:, 0].contiguous(), torch.linspace(-0.03, 0.03, 5)[1:-1]
    )
    y_bins = torch.bucketize(
      face_points[:, 1].contiguous(), torch.linspace(-0.02, 0.02, 5)[1:-1]
    )
    occupied_bins = torch.unique(x_bins * 4 + y_bins)
    assert len(occupied_bins) == 16


def test_parahand_only_asset_removes_demo_scene_and_builds_articulation():
  spec = get_parahand_only_spec()
  model = spec.compile()
  standalone_model = mujoco.MjModel.from_xml_path(str(PARAHAND_ONLY_XML))

  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube") == -1
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
  assert model.nq == 26
  assert model.nv == 26
  assert model.nu == 22
  assert len(spec.keys) == 1
  assert spec.keys[0].name == "home"
  follow_key_id = mujoco.mj_name2id(
    standalone_model, mujoco.mjtObj.mjOBJ_KEY, "follow_object"
  )
  assert follow_key_id >= 0
  assert standalone_model.key_qpos[follow_key_id, :6] == pytest.approx(
    (-0.12, -0.005, 0.168, 0.23, 0.0, 0.0)
  )
  for tendon_name in TENDON_ACTUATOR_NAMES:
    actuator_id = mujoco.mj_name2id(
      standalone_model,
      mujoco.mjtObj.mjOBJ_ACTUATOR,
      tendon_name,
    )
    assert standalone_model.key_ctrl[follow_key_id, actuator_id] == pytest.approx(0.021)
  palm_translation_z_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "palm_translation_z"
  )
  standalone_palm_translation_z_id = mujoco.mj_name2id(
    standalone_model, mujoco.mjtObj.mjOBJ_JOINT, "palm_translation_z"
  )
  palm_translation_z_qpos_adr = model.jnt_qposadr[palm_translation_z_id]
  standalone_palm_translation_z_qpos_adr = standalone_model.jnt_qposadr[
    standalone_palm_translation_z_id
  ]
  assert model.key_qpos[0, palm_translation_z_qpos_adr] == pytest.approx(
    standalone_model.key_qpos[0, standalone_palm_translation_z_qpos_adr]
  )
  palm_translation_z_actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "palm_translation_z"
  )
  standalone_palm_translation_z_actuator_id = mujoco.mj_name2id(
    standalone_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "palm_translation_z"
  )
  assert model.key_ctrl[0, palm_translation_z_actuator_id] == pytest.approx(
    standalone_model.key_ctrl[0, standalone_palm_translation_z_actuator_id]
  )
  for tendon_name in TENDON_ACTUATOR_NAMES:
    tendon_actuator_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_ACTUATOR, tendon_name
    )
    assert model.key_ctrl[0, tendon_actuator_id] == pytest.approx(0.021)

  robot = get_parahand_only_robot_cfg().build()
  assert robot.root_body.name == "mocap_base"
  assert robot.root_body.mocap
  assert len(robot.joint_names) == 26
  assert len(robot.actuator_names) == 22


def test_parahand_only_standalone_cube_collides_with_fingertips_and_floor():
  model = mujoco.MjModel.from_xml_path(str(PARAHAND_ONLY_XML))
  cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")

  for finger_name in ("index", "middle", "ring", "little"):
    for geom_suffix in ("tip", "tac"):
      geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"{finger_name}_{geom_suffix}",
      )
      assert _geoms_can_collide(model, geom_id, cube_id)

  floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  assert _geoms_can_collide(model, cube_id, floor_id)

  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  index_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "index_tip")
  cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
  cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
  data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = data.geom_xpos[index_tip_id]
  data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
  mujoco.mj_forward(model, data)
  contact_pairs = {(contact.geom[0], contact.geom[1]) for contact in data.contact}
  assert any(cube_id in pair and index_tip_id in pair for pair in contact_pairs)


def _geoms_can_collide(model: mujoco.MjModel, geom_a: int, geom_b: int) -> bool:
  return bool(
    (model.geom_contype[geom_a] & model.geom_conaffinity[geom_b])
    or (model.geom_contype[geom_b] & model.geom_conaffinity[geom_a])
  )


def test_parahand_only_hand_names_match_parahand_fr3():
  hand_spec = get_parahand_only_spec()
  fr3_spec = get_parahand_fr3_spec()
  palm_motion_names = set(PALM_TRANSLATION_ACTUATOR_NAMES) | set(
    PALM_ROTATION_ACTUATOR_NAMES
  )

  assert PARAHAND_ONLY_HAND_ACTUATOR_NAMES == PARAHAND_FR3_HAND_ACTUATOR_NAMES
  assert {body.name for body in hand_spec.bodies} <= {
    body.name for body in fr3_spec.bodies
  }
  assert {
    joint.name for joint in hand_spec.joints if joint.name not in palm_motion_names
  } <= {joint.name for joint in fr3_spec.joints}
  assert {site.name for site in hand_spec.sites if site.name != "wrist_origin"} <= {
    site.name for site in fr3_spec.sites
  }
  assert "wrist_origin" in {site.name for site in hand_spec.sites}


def test_parahand_only_palm_translation_axes_match_world_axis_names():
  hand_spec = get_parahand_only_spec()

  assert tuple(hand_spec.joint("palm_translation_x").axis) == (0.0, 0.0, 1.0)
  assert tuple(hand_spec.joint("palm_translation_y").axis) == (1.0, 0.0, 0.0)
  assert tuple(hand_spec.joint("palm_translation_z").axis) == (0.0, 1.0, 0.0)


def test_parahand_only_task_uses_xml_home_and_requested_action_scales():
  cfg = parahand_only_grasp_object_env_cfg()
  play_cfg = parahand_only_grasp_object_env_cfg(play=True)
  agent_cfg = parahand_only_grasp_object_ppo_runner_cfg()

  joint_action = cfg.actions["joint_pos"]
  assert isinstance(joint_action, ParaHandRelativeJointPositionActionCfg)
  expected_joint_names = (
    PALM_TRANSLATION_ACTUATOR_NAMES
    + PALM_ROTATION_ACTUATOR_NAMES
    + PARAHAND_ONLY_HAND_ACTUATOR_NAMES
  )
  assert joint_action.actuator_names == expected_joint_names
  assert joint_action.coupled_finger_actuator_names == (
    "index_mcp_1",
    "middle_mcp_1",
    "ring_mcp_1",
    "little_mcp_1",
  )
  assert isinstance(joint_action.scale, dict)
  assert all(
    joint_action.scale[name] == PARAHAND_ONLY_ACTION_SCALE[name]
    for name in PALM_TRANSLATION_ACTUATOR_NAMES
  )
  assert all(
    joint_action.scale[name] == PARAHAND_ONLY_ACTION_SCALE[name]
    for name in PALM_ROTATION_ACTUATOR_NAMES
  )
  assert all(
    joint_action.scale[name] == PARAHAND_ONLY_ACTION_SCALE[name]
    for name in PARAHAND_ONLY_HAND_ACTUATOR_NAMES
  )
  tendon_action = cfg.actions["tendon_length"]
  assert isinstance(tendon_action, RelativeTendonLengthActionCfg)
  assert tendon_action.scale == PARAHAND_ONLY_ACTION_SCALE["index_tendon"]
  assert all(
    PARAHAND_ONLY_ACTION_SCALE[name] == tendon_action.scale
    for name in TENDON_ACTUATOR_NAMES
  )

  reset_joints_cfg = cfg.events["reset_robot_joints"].params["asset_cfg"]
  assert reset_joints_cfg.joint_names == expected_joint_names
  assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.05, 0.05)
  assert "curriculum_event_name" not in cfg.events["reset_robot_joints"].params
  assert cfg.events["reset_object_pose"].params["position_center"] == (0, 0)
  command_cfg = cfg.commands["object_pose"]
  assert isinstance(command_cfg, LiftingCommandCfg)
  target_range = command_cfg.target_position_range
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)
  assert cfg.scene.entities["object"].init_state.pos == (0.3, 0.1, 0.03)
  assert cfg.scene.num_envs == 4096
  assert "object_point_cloud_debug" not in cfg.rewards
  assert play_cfg.scene.num_envs == 1
  assert play_cfg.episode_length_s == int(1e9)
  assert not play_cfg.observations["actor"].enable_corruption
  assert "object_point_cloud_debug" in play_cfg.rewards
  assert agent_cfg.experiment_name == "parahand_only_grasp_object"
  assert agent_cfg.algorithm.learning_rate == 1.0e-4
  assert agent_cfg.algorithm.schedule == "fixed"
  assert agent_cfg.actor.distribution_cfg == {
    "class_name": "mjlab.rl.distributions:TanhGaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }

  fingertip_quat_cfg = cfg.observations["actor"].terms["fingertip_quat_b"]
  assert fingertip_quat_cfg.params["body_asset_cfg"].body_names == (
    "thumb_dp",
    "index_dp",
    "middle_dp",
    "ring_dp",
    "little_dp",
  )


def test_training_config_matches_playground_horizon_and_ppo_settings():
  env_cfg = parahand_grasp_object_env_cfg()
  play_env_cfg = parahand_grasp_object_env_cfg(play=True)
  agent_cfg = parahand_grasp_object_ppo_runner_cfg()

  assert env_cfg.scene.num_envs == 4096
  assert "object_point_cloud_debug" not in env_cfg.rewards
  assert play_env_cfg.scene.num_envs == 1
  assert play_env_cfg.episode_length_s == int(1e9)
  point_cloud_cfg = play_env_cfg.observations["actor"].terms["object_point_cloud_b"]
  assert point_cloud_cfg.params["cache_for_visualization"]
  assert "object_point_cloud_debug" in play_env_cfg.rewards
  assert play_env_cfg.rewards["object_point_cloud_debug"].weight == 0.0
  assert env_cfg.sim.mujoco.timestep == 0.005
  assert env_cfg.sim.nan_guard.enabled
  assert env_cfg.decimation == 10
  assert env_cfg.episode_length_s == 20
  assert env_cfg.episode_length_s / (env_cfg.sim.mujoco.timestep * 10) == 400
  assert agent_cfg.num_steps_per_env == 32
  assert agent_cfg.max_iterations == 763
  assert agent_cfg.algorithm.num_learning_epochs == 2
  assert agent_cfg.algorithm.num_mini_batches == 32
  assert agent_cfg.algorithm.learning_rate == 1.0e-4
  assert agent_cfg.algorithm.schedule == "adaptive"
  assert agent_cfg.algorithm.entropy_coef == 0.005
  assert agent_cfg.algorithm.value_loss_coef == 1.0
  assert agent_cfg.algorithm.use_clipped_value_loss
  assert agent_cfg.algorithm.class_name == "mjlab.rl.ppo:StablePPO"
  assert agent_cfg.actor.distribution_cfg == {
    "class_name": "mjlab.rl.distributions:TanhGaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }
