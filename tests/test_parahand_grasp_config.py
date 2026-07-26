from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import mujoco
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
from mjlab.entity import VariantEntityCfg
from mjlab.tasks.manipulation.mdp import LiftingCommand, LiftingCommandCfg
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  get_object_spec,
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
  NOMINAL_BOX_OBJECT_NAME,
  OBJECT_SCALE_FACTORS,
  OBJECT_SCALE_RANGE,
  PRIMITIVE_OBJECTS,
)
from mjlab.tasks.parahand_grasp.mdp.curriculums import object_lesson_curriculum
from mjlab.tasks.parahand_grasp.mdp.events import (
  _VARIANT_MODEL_FIELDS,
  reset_variant_object_pose,
)
from mjlab.tasks.parahand_grasp.mdp.observations import (
  _sample_model_surface_points,
  object_point_cloud_b,
  object_quaternion_b,
)


def test_object_spec_contains_all_curriculum_primitives():
  for obj in PRIMITIVE_OBJECTS:
    model = get_object_spec(obj).compile()
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, 0)

    assert geom_name == "object_geom"
    assert mujoco.mjtGeom(model.geom_type[0]) == mujoco.mjtGeom.mjGEOM_MESH


def test_mesh_surface_sampler_uses_compiled_mesh_faces():
  model = get_object_spec(PRIMITIVE_OBJECTS[0]).compile()

  points = _sample_model_surface_points(model, num_points=256, device="cpu")

  assert points.shape == (256, 3)
  assert torch.isfinite(points).all()
  assert torch.all(points.abs() <= 0.03 + 1.0e-6)


def test_object_curriculum_starts_with_box_then_uses_scaled_variants():
  cfg = parahand_grasp_object_env_cfg()

  command_cfg = cfg.commands["object_pose"]
  reset_cfg = cfg.events["reset_object_pose"]
  curriculum_cfg = cfg.curriculum["object_lesson"]
  object_cfg = cfg.scene.entities["object"]

  assert isinstance(object_cfg, VariantEntityCfg)
  assert tuple(object_cfg.variants) == tuple(obj.name for obj in PRIMITIVE_OBJECTS)
  assert object_cfg.assignment is None
  assert next(iter(object_cfg.variants)) == NOMINAL_BOX_OBJECT_NAME
  assert len(object_cfg.variants) == 3 * len(OBJECT_SCALE_FACTORS)
  assert min(OBJECT_SCALE_FACTORS) == OBJECT_SCALE_RANGE[0]
  assert max(OBJECT_SCALE_FACTORS) == OBJECT_SCALE_RANGE[1]
  assert isinstance(command_cfg, LiftingCommandCfg)
  assert command_cfg.object_pose_range is None
  assert not hasattr(reset_cfg.func, "model_fields")
  assert all(event_cfg.mode == "reset" for event_cfg in cfg.events.values())
  assert reset_cfg.params["curriculum_stage"] == 0
  point_cloud_cfg = cfg.observations["actor"].terms["object_point_cloud_b"]
  assert point_cloud_cfg.params["curriculum_event_name"] == "reset_object_pose"
  assert point_cloud_cfg.params["dynamic_sampling_stage"] == 2
  actor_term_names = tuple(cfg.observations["actor"].terms)
  assert actor_term_names.index("object_quaternion_b") > actor_term_names.index(
    "object_point_cloud_b"
  )
  assert cfg.observations["actor"].terms["object_quaternion_b"].params == {
    "object_name": "object"
  }
  assert "object_quaternion_b" in cfg.observations["critic"].terms
  reset_joints_cfg = cfg.events["reset_robot_joints"]
  assert reset_joints_cfg.func.__name__ == "reset_joints_by_offset"
  assert reset_joints_cfg.params["position_range"] == (-0.05, 0.05)
  assert curriculum_cfg.params["promotion_threshold"] == 0.85
  assert curriculum_cfg.params["success_threshold"] == 0.05
  assert curriculum_cfg.params["success_window_size"] == 4096
  assert curriculum_cfg.params["min_completed_episodes"] == 1024


def test_grasp_rewards_use_aligned_smooth_contact_gate():
  cfg = parahand_only_grasp_object_env_cfg()
  object_lift = cfg.rewards["object_lift"]
  position_tracking = cfg.rewards["position_tracking"]
  good_finger_contact = cfg.rewards["good_finger_contact"]
  success = cfg.rewards["success"]

  assert object_lift.weight == 2.0
  assert object_lift.params["contact_threshold"] == 0.5
  assert object_lift.params["contact_temperature"] == 0.1
  assert position_tracking.params["contact_threshold"] == 0.5
  assert position_tracking.params["contact_temperature"] == 0.1
  assert good_finger_contact.params["threshold"] == 0.5
  assert good_finger_contact.params["temperature"] == 0.1
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
  event_cfg = SimpleNamespace(params={"curriculum_stage": 0})
  params = {
    "event_name": "reset_object_pose",
    "command_name": "object_pose",
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
    event_manager=SimpleNamespace(get_term_cfg=lambda _name: event_cfg),
    command_manager=SimpleNamespace(get_term=lambda _name: command),
    scene={
      "object": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=object_pos),
      )
    },
  )
  curriculum = cast(Any, object_lesson_curriculum)(cfg, env)
  target_range = command.cfg.target_position_range
  assert target_range.x == (0.0, 0.0)
  assert target_range.y == (0.0, 0.0)
  assert target_range.z == (0.45, 0.45)

  initial_state = curriculum(env, torch.arange(num_envs), **params)
  assert initial_state["completed_episodes"].item() == 0.0
  assert initial_state["stage"].item() == 0.0

  env.episode_length_buf.fill_(1)
  state = curriculum(env, torch.arange(num_envs), **params)

  assert state["success_rate"].item() == pytest.approx(0.85)
  assert state["completed_episodes"].item() == num_envs
  assert state["stage"].item() == 1.0
  assert event_cfg.params["curriculum_stage"] == 1
  assert state["window_count"].item() == 0.0
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)

  state = curriculum(env, torch.arange(num_envs), **params)

  assert state["success_rate"].item() == pytest.approx(0.85)
  assert state["completed_episodes"].item() == 2 * num_envs
  assert state["stage"].item() == 2.0
  assert event_cfg.params["curriculum_stage"] == 2
  assert state["window_count"].item() == 0.0


def test_object_reset_switches_between_box_lesson_and_assigned_variants():
  num_envs = 3
  assigned_fields = {
    field: torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1)
    for field in _VARIANT_MODEL_FIELDS
  }
  nominal_fields = {field: torch.tensor([9.0]) for field in _VARIANT_MODEL_FIELDS}
  model = SimpleNamespace(
    **{field: torch.zeros_like(values) for field, values in assigned_fields.items()}
  )
  env = SimpleNamespace(
    sim=SimpleNamespace(
      model=model,
    )
  )
  event = object.__new__(reset_variant_object_pose)
  event._assigned_model_fields = assigned_fields
  event._nominal_box_model_fields = nominal_fields
  event._assigned_variant_ids = torch.tensor([0, 1, 2])
  event.variant_ids = event._assigned_variant_ids.clone()
  event._nominal_box_variant_id = 0
  event._applied_stage = torch.full((num_envs,), -1, dtype=torch.int8)
  apply_stage = cast(Any, event._apply_curriculum_stage)
  env_ids = torch.arange(num_envs)

  apply_stage(env, env_ids, 0)
  assert event.variant_ids.tolist() == [0, 0, 0]
  for field in _VARIANT_MODEL_FIELDS:
    assert torch.all(getattr(model, field) == 9.0)

  apply_stage(env, env_ids, 1)
  assert event.variant_ids.tolist() == [0, 1, 2]
  for field, assigned in assigned_fields.items():
    torch.testing.assert_close(getattr(model, field), assigned)

  apply_stage(env, env_ids, 2)
  assert event.variant_ids.tolist() == [0, 1, 2]
  for field, assigned in assigned_fields.items():
    torch.testing.assert_close(getattr(model, field), assigned)


def test_object_position_randomizes_in_first_lesson_and_yaw_in_second():
  num_envs = 2
  event = cast(Any, object.__new__(reset_variant_object_pose))
  event._apply_curriculum_stage = Mock()
  event.variant_ids = torch.zeros(num_envs, dtype=torch.long)
  event._floor_offsets = torch.tensor([0.025])
  event.object = SimpleNamespace(
    write_root_link_pose_to_sim=Mock(),
    write_root_link_velocity_to_sim=Mock(),
  )
  env = SimpleNamespace(
    device="cpu",
    scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
  )
  env_ids = torch.arange(num_envs)
  params = {
    "object_name": "object",
    "position_center": (0.0, 0.0),
    "position_noise": (0.05, 0.1),
    "yaw_range": (0.5, 0.5),
  }

  torch.manual_seed(0)
  event(env, env_ids, curriculum_stage=0, **params)
  first_lesson_pose = event.object.write_root_link_pose_to_sim.call_args.args[0]
  assert torch.all(first_lesson_pose[:, 0].abs() <= 0.05)
  assert torch.all(first_lesson_pose[:, 1].abs() <= 0.1)
  assert torch.any(first_lesson_pose[:, :2] != 0.0)
  torch.testing.assert_close(
    first_lesson_pose[:, 3:],
    torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(num_envs, -1),
  )

  torch.manual_seed(0)
  event(env, env_ids, curriculum_stage=1, **params)
  second_lesson_pose = event.object.write_root_link_pose_to_sim.call_args.args[0]
  assert torch.all(second_lesson_pose[:, 0].abs() <= 0.05)
  assert torch.all(second_lesson_pose[:, 1].abs() <= 0.1)
  assert torch.any(second_lesson_pose[:, :2] != 0.0)
  assert torch.all(second_lesson_pose[:, 3] < 1.0)
  assert torch.all(second_lesson_pose[:, 6] > 0.0)


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


def test_point_cloud_sampling_is_fixed_until_final_curriculum_stage():
  term = cast(Any, object.__new__(object_point_cloud_b))
  term._pool_size = 4
  term._sample_size = 2
  term._dynamic_sampling_stage = 2
  term._cache_for_visualization = False
  term._latest_points_w = None
  term._curriculum_event_cfg = SimpleNamespace(params={"curriculum_stage": 0})
  term._variant_ids = torch.tensor([0])
  term._points_local = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
  )
  term._cached_sample_ids = torch.tensor([[0, 1]])
  term._draw_sample_ids = Mock(
    side_effect=(torch.tensor([[2, 3]]), torch.tensor([[1, 3]]))
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
    "sample_size": 2,
    "flatten": False,
    "curriculum_event_name": "reset_object_pose",
    "dynamic_sampling_stage": 2,
  }

  first = term(env, **params)
  second = term(env, **params)
  torch.testing.assert_close(first, second)
  assert term._draw_sample_ids.call_count == 0

  object_entity.data.root_link_pos_w[:, 0] = 0.5
  moved = term(env, **params)
  torch.testing.assert_close(
    moved - first,
    torch.tensor([[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]]),
  )
  assert term._draw_sample_ids.call_count == 0

  term._curriculum_event_cfg.params["curriculum_stage"] = 2
  third = term(env, **params)
  fourth = term(env, **params)

  assert term._draw_sample_ids.call_count == 2
  assert not torch.equal(third, fourth)


def test_point_cloud_cache_randomizes_per_env_on_reset_before_final_lesson():
  term = cast(Any, object.__new__(object_point_cloud_b))
  term._dynamic_sampling_stage = 2
  term._curriculum_event_cfg = SimpleNamespace(params={"curriculum_stage": 0})
  term._cached_sample_ids = torch.tensor([[0, 1], [0, 1]])
  term._draw_sample_ids = Mock(
    side_effect=(
      torch.tensor([[2, 3], [1, 2]]),
      torch.tensor([[3, 0]]),
    )
  )

  term.reset(torch.tensor([0, 1]))
  assert term._cached_sample_ids.tolist() == [[2, 3], [1, 2]]
  assert term._draw_sample_ids.call_args.args[0] == 2

  term._curriculum_event_cfg.params["curriculum_stage"] = 1
  term.reset(torch.tensor([1]))
  assert term._cached_sample_ids.tolist() == [[2, 3], [3, 0]]
  assert term._draw_sample_ids.call_count == 2

  term._curriculum_event_cfg.params["curriculum_stage"] = 2
  term.reset(torch.tensor([0]))
  assert term._cached_sample_ids.tolist() == [[2, 3], [3, 0]]
  assert term._draw_sample_ids.call_count == 2


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
    assert model.key_ctrl[0, tendon_actuator_id] == pytest.approx(0.0285)

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
  assert cfg.events["reset_object_pose"].params["position_center"] == (0.3, 0.1)
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
  assert agent_cfg.algorithm.learning_rate == 3.0e-4
  assert agent_cfg.algorithm.schedule == "fixed"
  assert agent_cfg.actor.distribution_cfg == {
    "class_name": ("mjlab.rl.distributions:StateDependentTanhGaussianDistribution"),
    "init_std": 1.0,
    "min_std": 0.001,
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
  assert agent_cfg.algorithm.learning_rate == 3.0e-4
  assert agent_cfg.algorithm.schedule == "adaptive"
  assert agent_cfg.algorithm.entropy_coef == 0.005
  assert agent_cfg.algorithm.value_loss_coef == 1.0
  assert agent_cfg.algorithm.use_clipped_value_loss
  assert agent_cfg.actor.distribution_cfg == {
    "class_name": ("mjlab.rl.distributions:StateDependentTanhGaussianDistribution"),
    "init_std": 1.0,
    "min_std": 0.001,
  }
