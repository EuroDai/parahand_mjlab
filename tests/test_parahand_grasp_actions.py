import math

import mujoco
import torch

from mjlab.asset_zoo.robots.parahand_only.parahand_only_constants import (
  PARAHAND_ONLY_XML,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  parahand_only_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.dfc_objects import apply_primitive_stage_randomization
from mjlab.tasks.parahand_grasp.mdp.actions import (
  ParaHandRelativeJointPositionAction,
  RelativeTendonLengthAction,
)
from mjlab.tasks.parahand_grasp.mdp.consts import (
  GRAVITY_TILT_MAX_RAD,
  OBJECT_COM_OFFSET_MAX_M,
  OBJECT_DENSITY_FACTOR_RANGE,
  OBJECT_FRICTION_FACTOR_RANGE,
  ORIGINAL_PALM_STAGE,
  TABLE_FRICTION_FACTOR_RANGE,
)
from mjlab.tasks.parahand_grasp.mdp.events import (
  randomize_teacher_physics,
  reset_joints_above_table,
  reset_primitive_object_pose,
)
from mjlab.utils.lab_api.math import matrix_from_quat


def test_relative_actions_linearly_ramp_targets_across_physics_substeps():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset(seed=0)

  action = torch.zeros(1, env.action_manager.total_action_dim)
  action[:, 0] = 1.0
  action[:, 3] = 1.0
  action[:, 6] = 1.0
  action[:, -1] = -1.0
  env.action_manager.process_action(action)

  joint_action = env.action_manager.get_term("joint_pos")
  tendon_action = env.action_manager.get_term("tendon_length")
  assert isinstance(joint_action, ParaHandRelativeJointPositionAction)
  assert isinstance(tendon_action, RelativeTendonLengthAction)

  joint_start = joint_action._ramp_start.clone()
  tendon_start = tendon_action._ramp_start.clone()
  joint_target = joint_action._ctrl_target.clone()
  tendon_target = tendon_action._ctrl_target.clone()
  initial_joint_pos = env.scene["robot"].data.joint_pos.clone()

  for substep in range(1, cfg.decimation + 1):
    env.action_manager.apply_action()
    alpha = substep / cfg.decimation
    expected_joint_target = torch.lerp(joint_start, joint_target, alpha)
    expected_tendon_target = torch.lerp(tendon_start, tendon_target, alpha)
    assert torch.equal(joint_action._ctrl_target, joint_target)
    assert torch.equal(tendon_action._ctrl_target, tendon_target)
    assert torch.allclose(
      env.scene["robot"].data.joint_pos_target[:, joint_action.target_ids],
      expected_joint_target,
    )
    assert torch.allclose(
      env.scene["robot"].data.tendon_len_target[:, tendon_action.target_ids],
      expected_tendon_target,
    )
    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(dt=env.physics_dt)

  final_joint_pos = env.scene["robot"].data.joint_pos
  assert not torch.equal(final_joint_pos, initial_joint_pos)
  env.close()


def test_relative_joint_action_accumulates_from_previous_target():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset(seed=0)

  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, ParaHandRelativeJointPositionAction)

  action = torch.zeros(1, env.action_manager.total_action_dim)
  z_action_id = 2
  action[:, z_action_id] = 1.0
  initial_target = joint_action._ctrl_target.clone()

  env.action_manager.process_action(action)
  first_target = joint_action._ctrl_target.clone()
  expected_first_z = (
    initial_target[:, z_action_id] + joint_action._processed_actions[:, z_action_id]
  ).clamp(
    joint_action._ctrl_range[:, z_action_id, 0],
    joint_action._ctrl_range[:, z_action_id, 1],
  )
  assert torch.allclose(first_target[:, z_action_id], expected_first_z)

  env.action_manager.process_action(action)
  expected_second_z = (
    first_target[:, z_action_id] + joint_action._processed_actions[:, z_action_id]
  ).clamp(
    joint_action._ctrl_range[:, z_action_id, 0],
    joint_action._ctrl_range[:, z_action_id, 1],
  )
  assert torch.allclose(joint_action._ctrl_target[:, z_action_id], expected_second_z)

  joint_action.reset(torch.tensor([0]))
  current_position = env.scene["robot"].data.joint_pos[:, joint_action.target_ids]
  assert torch.allclose(joint_action._ctrl_target, current_position)
  assert torch.allclose(joint_action._ramp_start, current_position)
  assert torch.allclose(joint_action._applied_ctrl_target, current_position)
  env.close()


def test_stage_zero_uses_complete_follow_object_hand_frame():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=0)
    robot = env.scene["robot"]
    standalone_model = mujoco.MjModel.from_xml_path(str(PARAHAND_ONLY_XML))
    follow_key_id = mujoco.mj_name2id(
      standalone_model,
      mujoco.mjtObj.mjOBJ_KEY,
      "follow_object",
    )
    for joint_name in robot.joint_names:
      if joint_name.startswith("palm_"):
        continue
      standalone_joint_id = mujoco.mj_name2id(
        standalone_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
      )
      qpos_address = standalone_model.jnt_qposadr[standalone_joint_id]
      expected = standalone_model.key_qpos[follow_key_id, qpos_address]
      joint_ids, _ = robot.find_joints((joint_name,), preserve_order=True)
      actual = robot.data.joint_pos[0, joint_ids[0]]
      torch.testing.assert_close(actual, actual.new_tensor(expected))

    tendon_action = env.action_manager.get_term("tendon_length")
    assert isinstance(tendon_action, RelativeTendonLengthAction)
    torch.testing.assert_close(
      tendon_action._ctrl_target,
      torch.full_like(tendon_action._ctrl_target, 0.021),
    )
  finally:
    env.close()


def test_stage_zero_static_override_allows_curriculum_startup():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 1
  apply_primitive_stage_randomization(cfg, 0)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=0)
    point_cloud_cfg = env.observation_manager.get_term_cfg(
      "actor", "object_point_cloud_b"
    )
    assert point_cloud_cfg.noise is None
  finally:
    env.close()


def test_capsule_reset_uses_dedicated_active_joint_and_tendon_references():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 128
  apply_primitive_stage_randomization(cfg, 2)
  cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=0)
    reset_cfg = env.event_manager.get_term_cfg("reset_object_pose")
    assert isinstance(reset_cfg.func, reset_primitive_object_pose)
    capsule = reset_cfg.func.shape_ids == 0
    assert capsule.any()

    robot = env.scene["robot"]
    joint_reset_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
    assert isinstance(joint_reset_cfg.func, reset_joints_above_table)
    palm_body_ids, _ = robot.find_bodies(("palm",), preserve_order=True)
    global_palm_body_id = int(robot.indexing.body_ids[palm_body_ids[0]].item())
    expected_palm_base_rotation = matrix_from_quat(
      torch.tensor(
        env.sim.mj_model.body_quat[global_palm_body_id],
        dtype=torch.float32,
      )[None]
    )[0]
    torch.testing.assert_close(
      joint_reset_cfg.func._palm_base_rotation,
      expected_palm_base_rotation,
    )
    swing_joint_ids, _ = robot.find_joints(
      ("index_mcp_1", "middle_mcp_1", "ring_mcp_1", "little_mcp_1"),
      preserve_order=True,
    )
    swing_position = robot.data.joint_pos[:, swing_joint_ids]
    assert torch.all(swing_position[:, 0] <= swing_position[:, 1])
    assert torch.all(swing_position[:, 1] <= swing_position[:, 2])
    assert torch.all(swing_position[:, 2] <= swing_position[:, 3])

    capsule_reference_by_joint = {
      "thumb_mcp": 0.32,
      "index_mcp_2": 0.95,
      "middle_mcp_2": 0.95,
      "ring_mcp_2": 0.95,
      "little_mcp_2": 0.95,
    }
    for joint_name, reference in capsule_reference_by_joint.items():
      joint_ids, _ = robot.find_joints((joint_name,), preserve_order=True)
      position = robot.data.joint_pos[capsule, joint_ids[0]]
      assert torch.all(position >= reference - 0.05)
      assert torch.all(position <= reference + 0.05)

    box_sphere = ~capsule
    box_sphere_reference_by_joint = {
      "thumb_mcp": 0.23,
      "index_mcp_2": 0.7,
      "middle_mcp_2": 0.7,
      "ring_mcp_2": 0.7,
      "little_mcp_2": 0.7,
    }
    for joint_name, reference in box_sphere_reference_by_joint.items():
      joint_ids, _ = robot.find_joints((joint_name,), preserve_order=True)
      position = robot.data.joint_pos[box_sphere, joint_ids[0]]
      assert torch.all(position >= reference - 0.05)
      assert torch.all(position <= reference + 0.05)

    tendon_action = env.action_manager.get_term("tendon_length")
    assert isinstance(tendon_action, RelativeTendonLengthAction)
    assert torch.all(tendon_action._ctrl_target >= 0.016)
    assert torch.all(tendon_action._ctrl_target <= 0.026)

    passive_ids, _ = robot.find_joints(
      (".*_(pip|dip)",),
      preserve_order=True,
    )
    passive_position = robot.data.joint_pos[:, passive_ids]
    torch.testing.assert_close(
      passive_position[capsule],
      torch.tensor([0.45, 0.39] * 4).expand(int(capsule.sum().item()), -1),
    )
    torch.testing.assert_close(
      passive_position[box_sphere],
      torch.tensor([0.47, 0.36] * 4).expand(int(box_sphere.sum().item()), -1),
    )
  finally:
    env.close()


def test_home_random_palm_stage_keeps_existing_passive_joint_reset_path():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 8
  apply_primitive_stage_randomization(cfg, ORIGINAL_PALM_STAGE)
  cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=0)
    robot = env.scene["robot"]
    passive_ids, _ = robot.find_joints(
      (".*_(pip|dip)",),
      preserve_order=True,
    )
    assert torch.all(robot.data.joint_pos[:, passive_ids] == 0.0)
    joint_reset = env.event_manager.get_term_cfg("reset_robot_joints").func
    assert isinstance(joint_reset, reset_joints_above_table)
    assert torch.isfinite(robot.data.joint_pos).all()
    tendon_action = env.action_manager.get_term("tendon_length")
    assert isinstance(tendon_action, RelativeTendonLengthAction)
    assert torch.isfinite(tendon_action._ctrl_target).all()
  finally:
    env.close()


def test_teacher_physics_randomization_uses_full_ranges_without_accumulation():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 64
  apply_primitive_stage_randomization(cfg, ORIGINAL_PALM_STAGE)
  cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

  def assert_current_reset_is_bounded() -> None:
    object_reset = env.event_manager.get_term_cfg("reset_object_pose").func
    physics = env.event_manager.get_term_cfg("reset_teacher_physics").func
    assert isinstance(object_reset, reset_primitive_object_pose)
    assert isinstance(physics, randomize_teacher_physics)

    sizes = object_reset.sizes
    shape_ids = object_reset.shape_ids
    nominal_mass = torch.empty(env.num_envs)
    capsule = shape_ids == 0
    box = shape_ids == 1
    sphere = shape_ids == 2
    nominal_mass[box] = 500.0 * 8.0 * sizes[box].prod(dim=-1)
    nominal_mass[sphere] = 500.0 * (4.0 / 3.0) * math.pi * sizes[sphere, 0].pow(3)
    radius = sizes[capsule, 0]
    half_length = sizes[capsule, 1]
    nominal_mass[capsule] = 500.0 * (
      math.pi * radius.square() * (2.0 * half_length)
      + (4.0 / 3.0) * math.pi * radius.pow(3)
    )
    object_entity = env.scene["object"]
    object_body_id = int(object_entity.indexing.body_ids[0].item())
    model = env.sim.model
    density_factor = model.body_mass[:, object_body_id] / nominal_mass
    assert torch.all(density_factor >= OBJECT_DENSITY_FACTOR_RANGE[0])
    assert torch.all(density_factor <= OBJECT_DENSITY_FACTOR_RANGE[1])
    assert torch.all(
      torch.linalg.vector_norm(model.body_ipos[:, object_body_id], dim=-1)
      <= math.sqrt(3.0) * OBJECT_COM_OFFSET_MAX_M
    )
    object_friction = model.geom_friction[:, object_entity.indexing.geom_ids, 0].mean(
      dim=-1
    )
    table_entity = env.scene["table"]
    table_friction = model.geom_friction[:, table_entity.indexing.geom_ids, 0].mean(
      dim=-1
    )
    assert torch.all(object_friction >= OBJECT_FRICTION_FACTOR_RANGE[0])
    assert torch.all(object_friction <= OBJECT_FRICTION_FACTOR_RANGE[1])
    assert torch.all(table_friction >= TABLE_FRICTION_FACTOR_RANGE[0])
    assert torch.all(table_friction <= TABLE_FRICTION_FACTOR_RANGE[1])

    gravity = model.opt.gravity
    gravity_magnitude = torch.linalg.vector_norm(gravity, dim=-1)
    assert torch.all(gravity_magnitude >= 9.81 * 0.99)
    assert torch.all(gravity_magnitude <= 9.81 * 1.01)
    gravity_tilt = torch.acos((-gravity[:, 2] / gravity_magnitude).clamp(-1.0, 1.0))
    assert torch.all(gravity_tilt <= GRAVITY_TILT_MAX_RAD + 1.0e-5)
    robot = env.scene["robot"]
    assert robot.indexing.ctrl_ids is not None
    for values in (
      model.dof_damping[:, robot.indexing.joint_v_adr],
      model.actuator_gainprm[:, robot.indexing.ctrl_ids],
      model.actuator_biasprm[:, robot.indexing.ctrl_ids],
      model.actuator_forcerange[:, robot.indexing.ctrl_ids],
    ):
      assert torch.isfinite(values).all()

  try:
    env.reset(seed=0)
    assert_current_reset_is_bounded()
    env.reset()
    assert_current_reset_is_bounded()
  finally:
    env.close()
