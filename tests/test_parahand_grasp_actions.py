import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  parahand_only_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.actions import (
  ParaHandRelativeJointPositionAction,
  RelativeTendonLengthAction,
)


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
