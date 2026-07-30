from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  parahand_only_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.observations import (
  contact_force_b,
  contact_force_magnitude,
)


def test_contact_force_observation_preserves_xyz_in_robot_base_frame():
  force_w = torch.zeros(2, 5, 3)
  force_w[0, 0] = torch.tensor([1.0, 2.0, 3.0])
  force_w[1, 2] = torch.tensor([4.0, 5.0, 6.0])
  sensor = SimpleNamespace(data=SimpleNamespace(force=force_w))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_quat_w=torch.tensor(
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
        ]
      )
    )
  )
  env = SimpleNamespace(
    scene={
      "fingertip_object_contact": sensor,
      "robot": robot,
    }
  )

  force_b = cast(Any, contact_force_b)(
    env,
    sensor_name="fingertip_object_contact",
  )

  expected = force_w.clone()
  expected[1, :, :2] *= -1.0
  assert force_b.shape == (2, 15)
  torch.testing.assert_close(force_b, expected.flatten(start_dim=1))


def test_contact_force_observation_selects_named_sensor_primary():
  force = torch.zeros(2, 5, 3)
  force[:, 2] = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
  sensor = SimpleNamespace(
    primary_names=[
      "thumb_tac",
      "index_tac",
      "middle_tac",
      "ring_tac",
      "little_tac",
    ],
    data=SimpleNamespace(force=force),
  )
  env = SimpleNamespace(scene={"fingertip_object_contact": sensor})

  all_fingertips = cast(Any, contact_force_magnitude)(
    env,
    sensor_name="fingertip_object_contact",
  )
  middle_fingertip = cast(Any, contact_force_magnitude)(
    env,
    sensor_name="fingertip_object_contact",
    fingertip_name="middle_tac",
  )

  assert all_fingertips.shape == (2, 5)
  torch.testing.assert_close(middle_fingertip, torch.tensor([5.0, 2.0]))


def test_contact_force_observation_rejects_unknown_primary():
  sensor = SimpleNamespace(
    primary_names=["thumb_tac"],
    data=SimpleNamespace(force=torch.zeros(1, 1, 3)),
  )
  env = SimpleNamespace(scene={"fingertip_object_contact": sensor})

  with pytest.raises(ValueError, match="missing_tac"):
    cast(Any, contact_force_magnitude)(
      env,
      sensor_name="fingertip_object_contact",
      fingertip_name="missing_tac",
    )


def test_parahand_registers_last_contact_force_for_each_fingertip():
  cfg = parahand_only_grasp_object_env_cfg()
  expected_fingertips = ("thumb", "index", "middle", "ring", "little")

  contact_force_cfg = cfg.observations["actor"].terms["contact_force"]
  assert contact_force_cfg.func is contact_force_b
  assert contact_force_cfg.clip == (-20.0, 20.0)
  assert set(cfg.metrics) == {
    f"{fingertip}_contact_force_last" for fingertip in expected_fingertips
  }
  for fingertip in expected_fingertips:
    metric_cfg = cfg.metrics[f"{fingertip}_contact_force_last"]
    assert metric_cfg.func is contact_force_magnitude
    assert metric_cfg.reduce == "last"
    assert metric_cfg.per_substep is False
    assert metric_cfg.params == {
      "sensor_name": "fingertip_object_contact",
      "fingertip_name": f"{fingertip}_tac",
    }


def test_parahand_logs_last_fingertip_contact_force_on_episode_reset():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 1
  cfg.episode_length_s = cfg.decimation * cfg.sim.mujoco.timestep
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset(seed=0)
    actor_obs = obs["actor"]
    critic_obs = obs["critic"]
    assert isinstance(actor_obs, torch.Tensor)
    assert isinstance(critic_obs, torch.Tensor)
    assert actor_obs.shape == (1, 5, 890)
    assert critic_obs.shape == (1, 5, 890)
    action = torch.zeros(1, env.action_manager.total_action_dim)
    _, _, _, time_out, extras = env.step(action)

    assert time_out.item()
    for fingertip in ("thumb", "index", "middle", "ring", "little"):
      value = extras["log"][f"Episode_Metrics/{fingertip}_contact_force_last"]
      assert torch.isfinite(value)
  finally:
    env.close()
