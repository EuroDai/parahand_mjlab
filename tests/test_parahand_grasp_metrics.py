from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  parahand_only_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.observations import contact_force_magnitude


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
    env.reset(seed=0)
    action = torch.zeros(1, env.action_manager.total_action_dim)
    _, _, _, time_out, extras = env.step(action)

    assert time_out.item()
    for fingertip in ("thumb", "index", "middle", "ring", "little"):
      value = extras["log"][f"Episode_Metrics/{fingertip}_contact_force_last"]
      assert torch.isfinite(value)
  finally:
    env.close()
