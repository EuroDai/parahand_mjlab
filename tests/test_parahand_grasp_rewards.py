from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp import LiftingCommand
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  parahand_only_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.rewards import (
  contact_score,
  object_lift,
  smooth_contact_score,
  success,
)


def test_smooth_contact_score_requires_core_contact_and_rewards_auxiliary_contact():
  force_magnitude = torch.tensor(
    [
      [0.0, 0.0, 0.0, 0.0, 0.0],
      [0.7, 0.7, 0.0, 0.0, 0.0],
      [0.7, 0.0, 0.0, 0.7, 0.0],
      [0.7, 0.7, 0.7, 0.0, 0.0],
      [0.7, 0.7, 0.7, 0.7, 0.7],
    ]
  )

  score = smooth_contact_score(
    force_magnitude,
    threshold=0.5,
    temperature=0.1,
  )

  finger_scores = torch.sigmoid((force_magnitude - 0.5) / 0.1)
  thumb, index, middle, ring, little = finger_scores.unbind(dim=-1)
  likelihoods = torch.tensor([0.96, 0.92, 0.79, 0.39])
  weights = 0.2 + 0.8 * torch.square(likelihoods / likelihoods[0])
  index_weight, middle_weight, ring_weight, little_weight = weights
  core_any = 1.0 - (1.0 - index) * (1.0 - middle)
  core_both = torch.exp(
    (index_weight * torch.log(index) + middle_weight * torch.log(middle))
    / (index_weight + middle_weight)
  )
  core = 0.5 * core_any + 0.5 * core_both
  auxiliary = (ring_weight * ring + little_weight * little) / (
    ring_weight + little_weight
  )
  base_gate = thumb * core
  expected = base_gate * (1.0 + 0.5 * (1.0 - base_gate) * auxiliary)

  torch.testing.assert_close(score, expected)
  assert score[0].item() < 1.0e-4
  assert score[1].item() > 0.4
  assert score[2].item() < 0.02
  assert score[3].item() > 0.8
  assert score[4].item() > score[3].item()


def test_smooth_contact_score_rejects_nonpositive_temperature():
  with pytest.raises(ValueError, match="must be positive"):
    smooth_contact_score(
      torch.ones(1, 5),
      threshold=0.5,
      temperature=0.0,
    )


@pytest.mark.parametrize(
  ("parameter", "value", "message"),
  [
    ("core_both_weight", -0.1, "Core-both weight"),
    ("core_both_weight", 1.1, "Core-both weight"),
    ("auxiliary_bonus", -0.1, "Auxiliary bonus"),
    ("auxiliary_bonus", 1.1, "Auxiliary bonus"),
  ],
)
def test_smooth_contact_score_rejects_invalid_weights(
  parameter: str,
  value: float,
  message: str,
):
  kwargs = {parameter: value}
  with pytest.raises(ValueError, match=message):
    smooth_contact_score(
      torch.ones(1, 5),
      threshold=0.5,
      temperature=0.1,
      **kwargs,
    )


def test_smooth_contact_score_requires_five_ordered_fingertips():
  with pytest.raises(ValueError, match="requires thumb, index, middle, ring"):
    smooth_contact_score(
      torch.ones(1, 4),
      threshold=0.5,
      temperature=0.1,
    )


def test_contact_score_resolves_fingertips_by_sensor_name():
  ordered_force_magnitude = torch.tensor([[0.7, 0.7, 0.7, 0.2, 0.1]])
  sensor_order = [3, 0, 4, 2, 1]
  sensor = SimpleNamespace(
    data=SimpleNamespace(force=ordered_force_magnitude[:, sensor_order].unsqueeze(-1)),
    primary_names=[
      "ring_tac",
      "thumb_tac",
      "little_tac",
      "middle_tac",
      "index_tac",
    ],
  )
  env = SimpleNamespace(scene={"fingertip_contact": sensor})

  score = contact_score(
    cast(Any, env),
    sensor_name="fingertip_contact",
    threshold=0.5,
    temperature=0.1,
  )

  expected = smooth_contact_score(
    ordered_force_magnitude,
    threshold=0.5,
    temperature=0.1,
  )
  torch.testing.assert_close(score, expected)


def test_success_reward_does_not_require_contact():
  command = object.__new__(LiftingCommand)
  command.target_pos = torch.zeros(2, 3)
  object_pos = torch.zeros(2, 3)
  env = SimpleNamespace(
    scene={
      "object": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=object_pos),
      ),
    },
    command_manager=SimpleNamespace(get_term=lambda _name: command),
  )

  reward = success(
    cast(Any, env),
    command_name="object_pose",
    object_cfg=SceneEntityCfg("object"),
    pos_std=0.1,
  )

  torch.testing.assert_close(reward, torch.ones(2))


def test_object_lift_uses_reset_relative_progress_and_smooth_contact():
  command = object.__new__(LiftingCommand)
  command.target_pos = torch.tensor([[0.0, 0.0, 0.42], [0.0, 0.0, 0.50]])
  qpos = torch.tensor(
    [
      [0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
    ]
  )
  obj = SimpleNamespace(
    data=SimpleNamespace(
      indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
      data=SimpleNamespace(qpos=qpos),
    ),
  )
  force_magnitude = torch.tensor(
    [
      [0.7, 0.7, 0.0, 0.0, 0.0],
      [0.5, 0.3, 0.6, 0.4, 0.2],
    ]
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(force=force_magnitude.unsqueeze(-1)),
    primary_names=[
      "thumb_tac",
      "index_tac",
      "middle_tac",
      "ring_tac",
      "little_tac",
    ],
  )
  env = SimpleNamespace(
    scene={"object": obj, "fingertip_contact": sensor},
    command_manager=SimpleNamespace(get_term=lambda _name: command),
  )
  term = cast(Any, object.__new__(object_lift))
  term._object = obj
  term._initial_height = torch.zeros(2)
  term.reset(None)

  qpos[:, 2] = torch.tensor([0.22, 0.50])
  reward = term(
    cast(Any, env),
    command_name="object_pose",
    object_cfg=SceneEntityCfg("object"),
    sensor_name="fingertip_contact",
    contact_threshold=0.5,
    contact_temperature=0.1,
  )

  contact_gate = smooth_contact_score(
    force_magnitude,
    threshold=0.5,
    temperature=0.1,
  )
  expected_progress = torch.tensor([0.5, 1.0])
  torch.testing.assert_close(reward, expected_progress * contact_gate)


def test_object_lift_clamps_downward_and_above_target_progress():
  command = object.__new__(LiftingCommand)
  command.target_pos = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]])
  qpos = torch.tensor(
    [
      [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
    ]
  )
  obj = SimpleNamespace(
    data=SimpleNamespace(
      indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
      data=SimpleNamespace(qpos=qpos),
    ),
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(force=torch.full((2, 5, 1), 10.0)),
    primary_names=[
      "thumb_tac",
      "index_tac",
      "middle_tac",
      "ring_tac",
      "little_tac",
    ],
  )
  env = SimpleNamespace(
    scene={"object": obj, "fingertip_contact": sensor},
    command_manager=SimpleNamespace(get_term=lambda _name: command),
  )
  term = cast(Any, object.__new__(object_lift))
  term._object = obj
  term._initial_height = torch.tensor([0.5, 0.1])

  reward = term(
    cast(Any, env),
    command_name="object_pose",
    object_cfg=SceneEntityCfg("object"),
    sensor_name="fingertip_contact",
    contact_threshold=0.5,
    contact_temperature=0.1,
  )

  torch.testing.assert_close(reward, torch.tensor([0.0, 1.0]))


def test_object_lift_captures_object_height_after_reset_event():
  cfg = parahand_only_grasp_object_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset(seed=0)
    term_cfg = env.reward_manager.get_term_cfg("object_lift")
    assert isinstance(term_cfg.func, object_lift)

    obj = env.scene["object"]
    q_adr = obj.data.indexing.free_joint_q_adr
    current_height = obj.data.data.qpos[:, q_adr[2]]
    torch.testing.assert_close(term_cfg.func._initial_height, current_height)

    reward = env.reward_manager.compute(dt=env.step_dt)
    assert torch.all(torch.isfinite(reward))
  finally:
    env.close()
