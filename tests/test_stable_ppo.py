from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab.rl.distributions import TanhGaussianDistribution
from mjlab.rl.ppo import StablePPO


def test_stable_ppo_clamps_log_ratio_before_exponentiation():
  new_log_prob = torch.tensor([-100.0, 0.0, 100.0])
  old_log_prob = torch.zeros(3)

  ratio = StablePPO._compute_ratio(new_log_prob, old_log_prob, 20.0)

  torch.testing.assert_close(
    ratio,
    torch.exp(torch.tensor([-20.0, 0.0, 20.0])),
  )
  assert torch.all(torch.isfinite(ratio))


def test_stable_ppo_rejects_non_finite_losses_and_gradients():
  ppo = StablePPO.__new__(StablePPO)
  ppo.device = "cpu"
  ppo.is_multi_gpu = False
  parameter = torch.nn.Parameter(torch.tensor([1.0]))

  assert ppo._all_ranks_finite([torch.tensor(1.0)])
  assert not ppo._all_ranks_finite([torch.tensor(float("nan"))])

  parameter.grad = torch.tensor([1.0])
  assert ppo._all_ranks_gradients_finite([parameter])
  parameter.grad = torch.tensor([float("inf")])
  assert not ppo._all_ranks_gradients_finite([parameter])


def test_stable_ppo_postprocesses_base_ppo_raw_actions_for_environment(monkeypatch):
  raw_actions = torch.tensor([[0.0, 1.0, 98.8]])
  expected_raw_actions = raw_actions.clone()
  distribution = TanhGaussianDistribution(output_dim=3)
  ppo = StablePPO.__new__(StablePPO)
  ppo.actor = cast(Any, torch.nn.Module())
  ppo.actor.distribution = distribution

  def return_raw_actions(_self, _obs):
    return raw_actions

  monkeypatch.setattr(PPO, "act", return_raw_actions)

  env_actions = ppo.act(TensorDict({}, batch_size=[1]))

  torch.testing.assert_close(raw_actions, expected_raw_actions)
  torch.testing.assert_close(env_actions, torch.tanh(raw_actions))
