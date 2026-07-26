import math

import pytest
import torch
from torch.distributions import Normal

from mjlab.rl.distributions import StateDependentTanhGaussianDistribution


def test_tanh_gaussian_outputs_are_bounded():
  distribution = StateDependentTanhGaussianDistribution(
    output_dim=4,
    init_std=1.0,
  )
  mean = torch.tensor([[0.0, 1.0, -2.0, 20.0]])
  scale_logits = torch.zeros_like(mean)
  mlp_output = torch.stack((mean, scale_logits), dim=-2)
  distribution.update(mlp_output)

  samples = torch.stack([distribution.sample() for _ in range(128)])

  assert torch.all(samples > -1.0)
  assert torch.all(samples < 1.0)
  torch.testing.assert_close(
    distribution.deterministic_output(mlp_output),
    torch.tanh(mean),
  )
  torch.testing.assert_close(distribution.mean, torch.tanh(mean))


def test_tanh_gaussian_log_prob_includes_change_of_variables():
  distribution = StateDependentTanhGaussianDistribution(
    output_dim=2,
    init_std=0.7,
  )
  mean = torch.tensor([[0.2, -0.4]])
  desired_std = torch.full_like(mean, 0.7)
  scale_logits = torch.log(torch.expm1(desired_std - distribution.min_std))
  pre_tanh = torch.tensor([[0.5, -1.2]])
  actions = torch.tanh(pre_tanh)
  distribution.update(torch.stack((mean, scale_logits), dim=-2))

  base = Normal(mean, desired_std)
  log_det = 2.0 * (
    math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
  )
  expected = (base.log_prob(pre_tanh) - log_det).sum(dim=-1)

  torch.testing.assert_close(distribution.log_prob(actions), expected)


def test_tanh_gaussian_boundary_log_prob_and_entropy_are_finite():
  distribution = StateDependentTanhGaussianDistribution(
    output_dim=3,
    init_std=1.0,
  )
  mean = torch.zeros(2, 3, requires_grad=True)
  scale_logits = torch.zeros(2, 3, requires_grad=True)
  distribution.update(torch.stack((mean, scale_logits), dim=-2))

  boundary_actions = torch.tensor(
    [[-1.0, 0.0, 1.0], [1.0, -1.0, 0.5]],
  )
  objective = distribution.log_prob(boundary_actions).sum()
  objective = objective + distribution.entropy.sum()
  objective.backward()

  assert torch.isfinite(objective)
  assert mean.grad is not None
  assert torch.all(torch.isfinite(mean.grad))
  assert scale_logits.grad is not None
  assert torch.all(torch.isfinite(scale_logits.grad))


def test_tanh_gaussian_std_depends_on_network_output():
  distribution = StateDependentTanhGaussianDistribution(output_dim=2)
  mean = torch.zeros(2, 2)
  scale_logits = torch.tensor([[-2.0, 0.0], [1.0, 2.0]])

  distribution.update(torch.stack((mean, scale_logits), dim=-2))

  expected = torch.nn.functional.softplus(scale_logits) + distribution.min_std
  torch.testing.assert_close(distribution.std, expected)
  assert not torch.equal(distribution.std[0], distribution.std[1])


def test_tanh_gaussian_kl_uses_pre_tanh_gaussians():
  distribution = StateDependentTanhGaussianDistribution(
    output_dim=2,
    init_std=0.5,
  )
  old_mean = torch.tensor([[0.2, -0.4]])
  new_mean = torch.tensor([[-0.1, 0.3]])
  old_std = torch.tensor([0.5, 0.7])
  new_std = torch.tensor([0.8, 0.6])

  actual = distribution.kl_divergence(
    (old_mean, old_std),
    (new_mean, new_std),
  )
  expected = torch.distributions.kl_divergence(
    Normal(old_mean, old_std),
    Normal(new_mean, new_std),
  ).sum(dim=-1)

  torch.testing.assert_close(actual, expected)


def test_tanh_gaussian_validates_squash_epsilon():
  with pytest.raises(ValueError, match="squash_epsilon"):
    StateDependentTanhGaussianDistribution(output_dim=2, squash_epsilon=0.0)


def test_tanh_gaussian_validates_standard_deviations():
  with pytest.raises(ValueError, match="min_std"):
    StateDependentTanhGaussianDistribution(output_dim=2, min_std=0.0)
  with pytest.raises(ValueError, match="init_std"):
    StateDependentTanhGaussianDistribution(
      output_dim=2,
      init_std=0.001,
      min_std=0.001,
    )
