import math

import torch
from torch.distributions import Normal

from mjlab.rl.distributions import TanhGaussianDistribution


def test_tanh_gaussian_postprocessed_outputs_are_bounded():
  distribution = TanhGaussianDistribution(
    output_dim=4,
    init_std=1.0,
  )
  mean = torch.tensor([[0.0, 1.0, -2.0, 20.0]])
  distribution.update(mean)

  raw_samples = torch.stack([distribution.sample() for _ in range(128)])
  samples = distribution.postprocess(raw_samples)

  assert torch.all(samples >= -1.0)
  assert torch.all(samples <= 1.0)
  assert torch.any(raw_samples.abs() > 1.0)
  torch.testing.assert_close(
    distribution.deterministic_output(mean),
    torch.tanh(mean),
  )
  torch.testing.assert_close(distribution.mean, torch.tanh(mean))


def test_tanh_gaussian_log_prob_includes_change_of_variables():
  distribution = TanhGaussianDistribution(
    output_dim=2,
    init_std=0.7,
  )
  mean = torch.tensor([[0.2, -0.4]])
  desired_std = torch.full_like(mean, 0.7)
  pre_tanh = torch.tensor([[0.5, -1.2]])
  distribution.update(mean)

  base = Normal(mean, desired_std)
  log_det = 2.0 * (
    math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
  )
  expected = (base.log_prob(pre_tanh) - log_det).sum(dim=-1)

  torch.testing.assert_close(distribution.log_prob(pre_tanh), expected)


def test_tanh_gaussian_saturated_postprocessing_log_prob_and_entropy_are_finite():
  distribution = TanhGaussianDistribution(
    output_dim=3,
    init_std=1.0,
  )
  mean = torch.zeros(2, 3, requires_grad=True)
  distribution.update(mean)

  raw_actions = torch.tensor(
    [[-100.0, 0.0, 100.0], [100.0, -100.0, 0.5]],
  )
  objective = distribution.log_prob(raw_actions).sum()
  objective = objective + distribution.entropy.sum()
  objective.backward()

  assert torch.isfinite(objective)
  assert mean.grad is not None
  assert torch.all(torch.isfinite(mean.grad))
  assert distribution.std_param.grad is not None
  assert torch.all(torch.isfinite(distribution.std_param.grad))
  assert torch.all(distribution.postprocess(raw_actions).abs() <= 1.0)


def test_tanh_gaussian_uses_state_independent_std_and_mean_only_output():
  distribution = TanhGaussianDistribution(output_dim=2, init_std=0.7)
  first_mean = torch.zeros(2, 2)
  second_mean = torch.ones(2, 2)

  distribution.update(first_mean)
  first_std = distribution.std.clone()
  distribution.update(second_mean)

  assert distribution.input_dim == 2
  torch.testing.assert_close(first_std, torch.full_like(first_mean, 0.7))
  torch.testing.assert_close(distribution.std, first_std)


def test_tanh_gaussian_kl_uses_pre_tanh_gaussians():
  distribution = TanhGaussianDistribution(
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


def test_tanh_gaussian_log_prob_uses_raw_action_after_tanh_saturation():
  distribution = TanhGaussianDistribution(output_dim=1, init_std=0.3)
  old_mean = torch.tensor([[98.8]])
  raw_action = old_mean.clone()
  distribution.update(old_mean)
  old_log_prob = distribution.log_prob(raw_action)

  new_mean = torch.tensor([[98.78]])
  distribution.update(new_mean)
  new_log_prob = distribution.log_prob(raw_action)

  log_ratio = new_log_prob - old_log_prob

  torch.testing.assert_close(
    log_ratio,
    torch.tensor([-0.0022222222]),
    atol=1e-4,
    rtol=0,
  )
  assert log_ratio.abs().item() < 0.01
  assert distribution.postprocess(raw_action).item() == 1.0
