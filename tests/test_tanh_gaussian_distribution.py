import torch

from mjlab.rl import TanhGaussianDistribution


def test_tanh_gaussian_distribution_is_bounded_and_finite():
  distribution = TanhGaussianDistribution(output_dim=3)
  distribution.update(torch.zeros(8, 2, 3))

  actions = distribution.sample()

  assert actions.shape == (8, 3)
  assert torch.all(actions > -1.0)
  assert torch.all(actions < 1.0)
  assert torch.isfinite(distribution.log_prob(actions)).all()
  assert torch.isfinite(distribution.entropy).all()


def test_tanh_gaussian_deterministic_output_uses_mean():
  distribution = TanhGaussianDistribution(output_dim=2)
  mlp_output = torch.tensor([[[0.5, -0.5], [10.0, -10.0]]])

  torch.testing.assert_close(
    distribution.deterministic_output(mlp_output),
    torch.tanh(torch.tensor([[0.5, -0.5]])),
  )
