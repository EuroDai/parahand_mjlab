from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal


class TanhGaussianDistribution(GaussianDistribution):
  """State-independent diagonal Gaussian followed by ``tanh``.

  This retains :class:`GaussianDistribution`'s state-independent, optionally
  learnable standard deviation. PPO operates on unsquashed Gaussian samples while
  environment and deterministic outputs are bounded to ``[-1, 1]``.
  Log-probabilities include the tanh change-of-variables correction.
  """

  def __init__(
    self,
    output_dim: int,
    init_std: float = 1.0,
    std_range: tuple[float, float] = (1e-6, 1e6),
    std_type: str = "scalar",
    learn_std: bool = True,
  ) -> None:
    super().__init__(
      output_dim=output_dim,
      init_std=init_std,
      std_range=std_range,
      std_type=std_type,
      learn_std=learn_std,
    )

  def sample(self) -> torch.Tensor:
    """Sample an unsquashed action for PPO storage and optimization."""
    return self._distribution.sample()  # type: ignore[union-attr]

  def postprocess(self, raw_actions: torch.Tensor) -> torch.Tensor:
    """Squash raw Gaussian actions before sending them to the environment."""
    return torch.tanh(raw_actions)

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    """Squash the Gaussian location for deterministic inference."""
    return torch.tanh(mlp_output)

  def as_deterministic_output_module(self) -> nn.Module:
    """Return an export-friendly tanh output module."""
    return _TanhDeterministicOutput()

  @property
  def mean(self) -> torch.Tensor:
    """Return the squashed Gaussian location used for deterministic actions."""
    return torch.tanh(self._distribution.mean)  # type: ignore[union-attr]

  @property
  def entropy(self) -> torch.Tensor:
    """Estimate transformed entropy with one reparameterized Gaussian sample."""
    pre_tanh = self._distribution.rsample()  # type: ignore[union-attr]
    entropy = self._distribution.entropy()  # type: ignore[union-attr]
    return (entropy + self._log_abs_det_jacobian(pre_tanh)).sum(dim=-1)

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    """Return the pre-tanh Gaussian parameters used for exact KL divergence."""
    return (
      self._distribution.mean,  # type: ignore[union-attr]
      self._distribution.stddev,  # type: ignore[union-attr]
    )

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    """Compute transformed log-probabilities from unsquashed actions."""
    log_prob = self._distribution.log_prob(outputs)  # type: ignore[union-attr]
    return (log_prob - self._log_abs_det_jacobian(outputs)).sum(dim=-1)

  def kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    """Compute the exact KL between the pre-tanh Gaussian distributions."""
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    return torch.distributions.kl_divergence(
      Normal(old_mean, old_std),
      Normal(new_mean, new_std),
    ).sum(dim=-1)

  @staticmethod
  def _log_abs_det_jacobian(pre_tanh: torch.Tensor) -> torch.Tensor:
    """Compute ``log(1 - tanh(x)^2)`` without saturation."""
    return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))


class _TanhDeterministicOutput(nn.Module):
  """Squash the Gaussian location for deterministic export."""

  def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return torch.tanh(mlp_output)
