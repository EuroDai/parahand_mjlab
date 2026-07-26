from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.distribution import Distribution
from torch.distributions import Normal


class StateDependentTanhGaussianDistribution(Distribution):
  """State-dependent diagonal Gaussian followed by ``tanh``.

  The policy network outputs ``[..., 2, action_dim]``: the first slice contains
  Gaussian locations and the second contains scale logits. Scale logits are
  transformed with ``softplus`` to keep standard deviations positive. Samples and
  deterministic outputs are squashed before they reach the environment, while
  log-probabilities include the tanh change-of-variables correction.
  """

  def __init__(
    self,
    output_dim: int,
    init_std: float = 1.0,
    min_std: float = 0.001,
    squash_epsilon: float = 1e-6,
  ) -> None:
    super().__init__(output_dim)
    if min_std <= 0.0:
      raise ValueError(f"min_std must be positive, got {min_std}.")
    if init_std <= min_std:
      raise ValueError(
        f"init_std must be greater than min_std, got {init_std} <= {min_std}."
      )
    if not 0.0 < squash_epsilon < 1.0:
      raise ValueError(f"squash_epsilon must be in (0, 1), got {squash_epsilon}.")
    self.init_std = init_std
    self.min_std = min_std
    self.squash_epsilon = squash_epsilon
    self._distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  def update(self, mlp_output: torch.Tensor) -> None:
    """Build the pre-tanh Gaussian from state-dependent network outputs."""
    mean, scale_logits = torch.unbind(mlp_output, dim=-2)
    std = F.softplus(scale_logits) + self.min_std
    self._distribution = Normal(mean, std)

  def sample(self) -> torch.Tensor:
    """Sample a bounded action from the transformed distribution."""
    pre_tanh = self._distribution.sample()  # type: ignore[union-attr]
    return torch.tanh(pre_tanh).clamp(
      -1.0 + self.squash_epsilon,
      1.0 - self.squash_epsilon,
    )

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    """Squash the Gaussian location for deterministic inference."""
    return torch.tanh(mlp_output[..., 0, :])

  def as_deterministic_output_module(self) -> nn.Module:
    """Return an export-friendly tanh output module."""
    return _StateDependentTanhDeterministicOutput()

  @property
  def input_dim(self) -> list[int]:
    """Return the required ``[location/scale, action]`` network output shape."""
    return [2, self.output_dim]

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

  @property
  def std(self) -> torch.Tensor:
    """Return the state-dependent pre-tanh standard deviation."""
    return self._distribution.stddev  # type: ignore[union-attr]

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    """Compute transformed log-probabilities for bounded actions."""
    safe_outputs = outputs.clamp(
      -1.0 + self.squash_epsilon,
      1.0 - self.squash_epsilon,
    )
    pre_tanh = torch.atanh(safe_outputs)
    log_prob = self._distribution.log_prob(pre_tanh)  # type: ignore[union-attr]
    return (log_prob - self._log_abs_det_jacobian(pre_tanh)).sum(dim=-1)

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

  def init_mlp_weights(self, mlp: nn.Module) -> None:
    """Initialize the scale head to produce ``init_std`` in every state."""
    output_layer = mlp[-2]  # type: ignore[index]
    assert isinstance(output_layer, nn.Linear)
    raw_init_std = math.log(math.expm1(self.init_std - self.min_std))
    torch.nn.init.zeros_(output_layer.weight[self.output_dim :])
    torch.nn.init.constant_(
      output_layer.bias[self.output_dim :],
      raw_init_std,
    )

  @staticmethod
  def _log_abs_det_jacobian(pre_tanh: torch.Tensor) -> torch.Tensor:
    """Compute ``log(1 - tanh(x)^2)`` without saturation."""
    return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))


class _StateDependentTanhDeterministicOutput(nn.Module):
  """Extract and squash the location head for deterministic export."""

  def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return torch.tanh(mlp_output[..., 0, :])
