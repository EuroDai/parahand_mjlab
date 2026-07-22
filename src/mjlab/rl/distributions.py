from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import Distribution
from torch.distributions import Normal


class TanhGaussianDistribution(Distribution):
  """State-dependent diagonal Gaussian followed by a tanh transform."""

  def __init__(
    self,
    output_dim: int,
    min_std: float = 0.001,
    var_scale: float = 1.0,
  ) -> None:
    super().__init__(output_dim)
    self.min_std = min_std
    self.var_scale = var_scale
    self._distribution: Normal | None = None
    self._raw_sample: torch.Tensor | None = None
    Normal.set_default_validate_args(False)

  def update(self, mlp_output: torch.Tensor) -> None:
    mean, raw_std = torch.unbind(mlp_output, dim=-2)
    std = (torch.nn.functional.softplus(raw_std) + self.min_std) * self.var_scale
    self._distribution = Normal(mean, std)
    self._raw_sample = None

  def sample(self) -> torch.Tensor:
    assert self._distribution is not None
    self._raw_sample = self._distribution.sample()
    return torch.tanh(self._raw_sample)

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    mean = mlp_output[..., 0, :]
    return torch.tanh(mean)

  def as_deterministic_output_module(self) -> nn.Module:
    return _TanhMeanOutput()

  @property
  def input_dim(self) -> list[int]:
    return [2, self.output_dim]

  @property
  def mean(self) -> torch.Tensor:
    assert self._distribution is not None
    return torch.tanh(self._distribution.mean)

  @property
  def std(self) -> torch.Tensor:
    assert self._distribution is not None
    return self._distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    assert self._distribution is not None
    raw_sample = self._raw_sample
    if raw_sample is None:
      raw_sample = self._distribution.sample()
    log_det_jacobian = 2.0 * (
      math.log(2.0) - raw_sample - torch.nn.functional.softplus(-2.0 * raw_sample)
    )
    return (self._distribution.entropy() + log_det_jacobian).sum(dim=-1)

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    assert self._distribution is not None
    return self._distribution.mean, self._distribution.stddev

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    assert self._distribution is not None
    bounded_outputs = outputs.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    raw_outputs = torch.atanh(bounded_outputs)
    log_det_jacobian = 2.0 * (
      math.log(2.0) - raw_outputs - torch.nn.functional.softplus(-2.0 * raw_outputs)
    )
    return (self._distribution.log_prob(raw_outputs) - log_det_jacobian).sum(dim=-1)

  def kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    return torch.distributions.kl_divergence(
      Normal(old_mean, old_std), Normal(new_mean, new_std)
    ).sum(dim=-1)


class _TanhMeanOutput(nn.Module):
  def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return torch.tanh(mlp_output[..., 0, :])
