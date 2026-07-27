from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab.rl.distributions import TanhGaussianDistribution

_distributed = cast(Any, torch.distributed)


class StablePPO(PPO):
  """PPO with guards against non-finite optimization updates."""

  def __init__(
    self,
    *args: Any,
    log_ratio_clip: float = 20.0,
    **kwargs: Any,
  ) -> None:
    if log_ratio_clip <= 0.0:
      raise ValueError(f"log_ratio_clip must be positive, got {log_ratio_clip}.")
    super().__init__(*args, **kwargs)
    self.log_ratio_clip = log_ratio_clip
    self.max_learning_rate = self.learning_rate

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Store raw Gaussian actions while sending squashed actions to the environment."""
    raw_actions = super().act(obs)
    distribution = self.actor.distribution
    if isinstance(distribution, TanhGaussianDistribution):
      return distribution.postprocess(raw_actions)
    return raw_actions

  def update(self) -> dict[str, float]:
    """Run PPO updates while skipping non-finite losses and gradients."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None
    successful_updates = 0
    skipped_updates = 0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      observations = cast(TensorDict, batch.observations)
      advantages = cast(torch.Tensor, batch.advantages)
      original_batch_size = observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
          )
          advantages = cast(torch.Tensor, batch.advantages)

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      observations = cast(TensorDict, batch.observations)
      advantages = cast(torch.Tensor, batch.advantages)
      old_distribution_params = cast(
        tuple[torch.Tensor, ...], batch.old_distribution_params
      )
      old_actions_log_prob = cast(torch.Tensor, batch.old_actions_log_prob)
      self.actor(
        observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(
        param[:original_batch_size] for param in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      self._adapt_learning_rate(old_distribution_params, distribution_params)

      ratio = self._compute_ratio(
        actions_log_prob,
        torch.squeeze(old_actions_log_prob),
        self.log_ratio_clip,
      )
      surrogate = -torch.squeeze(advantages) * ratio
      surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
        ratio,
        1.0 - self.clip_param,
        1.0 + self.clip_param,
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      rnd_loss = (
        self.rnd.compute_loss(cast(TensorDict, observations[:original_batch_size]))
        if self.rnd
        else None
      )

      symmetry_loss = None
      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(
          self.actor, batch, original_batch_size
        )
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      finite_losses = [loss]
      if rnd_loss is not None:
        finite_losses.append(rnd_loss)
      if not self._all_ranks_finite(finite_losses):
        self._zero_grad()
        skipped_updates += 1
        continue

      self._zero_grad()
      loss.backward()
      if rnd_loss is not None:
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      parameters = list(self.actor.parameters()) + list(self.critic.parameters())
      if self.rnd:
        parameters.extend(self.rnd.parameters())
      if not self._all_ranks_gradients_finite(parameters):
        self._zero_grad()
        skipped_updates += 1
        continue

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd and rnd_loss is not None:
        self.rnd.optimizer.step()

      successful_updates += 1
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      if mean_rnd_loss is not None and rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None and symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    divisor = max(successful_updates, 1)
    loss_dict = {
      "value": mean_value_loss / divisor,
      "surrogate": mean_surrogate_loss / divisor,
      "entropy": mean_entropy / divisor,
      "skipped_updates": float(skipped_updates),
    }
    if mean_rnd_loss is not None:
      loss_dict["rnd"] = mean_rnd_loss / divisor
    if mean_symmetry_loss is not None:
      loss_dict["symmetry"] = mean_symmetry_loss / divisor

    self.storage.clear()
    return loss_dict

  def _adapt_learning_rate(
    self,
    old_distribution_params: tuple[torch.Tensor, ...],
    distribution_params: tuple[torch.Tensor, ...],
  ) -> None:
    if self.desired_kl is None or self.schedule != "adaptive":
      return

    with torch.inference_mode():
      kl = self.actor.get_kl_divergence(old_distribution_params, distribution_params)
      kl_mean = torch.mean(kl)

      if self.is_multi_gpu:
        _distributed.all_reduce(kl_mean, op=_distributed.ReduceOp.SUM)
        kl_mean /= self.gpu_world_size

      if self.gpu_global_rank == 0 and torch.isfinite(kl_mean):
        if kl_mean > self.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
          self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)

      if self.is_multi_gpu:
        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
        _distributed.broadcast(lr_tensor, src=0)
        self.learning_rate = lr_tensor.item()

      for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.learning_rate

  def _all_ranks_finite(self, values: Iterable[torch.Tensor]) -> bool:
    finite = torch.tensor(
      all(torch.isfinite(value).all().item() for value in values),
      device=self.device,
      dtype=torch.int32,
    )
    if self.is_multi_gpu:
      _distributed.all_reduce(finite, op=_distributed.ReduceOp.MIN)
    return bool(finite.item())

  def _all_ranks_gradients_finite(self, parameters: Iterable[nn.Parameter]) -> bool:
    finite = torch.tensor(
      all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in parameters
      ),
      device=self.device,
      dtype=torch.int32,
    )
    if self.is_multi_gpu:
      _distributed.all_reduce(finite, op=_distributed.ReduceOp.MIN)
    return bool(finite.item())

  def _zero_grad(self) -> None:
    self.optimizer.zero_grad()
    if self.rnd:
      self.rnd.optimizer.zero_grad()

  @staticmethod
  def _compute_ratio(
    actions_log_prob: torch.Tensor,
    old_actions_log_prob: torch.Tensor,
    log_ratio_clip: float,
  ) -> torch.Tensor:
    log_ratio = actions_log_prob - old_actions_log_prob
    return torch.exp(log_ratio.clamp(-log_ratio_clip, log_ratio_clip))
