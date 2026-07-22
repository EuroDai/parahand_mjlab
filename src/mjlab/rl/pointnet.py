from __future__ import annotations

import copy
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


class PointNetEncoder(nn.Module):
  """Shared per-point MLP followed by max pooling."""

  def __init__(
    self,
    feature_dims: tuple[int, ...] | list[int],
    activation: str,
  ) -> None:
    super().__init__()
    if not feature_dims:
      raise ValueError("PointNet feature_dims must not be empty.")
    layers: list[nn.Module] = []
    input_dim = 3
    for output_dim in feature_dims:
      layers.append(nn.Linear(input_dim, output_dim))
      layers.append(copy.deepcopy(resolve_nn_activation(activation)))
      input_dim = output_dim
    self.mlp = nn.Sequential(*layers)
    self.output_dim = feature_dims[-1]

  def forward(self, points: torch.Tensor) -> torch.Tensor:
    return self.mlp(points).max(dim=-2).values


class PointNetModel(MLPModel):
  """RSL-RL model that replaces each frame's point cloud with PointNet features."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256),
    activation: str = "swish",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    pointnet_cfg: dict[str, Any] | None = None,
  ) -> None:
    if pointnet_cfg is None:
      raise ValueError("PointNetModel requires pointnet_cfg.")
    self.point_cloud_offset = int(pointnet_cfg.get("point_cloud_offset", 6))
    self.point_cloud_points = int(pointnet_cfg.get("point_cloud_points", 64))
    self.point_dim = int(pointnet_cfg.get("point_dim", 3))
    feature_dims = tuple(pointnet_cfg.get("feature_dims", (32, 64, 128)))
    self.point_feature_dim = feature_dims[-1]
    self._history_length = 0
    self._frame_dim = 0
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self.pointnet = PointNetEncoder(feature_dims, activation)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    frame_obs = obs[self.obs_groups[0]]
    leading_shape = frame_obs.shape[:-2]
    normalized = self.obs_normalizer(
      frame_obs.reshape(*leading_shape, self.obs_dim)
    ).reshape(*leading_shape, self._history_length, self._frame_dim)

    point_size = self.point_cloud_points * self.point_dim
    point_end = self.point_cloud_offset + point_size
    points = normalized[..., self.point_cloud_offset : point_end].reshape(
      *leading_shape,
      self._history_length,
      self.point_cloud_points,
      self.point_dim,
    )
    point_features = self.pointnet(points)
    encoded_frames = torch.cat(
      (
        normalized[..., : self.point_cloud_offset],
        point_features,
        normalized[..., point_end:],
      ),
      dim=-1,
    )
    return encoded_frames.flatten(start_dim=-2)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      frame_obs = obs[self.obs_groups[0]]
      cast(Any, self.obs_normalizer).update(frame_obs.flatten(start_dim=-2))

  def as_jit(self) -> nn.Module:
    return _ExportPointNetModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _ExportPointNetModel(self)

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    if len(active_obs_groups) != 1:
      raise ValueError("PointNetModel requires exactly one observation group.")
    obs_shape = obs[active_obs_groups[0]].shape
    if len(obs_shape) != 3:
      raise ValueError(
        "PointNetModel expects observations shaped (batch, history, frame), "
        f"got {obs_shape}."
      )
    self._history_length = obs_shape[-2]
    self._frame_dim = obs_shape[-1]
    point_end = self.point_cloud_offset + self.point_cloud_points * self.point_dim
    if point_end > self._frame_dim:
      raise ValueError("Point cloud slice exceeds the observation frame.")
    return active_obs_groups, self._history_length * self._frame_dim

  def _get_latent_dim(self) -> int:
    point_size = self.point_cloud_points * self.point_dim
    encoded_frame_dim = self._frame_dim - point_size + self.point_feature_dim
    return self._history_length * encoded_frame_dim


class _ExportPointNetModel(nn.Module):
  def __init__(self, model: PointNetModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.pointnet = copy.deepcopy(model.pointnet)
    self.mlp = copy.deepcopy(model.mlp)
    self.history_length = model._history_length
    self.frame_dim = model._frame_dim
    self.point_cloud_offset = model.point_cloud_offset
    self.point_cloud_points = model.point_cloud_points
    self.point_dim = model.point_dim
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, frame_obs: torch.Tensor) -> torch.Tensor:
    batch_size = frame_obs.shape[0]
    normalized = self.obs_normalizer(frame_obs.reshape(batch_size, -1)).reshape(
      batch_size, self.history_length, self.frame_dim
    )
    point_size = self.point_cloud_points * self.point_dim
    point_end = self.point_cloud_offset + point_size
    points = normalized[..., self.point_cloud_offset : point_end].reshape(
      batch_size,
      self.history_length,
      self.point_cloud_points,
      self.point_dim,
    )
    point_features = self.pointnet(points)
    encoded_frames = torch.cat(
      (
        normalized[..., : self.point_cloud_offset],
        point_features,
        normalized[..., point_end:],
      ),
      dim=-1,
    )
    output = self.mlp(encoded_frames.flatten(start_dim=-2))
    return self.deterministic_output(output)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.history_length, self.frame_dim),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass
