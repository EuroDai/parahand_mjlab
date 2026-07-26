import torch
from tensordict import TensorDict

from mjlab.rl.pointnet import PointNetModel


def _make_model(
  distribution_cfg: dict[str, object] | None = None,
) -> PointNetModel:
  obs = TensorDict(
    {"actor": torch.randn(4, 5, 297)},
    batch_size=[4],
  )
  return PointNetModel(
    obs=obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=22,
    hidden_dims=(512, 256, 256, 128),
    activation="swish",
    distribution_cfg=distribution_cfg,
    pointnet_cfg={
      "point_cloud_offset": 6,
      "point_cloud_points": 64,
      "feature_dims": (32, 64, 128),
    },
  )


def test_pointnet_model_shapes():
  model = _make_model()
  obs = TensorDict({"actor": torch.randn(4, 5, 297)}, batch_size=[4])

  assert model.get_latent(obs).shape == (4, 1165)
  assert model(obs).shape == (4, 22)


def test_pointnet_features_are_point_permutation_invariant():
  model = _make_model()
  frame_obs = torch.randn(4, 5, 297)
  shuffled_obs = frame_obs.clone()
  points = shuffled_obs[..., 6:198].reshape(4, 5, 64, 3)
  permutation = torch.randperm(64)
  shuffled_obs[..., 6:198] = points[..., permutation, :].reshape(4, 5, 192)

  obs = TensorDict({"actor": frame_obs}, batch_size=[4])
  shuffled = TensorDict({"actor": shuffled_obs}, batch_size=[4])

  torch.testing.assert_close(model.get_latent(obs), model.get_latent(shuffled))


def test_pointnet_supports_tanh_gaussian_distribution_and_export():
  model = _make_model(
    {
      "class_name": ("mjlab.rl.distributions:StateDependentTanhGaussianDistribution"),
      "init_std": 1.0,
      "min_std": 0.001,
    }
  )
  obs = TensorDict({"actor": torch.randn(4, 5, 297)}, batch_size=[4])

  deterministic_actions = model(obs)
  stochastic_actions = model(obs, stochastic_output=True)
  exported_actions = model.as_jit()(obs["actor"])

  assert model.mlp(model.get_latent(obs)).shape == (4, 2, 22)
  assert torch.all(deterministic_actions.abs() < 1.0)
  assert torch.all(stochastic_actions.abs() < 1.0)
  torch.testing.assert_close(model.output_std, torch.ones_like(model.output_std))
  torch.testing.assert_close(exported_actions, deterministic_actions)
