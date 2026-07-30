import torch
from tensordict import TensorDict

from mjlab.rl.distributions import TanhGaussianDistribution
from mjlab.rl.pointnet import PointNetEncoder, PointNetModel


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
      "pooling": "max_mean",
      "history_mode": "latest",
      "chunk_size": 16,
      "gradient_checkpointing": True,
    },
  )


def test_pointnet_model_shapes():
  model = _make_model()
  obs = TensorDict({"actor": torch.randn(4, 5, 297)}, batch_size=[4])

  assert model.get_latent(obs).shape == (4, 781)
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


def test_pointnet_latest_mode_ignores_older_point_cloud_frames():
  model = _make_model()
  frame_obs = torch.randn(4, 5, 297)
  changed_obs = frame_obs.clone()
  changed_obs[:, :-1, 6:198] = torch.randn_like(changed_obs[:, :-1, 6:198])

  original = TensorDict({"actor": frame_obs}, batch_size=[4])
  changed = TensorDict({"actor": changed_obs}, batch_size=[4])

  torch.testing.assert_close(model.get_latent(original), model.get_latent(changed))


def test_chunked_max_mean_pooling_matches_full_pointnet():
  full = PointNetEncoder((16, 32), "swish", pooling="max_mean")
  chunked = PointNetEncoder(
    (16, 32),
    "swish",
    pooling="max_mean",
    chunk_size=13,
    gradient_checkpointing=True,
  )
  chunked.load_state_dict(full.state_dict())
  points = torch.randn(3, 64, 3, requires_grad=True)

  expected = full(points)
  actual = chunked(points)

  torch.testing.assert_close(actual, expected)
  actual.sum().backward()
  assert points.grad is not None


def test_pointnet_supports_tanh_gaussian_distribution_and_export():
  model = _make_model(
    {
      "class_name": "mjlab.rl.distributions:TanhGaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )
  obs = TensorDict({"actor": torch.randn(4, 5, 297)}, batch_size=[4])

  deterministic_actions = model(obs)
  raw_stochastic_actions = model(obs, stochastic_output=True)
  exported_actions = model.as_jit()(obs["actor"])

  assert isinstance(model.distribution, TanhGaussianDistribution)
  stochastic_actions = model.distribution.postprocess(raw_stochastic_actions)
  assert model.mlp(model.get_latent(obs)).shape == (4, 22)
  assert torch.all(deterministic_actions.abs() < 1.0)
  assert torch.all(stochastic_actions.abs() <= 1.0)
  torch.testing.assert_close(model.output_std, torch.ones_like(model.output_std))
  torch.testing.assert_close(exported_actions, deterministic_actions)
