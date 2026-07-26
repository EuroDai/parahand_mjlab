from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def parahand_grasp_object_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="mjlab.rl.pointnet:PointNetModel",
      hidden_dims=(512, 256, 256, 128),
      activation="swish",
      obs_normalization=True,
      pointnet_cfg={
        "point_cloud_offset": 6,
        "point_cloud_points": 64,
        "point_dim": 3,
        "feature_dims": (32, 64, 128),
      },
      distribution_cfg={
        "class_name": ("mjlab.rl.distributions:StateDependentTanhGaussianDistribution"),
        "init_std": 1.0,
        "min_std": 0.001,
      },
    ),
    critic=RslRlModelCfg(
      class_name="mjlab.rl.pointnet:PointNetModel",
      hidden_dims=(512, 256, 256, 128),
      activation="swish",
      obs_normalization=True,
      pointnet_cfg={
        "point_cloud_offset": 6,
        "point_cloud_points": 64,
        "point_dim": 3,
        "feature_dims": (32, 64, 128),
      },
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=2,
      num_mini_batches=32,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="parahand_grasp_object",
    save_interval=76,
    clip_actions=1.0,
    num_steps_per_env=32,
    max_iterations=763,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
  )


def parahand_only_grasp_object_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = parahand_grasp_object_ppo_runner_cfg()
  cfg.experiment_name = "parahand_only_grasp_object"
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.schedule = "fixed"
  return cfg
