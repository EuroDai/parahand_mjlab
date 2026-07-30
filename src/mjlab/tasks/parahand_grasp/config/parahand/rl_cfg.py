from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.parahand_grasp.rl import ParaHandOnPolicyRunnerCfg


def parahand_grasp_object_ppo_runner_cfg() -> ParaHandOnPolicyRunnerCfg:
  return ParaHandOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="mjlab.rl.pointnet:PointNetModel",
      hidden_dims=(1024, 1024, 512, 512),
      activation="swish",
      obs_normalization=True,
      pointnet_cfg={
        "point_cloud_offset": 6,
        "point_cloud_points": 256,
        "point_dim": 3,
        "feature_dims": (64, 128, 256),
        "pooling": "max_mean",
        "history_mode": "latest",
        "chunk_size": 256,
        "gradient_checkpointing": True,
      },
      distribution_cfg={
        "class_name": "mjlab.rl.distributions:TanhGaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      class_name="mjlab.rl.pointnet:PointNetModel",
      hidden_dims=(1024, 1024, 512, 512),
      activation="swish",
      obs_normalization=True,
      pointnet_cfg={
        "point_cloud_offset": 6,
        "point_cloud_points": 256,
        "point_dim": 3,
        "feature_dims": (64, 128, 256),
        "pooling": "max_mean",
        "history_mode": "latest",
        "chunk_size": 256,
        "gradient_checkpointing": True,
      },
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="mjlab.rl.ppo:StablePPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=2,
      num_mini_batches=32,
      learning_rate=1.0e-4,
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


def parahand_only_grasp_object_ppo_runner_cfg() -> ParaHandOnPolicyRunnerCfg:
  cfg = parahand_grasp_object_ppo_runner_cfg()
  cfg.experiment_name = "parahand_only_grasp_object"
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.algorithm.schedule = "fixed"
  return cfg
