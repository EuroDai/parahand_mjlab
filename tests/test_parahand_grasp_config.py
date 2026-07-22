import math

import mujoco
import torch

from mjlab.entity import VariantEntityCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  get_object_spec,
  parahand_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.config.parahand.rl_cfg import (
  parahand_grasp_object_ppo_runner_cfg,
)
from mjlab.tasks.parahand_grasp.mdp.consts import PRIMITIVE_OBJECTS
from mjlab.tasks.parahand_grasp.mdp.observations import _sample_model_surface_points


def test_object_spec_contains_all_curriculum_primitives():
  for obj in PRIMITIVE_OBJECTS:
    model = get_object_spec(obj).compile()
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, 0)

    assert geom_name == "object_geom"
    assert mujoco.mjtGeom(model.geom_type[0]) == mujoco.mjtGeom.mjGEOM_MESH


def test_mesh_surface_sampler_uses_compiled_mesh_faces():
  model = get_object_spec(PRIMITIVE_OBJECTS[0]).compile()

  points = _sample_model_surface_points(model, num_points=256, device="cpu")

  assert points.shape == (256, 3)
  assert torch.isfinite(points).all()
  assert torch.all(points.abs() <= 0.03 + 1.0e-6)


def test_initial_object_curriculum_uses_uniform_fixed_variants():
  cfg = parahand_grasp_object_env_cfg()

  command_cfg = cfg.commands["object_pose"]
  reset_cfg = cfg.events["reset_object_pose"]
  curriculum_cfg = cfg.curriculum["object_pose"]
  object_cfg = cfg.scene.entities["object"]

  assert isinstance(object_cfg, VariantEntityCfg)
  assert tuple(object_cfg.variants) == tuple(obj.name for obj in PRIMITIVE_OBJECTS)
  assert object_cfg.assignment is None
  assert isinstance(command_cfg, LiftingCommandCfg)
  assert command_cfg.object_pose_range is None
  assert not hasattr(reset_cfg.func, "model_fields")
  assert all(event_cfg.mode == "reset" for event_cfg in cfg.events.values())
  assert curriculum_cfg.params["stages"] == [
    {
      "step": 0,
      "position_noise": (0.05, 0.1),
      "yaw_range": (-0.5 * math.pi, 0.5 * math.pi),
    }
  ]


def test_training_config_matches_playground_horizon_and_ppo_settings():
  env_cfg = parahand_grasp_object_env_cfg()
  play_env_cfg = parahand_grasp_object_env_cfg(play=True)
  agent_cfg = parahand_grasp_object_ppo_runner_cfg()

  assert env_cfg.scene.num_envs == 4096
  assert play_env_cfg.scene.num_envs == 1
  assert env_cfg.sim.mujoco.timestep == 0.005
  assert env_cfg.decimation == 10
  assert env_cfg.episode_length_s == 20
  assert env_cfg.episode_length_s / (env_cfg.sim.mujoco.timestep * 10) == 400
  assert agent_cfg.num_steps_per_env == 32
  assert agent_cfg.max_iterations == 763
  assert agent_cfg.algorithm.num_learning_epochs == 2
  assert agent_cfg.algorithm.num_mini_batches == 32
  assert agent_cfg.algorithm.learning_rate == 1.0e-4
  assert agent_cfg.algorithm.entropy_coef == 0.005
  assert agent_cfg.algorithm.value_loss_coef == 0.5
  assert not agent_cfg.algorithm.use_clipped_value_loss
  assert agent_cfg.actor.distribution_cfg == {
    "class_name": "mjlab.rl.distributions:TanhGaussianDistribution",
    "min_std": 0.001,
    "var_scale": 1.0,
  }
