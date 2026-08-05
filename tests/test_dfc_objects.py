from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trimesh
import tyro

from mjlab.entity import VariantEntityCfg
from mjlab.scripts.play import (
  PLAY_TYRO_FLAGS,
  PlayConfig,
  _make_play_runner_cfg,
  _make_stage2_play_env_cfg,
  _make_unseen_test_play_env_cfg,
)
from mjlab.scripts.train import (
  TrainConfig,
  _prepare_training_curriculum_stage,
  _set_training_curriculum_stage,
)
from mjlab.tasks.manipulation.mdp.commands import LiftingCommandCfg
from mjlab.tasks.parahand_grasp.config.parahand.env_cfgs import (
  parahand_only_grasp_object_env_cfg,
)
from mjlab.tasks.parahand_grasp.config.parahand.rl_cfg import (
  parahand_only_grasp_object_ppo_runner_cfg,
)
from mjlab.tasks.parahand_grasp.dfc_objects import (
  load_dfc_variants,
  load_robustdex_variants,
  make_dataset_train_env_cfg,
  make_dfc_eval_env_cfg,
  make_dfc_variant_spec,
  select_training_shard,
)
from mjlab.tasks.parahand_grasp.mdp.observations import (
  _load_preprocessed_point_cloud,
)
from mjlab.tasks.parahand_grasp.rl.runner import (
  ParaHandOnPolicyRunner,
  _fixed_eval_rng,
)


def _assert_final_primitive_randomization(cfg) -> None:
  assert cfg.events["reset_gravity"].params["curriculum_stage"] == 6
  assert cfg.events["reset_table_height"].params["curriculum_stage"] == 6
  assert cfg.events["reset_teacher_physics"].params["curriculum_stage"] == 6
  robot_cfg = cfg.events["reset_robot_joints"].params
  assert robot_cfg["curriculum_stage"] == 6
  assert robot_cfg["position_range"] == (-0.5, 0.5)
  assert robot_cfg["palm_height_range"] == pytest.approx((0.2, 0.4))
  assert robot_cfg["palm_joint_ranges"] == {
    "palm_translation_x": (-0.1, 0.2),
    "palm_translation_y": (-0.2, 0.2),
    "palm_rotation_x": (-0.5, 0.5),
    "palm_rotation_y": (-0.5, 0.5),
    "palm_rotation_z": (-0.5, 0.5),
  }
  assert cfg.actions["tendon_length"].reset_target_range == (-0.05, 0.05)


@pytest.fixture
def dfc_dataset(tmp_path: Path) -> Path:
  dataset_dir = tmp_path / "DFCData"
  processed_dir = dataset_dir / "processed" / "v1"
  object_dir = processed_dir / "objects" / "sem__Test-a"
  collision_dir = object_dir / "collision"
  collision_dir.mkdir(parents=True)

  mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
  mesh.export(object_dir / "mesh.obj")
  mesh.export(collision_dir / "part_000.obj")
  points = np.asarray(mesh.sample(1024), dtype=np.float32)
  np.save(object_dir / "surface_unit_1024.npy", points)

  record = {
    "object_code": "sem/Test-a",
    "safe_id": "sem__Test-a",
    "directory": "objects/sem__Test-a",
    "unit_mesh": "mesh.obj",
    "collision_meshes": ["collision/part_000.obj"],
    "unit_surface_points": "surface_unit_1024.npy",
    "point_bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
  }
  manifest = {
    "format_version": 1,
    "scale_mode": "unit_sphere_runtime_cfg",
    "splits": {
      "test_set_unseen_cat": [
        {"object_code": "sem/Test-a", "scale": 0.06},
        {"object_code": "sem/Test-a", "scale": 0.12},
      ]
    },
    "objects": [record],
  }
  (processed_dir / "manifest.json").write_text(json.dumps(manifest))
  return dataset_dir


@pytest.fixture
def robustdex_dataset(tmp_path: Path) -> Path:
  dataset_dir = tmp_path / "RobustDexGrasp"
  processed_dir = dataset_dir / "processed" / "v1"
  object_dir = processed_dir / "test_box"
  collision_dir = object_dir / "collision"
  collision_dir.mkdir(parents=True)
  mesh = trimesh.creation.box(extents=(0.04, 0.06, 0.08))
  mesh.export(object_dir / "mesh.obj")
  mesh.export(collision_dir / "part_000.obj")
  points = np.asarray(mesh.sample(1024), dtype=np.float32)
  np.save(object_dir / "surface_1024.npy", points)
  manifest = {
    "format_version": 1,
    "source": "RobustDexGrasp",
    "objects": [
      {
        "name": "test_box",
        "mesh": "test_box/mesh.obj",
        "collision_meshes": ["test_box/collision/part_000.obj"],
        "surface_points": "test_box/surface_1024.npy",
        "floor_offset": 0.04,
      }
    ],
  }
  (processed_dir / "manifest.json").write_text(json.dumps(manifest))
  return dataset_dir


def test_load_dfc_variants_preserves_scale_entries(dfc_dataset: Path):
  variants = load_dfc_variants(dfc_dataset)

  assert [variant.scale for variant in variants] == [0.06, 0.12]
  assert variants[0].name == "sem__Test-a__s006"
  assert variants[1].name == "sem__Test-a__s012"
  assert variants[0].floor_offset > 0.0


def test_dfc_variant_spec_applies_runtime_scale(dfc_dataset: Path):
  variants = load_dfc_variants(dfc_dataset)

  small_model = make_dfc_variant_spec(variants[0]).compile()
  large_model = make_dfc_variant_spec(variants[1]).compile()

  assert small_model.ngeom == 2
  assert large_model.geom_rbound[0] == pytest.approx(small_model.geom_rbound[0] * 2.0)
  assert large_model.body_mass[1] == pytest.approx(small_model.body_mass[1] * 8.0)
  assert small_model.geom_friction[1].tolist() == pytest.approx([1.0, 0.002, 0.0001])
  assert small_model.geom_solref[1].tolist() == pytest.approx([0.01, 1.0])
  assert small_model.geom_solimp[1].tolist() == pytest.approx(
    [0.9, 0.95, 0.001, 0.5, 2.0]
  )


def test_observation_deterministically_reduces_preprocessed_cloud(
  tmp_path: Path,
):
  source = np.arange(1024 * 3, dtype=np.float32).reshape(1024, 3)
  point_path = tmp_path / "surface_1024.npy"
  np.save(point_path, source)

  points = _load_preprocessed_point_cloud(point_path, 0.5, 256, "cpu")

  expected_indices = np.arange(2, 1024, 4)
  np.testing.assert_array_equal(points.numpy(), source[expected_indices] * 0.5)


def test_load_robustdex_variants_uses_preprocessed_metric_scale(
  robustdex_dataset: Path,
):
  variants = load_robustdex_variants(robustdex_dataset)

  assert len(variants) == 1
  assert variants[0].name == "robustdex__test_box"
  assert variants[0].scale == 1.0
  assert variants[0].dataset == "robustdex"
  assert make_dfc_variant_spec(variants[0]).compile().ngeom == 2


def test_dfc_shards_are_deterministic_and_rank_disjoint(dfc_dataset: Path):
  base = load_dfc_variants(dfc_dataset)
  variants = tuple(base[index % len(base)] for index in range(12))
  variants = tuple(
    replace(variant, name=f"{variant.name}_{index}")
    for index, variant in enumerate(variants)
  )

  rank_zero = select_training_shard(
    variants,
    shard_size_per_rank=4,
    rank=0,
    world_size=2,
    shard_index=0,
    seed=7,
  )
  rank_one = select_training_shard(
    variants,
    shard_size_per_rank=4,
    rank=1,
    world_size=2,
    shard_index=0,
    seed=7,
  )
  repeated = select_training_shard(
    variants,
    shard_size_per_rank=4,
    rank=0,
    world_size=2,
    shard_index=0,
    seed=7,
  )

  assert rank_zero == repeated
  assert {item.name for item in rank_zero}.isdisjoint(item.name for item in rank_one)


def test_make_dfc_eval_env_cfg_is_eval_only(dfc_dataset: Path):
  training_cfg = parahand_only_grasp_object_env_cfg()

  eval_cfg = make_dfc_eval_env_cfg(
    training_cfg,
    dfc_dataset,
    "test_set_unseen_cat",
    0.05,
  )

  assert eval_cfg.scene.num_envs == 2
  object_cfg = eval_cfg.scene.entities["object"]
  assert isinstance(object_cfg, VariantEntityCfg)
  assert len(object_cfg.variants) == 2
  assert list(eval_cfg.observations) == ["actor"]
  assert eval_cfg.observations["actor"].enable_corruption is True
  assert eval_cfg.rewards == {}
  assert eval_cfg.curriculum == {}
  assert list(eval_cfg.metrics) == ["unseen_success"]
  assert eval_cfg.sim.nconmax == 1024
  assert eval_cfg.sim.njmax == 4096
  reset_cfg = eval_cfg.events["reset_object_pose"]
  assert reset_cfg.func.__name__ == "reset_dropped_mesh_object_pose"
  assert reset_cfg.params["drop_height_range"] == (0.10, 0.15)
  assert reset_cfg.params["table_height_event_name"] == "reset_table_height"
  _assert_final_primitive_randomization(eval_cfg)


def test_make_dfc_eval_env_cfg_can_retain_critic_for_viewer(dfc_dataset: Path):
  training_cfg = parahand_only_grasp_object_env_cfg()

  eval_cfg = make_dfc_eval_env_cfg(
    training_cfg,
    dfc_dataset,
    "test_set_unseen_cat",
    0.05,
    actor_only=False,
  )

  assert list(eval_cfg.observations) == ["actor", "critic"]
  assert eval_cfg.rewards == {}
  assert eval_cfg.observations["actor"].enable_corruption is True
  assert eval_cfg.observations["critic"].enable_corruption is False
  for group_cfg in eval_cfg.observations.values():
    assert group_cfg.terms["object_point_cloud_b"].params[
      "variant_point_cloud_scales"
    ] == (0.06, 0.12)


def test_make_dataset_train_env_cfg_uses_random_drop_reset(dfc_dataset: Path):
  training_cfg = parahand_only_grasp_object_env_cfg()
  variants = load_dfc_variants(dfc_dataset)

  stage2_cfg = make_dataset_train_env_cfg(
    training_cfg,
    variants,
    drop_height_range=(0.10, 0.15),
    position_noise=(0.08, 0.08),
    clearance=0.003,
  )

  reset_cfg = stage2_cfg.events["reset_object_pose"]
  assert reset_cfg.func.__name__ == "reset_dropped_mesh_object_pose"
  assert reset_cfg.params["drop_height_range"] == (0.10, 0.15)
  assert reset_cfg.params["position_noise"] == (0.08, 0.08)
  assert reset_cfg.params["clearance"] == 0.003
  assert reset_cfg.params["table_height_event_name"] == "reset_table_height"
  assert stage2_cfg.curriculum == {}
  _assert_final_primitive_randomization(stage2_cfg)
  command_cfg = stage2_cfg.commands["object_pose"]
  assert isinstance(command_cfg, LiftingCommandCfg)
  target_range = command_cfg.target_position_range
  assert target_range.x == (-0.1, 0.1)
  assert target_range.y == (-0.1, 0.1)
  assert target_range.z == (0.35, 0.55)
  assert stage2_cfg.scene.num_envs == training_cfg.scene.num_envs
  assert stage2_cfg.sim.nconmax == 256
  assert stage2_cfg.sim.njmax == 2048
  assert stage2_cfg.observations["actor"].terms["object_point_cloud_b"].params[
    "variant_point_cloud_scales"
  ] == (0.06, 0.12)


def test_play_stage_two_builds_configured_dataset_environment(dfc_dataset: Path):
  env_cfg = parahand_only_grasp_object_env_cfg(play=True)
  agent_cfg = parahand_only_grasp_object_ppo_runner_cfg()
  agent_cfg.stage2_dfc_dataset_dir = str(dfc_dataset)
  agent_cfg.stage2_dfc_split = "test_set_unseen_cat"
  agent_cfg.stage2_shard_size_per_rank = 1

  stage2_cfg = _make_stage2_play_env_cfg(env_cfg, agent_cfg, num_envs=3)

  assert stage2_cfg.scene.num_envs == 3
  object_cfg = stage2_cfg.scene.entities["object"]
  assert isinstance(object_cfg, VariantEntityCfg)
  assert len(object_cfg.variants) == 1
  reset_cfg = stage2_cfg.events["reset_object_pose"]
  assert reset_cfg.func.__name__ == "reset_dropped_mesh_object_pose"
  assert reset_cfg.params["drop_height_range"] == (0.10, 0.15)
  assert stage2_cfg.events["reset_gravity"].params["curriculum_stage"] == 6
  assert stage2_cfg.curriculum == {}
  _assert_final_primitive_randomization(stage2_cfg)


def test_play_unseen_test_builds_viewable_eval_environment(dfc_dataset: Path):
  env_cfg = parahand_only_grasp_object_env_cfg(play=True)
  agent_cfg = parahand_only_grasp_object_ppo_runner_cfg()
  agent_cfg.unseen_eval_dataset_dir = str(dfc_dataset)

  eval_cfg = _make_unseen_test_play_env_cfg(env_cfg, agent_cfg)

  assert eval_cfg.scene.num_envs == 2
  assert eval_cfg.seed == agent_cfg.unseen_eval_seed
  assert list(eval_cfg.observations) == ["actor", "critic"]
  assert eval_cfg.observations["actor"].enable_corruption is True
  assert list(eval_cfg.rewards) == ["object_point_cloud_debug"]
  assert eval_cfg.rewards["object_point_cloud_debug"].weight == 0.0
  assert list(eval_cfg.metrics) == ["unseen_success"]
  assert eval_cfg.commands["object_pose"].debug_vis is True
  _assert_final_primitive_randomization(eval_cfg)


def test_play_unseen_test_is_value_free_flag():
  cfg = tyro.cli(
    PlayConfig,
    args=["--unseen-test"],
    config=PLAY_TYRO_FLAGS,
  )

  assert cfg.unseen_test is True


def test_play_disables_training_time_unseen_evaluator():
  agent_cfg = parahand_only_grasp_object_ppo_runner_cfg()

  runner_cfg = _make_play_runner_cfg(agent_cfg)

  assert runner_cfg["unseen_eval"] is False
  assert agent_cfg.unseen_eval is True


def test_train_curriculum_stage_six_keeps_final_primitive_stage():
  env_cfg = parahand_only_grasp_object_env_cfg()
  agent_cfg = parahand_only_grasp_object_ppo_runner_cfg()
  cfg = TrainConfig(env=env_cfg, agent=agent_cfg, curriculum_stage=6)

  _prepare_training_curriculum_stage(cfg)

  assert agent_cfg.stage2_start_immediately is False
  _assert_final_primitive_randomization(env_cfg)


def test_train_curriculum_stage_seven_enables_dataset_stage():
  env_cfg = parahand_only_grasp_object_env_cfg()
  agent_cfg = parahand_only_grasp_object_ppo_runner_cfg()
  cfg = TrainConfig(env=env_cfg, agent=agent_cfg, curriculum_stage=7)

  _prepare_training_curriculum_stage(cfg)

  assert agent_cfg.stage2_start_immediately is True
  _assert_final_primitive_randomization(env_cfg)


def test_train_curriculum_stage_sets_live_curriculum_term():
  term = Mock()
  manager = SimpleNamespace(
    active_terms=["object_lesson"],
    get_term_cfg=Mock(return_value=SimpleNamespace(func=term)),
  )
  env = SimpleNamespace(curriculum_manager=manager)

  _set_training_curriculum_stage(cast(Any, env), 7)

  term.set_stage.assert_called_once_with(7)


def test_fixed_eval_rng_restores_training_rng():
  torch.manual_seed(7)
  expected = torch.rand(3)
  torch.manual_seed(7)

  with _fixed_eval_rng(123):
    first_eval = torch.rand(3)
  actual = torch.rand(3)
  with _fixed_eval_rng(123):
    second_eval = torch.rand(3)

  assert torch.equal(actual, expected)
  assert torch.equal(first_eval, second_eval)


def test_runner_logs_only_unseen_success_rate():
  runner = ParaHandOnPolicyRunner.__new__(ParaHandOnPolicyRunner)
  evaluator = Mock()
  evaluator.evaluate.return_value = 0.625
  writer = Mock()
  runner._unseen_evaluator = evaluator
  runner._unseen_eval_interval = 300
  runner._unseen_eval_start_stage = 3
  runner._logical_curriculum_stage = Mock(return_value=3)
  runner.logger = cast(Any, SimpleNamespace(writer=writer))
  runner._training_log = Mock()

  runner._log_with_unseen_eval(
    it=600,
    start_it=0,
    total_it=10,
    collect_time=1.0,
    learn_time=2.0,
    loss_dict={},
    learning_rate=1.0e-4,
    action_std=torch.ones(1),
    rnd_weight=None,
  )

  writer.add_scalar.assert_called_once_with(
    "eval/unseen_success_rate",
    0.625,
    600,
  )
  runner._training_log.assert_called_once()


def test_runner_skips_unseen_success_before_curriculum_stage_three():
  runner = ParaHandOnPolicyRunner.__new__(ParaHandOnPolicyRunner)
  evaluator = Mock()
  writer = Mock()
  runner._unseen_evaluator = evaluator
  runner._unseen_eval_interval = 300
  runner._unseen_eval_start_stage = 3
  runner._logical_curriculum_stage = Mock(return_value=2)
  runner.logger = cast(Any, SimpleNamespace(writer=writer))
  runner._training_log = Mock()

  runner._log_with_unseen_eval(
    it=600,
    start_it=0,
    total_it=10,
    collect_time=1.0,
    learn_time=2.0,
    loss_dict={},
    learning_rate=1.0e-4,
    action_std=torch.ones(1),
    rnd_weight=None,
  )

  evaluator.evaluate.assert_not_called()
  writer.add_scalar.assert_not_called()
  runner._training_log.assert_called_once()


def test_runner_stage2_update_schedule_preserves_primitive_quarter():
  runner = ParaHandOnPolicyRunner.__new__(ParaHandOnPolicyRunner)
  runner.cfg = {"stage2_primitive_ratio": 0.25}

  domains = []
  for update in range(8):
    runner._stage2_update_count = update
    domains.append(runner._stage2_domain())

  assert domains == [
    "dataset",
    "dataset",
    "dataset",
    "primitive",
    "dataset",
    "dataset",
    "dataset",
    "primitive",
  ]


def test_runner_preserves_single_rank_rollout_error():
  runner = ParaHandOnPolicyRunner.__new__(ParaHandOnPolicyRunner)
  runner.is_distributed = False

  with pytest.raises(ValueError, match="actor observation contains NaN"):
    runner._synchronize_rollout_error(ValueError("actor observation contains NaN"))


def test_runner_propagates_remote_rollout_error(monkeypatch):
  runner = ParaHandOnPolicyRunner.__new__(ParaHandOnPolicyRunner)
  runner.is_distributed = True
  runner.device = "cpu"
  runner.gpu_global_rank = 0

  def mark_remote_failure(failed, op):
    del op
    failed.fill_(1)

  monkeypatch.setattr(
    "mjlab.tasks.parahand_grasp.rl.runner.dist.all_reduce",
    mark_remote_failure,
  )

  with pytest.raises(RuntimeError, match="another distributed rank"):
    runner._synchronize_rollout_error(None)
