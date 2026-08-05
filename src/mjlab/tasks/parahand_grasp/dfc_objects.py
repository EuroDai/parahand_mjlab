"""Load processed grasp datasets as scale-aware MuJoCo mesh variants."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import mujoco

from mjlab.entity import VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.parahand_grasp import mdp as parahand_mdp
from mjlab.tasks.parahand_grasp.mdp.actions import RelativeTendonLengthActionCfg
from mjlab.tasks.parahand_grasp.mdp.consts import (
  POINT_CLOUD_NOISE_STD_MAX_M,
  PRIMITIVE_DATASET_STAGE,
  primitive_randomization_fraction,
)
from mjlab.utils.noise import GaussianNoiseCfg

_EXPECTED_FORMAT_VERSION = 1
_EXPECTED_SCALE_MODE = "unit_sphere_runtime_cfg"
_DENSITY = 500.0
_ROBUSTDEX_DATASET = "robustdex"
_DFC_DATASET = "dfc"


def apply_primitive_stage_randomization(
  cfg: ManagerBasedRlEnvCfg,
  stage: int,
) -> None:
  """Apply one primitive stage's non-object randomization to an env config."""
  if not 0 <= stage < PRIMITIVE_DATASET_STAGE:
    raise ValueError(
      f"Primitive randomization stage must be in "
      f"[0, {PRIMITIVE_DATASET_STAGE - 1}], got {stage}."
    )
  curriculum_cfg = cfg.curriculum.get("object_lesson")
  if curriculum_cfg is None:
    raise ValueError("Primitive randomization requires an object_lesson curriculum.")

  params = curriculum_cfg.params
  cfg.events[params["event_name"]].params["curriculum_stage"] = stage
  cfg.events[params["table_event_name"]].params["curriculum_stage"] = stage
  cfg.events[params["gravity_event_name"]].params["curriculum_stage"] = stage
  cfg.events[params["physics_event_name"]].params["curriculum_stage"] = stage

  fraction = primitive_randomization_fraction(stage)
  robot_event_cfg = cfg.events[params["robot_event_name"]]
  robot_event_cfg.params["curriculum_stage"] = stage
  robot_event_cfg.params["position_range"] = (-0.5 * fraction, 0.5 * fraction)
  if "palm_joint_ranges" in robot_event_cfg.params:
    robot_event_cfg.params["palm_joint_ranges"] = {
      "palm_translation_x": (-0.1 * fraction, 0.2 * fraction),
      "palm_translation_y": (-0.2 * fraction, 0.2 * fraction),
      "palm_rotation_x": (-0.5 * fraction, 0.5 * fraction),
      "palm_rotation_y": (-0.5 * fraction, 0.5 * fraction),
      "palm_rotation_z": (-0.5 * fraction, 0.5 * fraction),
    }
    robot_event_cfg.params["palm_height_range"] = (
      0.3 - 0.1 * fraction,
      0.3 + 0.1 * fraction,
    )
  tendon_cfg = cfg.actions[params["tendon_action_name"]]
  if not isinstance(tendon_cfg, RelativeTendonLengthActionCfg):
    raise TypeError("Primitive randomization requires a relative tendon action.")
  tendon_cfg.reset_target_range = (
    -0.05 * fraction,
    0.05 * fraction,
  )
  point_cloud_cfg = cfg.observations["actor"].terms["object_point_cloud_b"]
  if point_cloud_cfg.noise is not None and not isinstance(
    point_cloud_cfg.noise, GaussianNoiseCfg
  ):
    raise TypeError("Primitive randomization requires Gaussian point-cloud noise.")
  point_cloud_noise_std = POINT_CLOUD_NOISE_STD_MAX_M * fraction
  point_cloud_cfg.noise = (
    GaussianNoiseCfg(mean=0.0, std=point_cloud_noise_std)
    if point_cloud_noise_std > 0.0
    else None
  )


@dataclass(frozen=True)
class DfcVariant:
  """One processed grasp-object variant.

  DFCData uses a runtime ``scale`` while RobustDexGrasp is already scaled during
  preprocessing and therefore uses a scale of one.
  """

  name: str
  object_code: str
  scale: float
  visual_mesh: Path
  collision_meshes: tuple[Path, ...]
  point_cloud: Path
  floor_offset: float
  dataset: str = _DFC_DATASET


def load_dfc_variants(
  dataset_dir: str | Path,
  split: str = "test_set_unseen_cat",
) -> tuple[DfcVariant, ...]:
  """Load one ordered DFC split from ``processed/v1/manifest.json``."""
  dataset_root = Path(dataset_dir).expanduser().resolve()
  processed_root = dataset_root / "processed" / "v1"
  manifest_path = processed_root / "manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(
      f"DFCData manifest not found at {manifest_path}. Run "
      "'uv run scripts/tools/prepare_dfc_objects.py' first."
    )

  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("format_version") != _EXPECTED_FORMAT_VERSION:
    raise ValueError(
      f"Unsupported DFCData manifest version "
      f"{manifest.get('format_version')!r}; expected {_EXPECTED_FORMAT_VERSION}."
    )
  if manifest.get("scale_mode") != _EXPECTED_SCALE_MODE:
    raise ValueError(
      f"Unsupported DFCData scale mode {manifest.get('scale_mode')!r}; "
      f"expected '{_EXPECTED_SCALE_MODE}'."
    )

  split_entries = manifest.get("splits", {}).get(split)
  if not isinstance(split_entries, list):
    available = sorted(manifest.get("splits", {}))
    raise ValueError(f"DFCData split '{split}' not found; available: {available}.")

  records = {
    record["object_code"]: record
    for record in manifest.get("objects", [])
    if isinstance(record, dict) and "object_code" in record
  }
  variants: list[DfcVariant] = []
  missing: list[str] = []
  for entry in split_entries:
    object_code = str(entry["object_code"])
    scale = float(entry["scale"])
    record = records.get(object_code)
    if record is None:
      missing.append(object_code)
      continue
    object_dir = processed_root / record["directory"]
    visual_mesh = object_dir / record["unit_mesh"]
    collision_meshes = tuple(
      object_dir / relative_path for relative_path in record["collision_meshes"]
    )
    point_cloud = object_dir / record["unit_surface_points"]
    required_paths = (visual_mesh, point_cloud, *collision_meshes)
    absent = [str(path) for path in required_paths if not path.is_file()]
    if absent:
      raise FileNotFoundError(
        f"Processed DFCData object '{object_code}' is incomplete: {absent[:5]}"
      )
    point_bounds = record["point_bounds"]
    floor_offset = max(0.0, -float(point_bounds[0][2]) * scale)
    safe_scale = f"{round(scale * 100):03d}"
    variants.append(
      DfcVariant(
        name=f"{record['safe_id']}__s{safe_scale}",
        object_code=object_code,
        scale=scale,
        visual_mesh=visual_mesh,
        collision_meshes=collision_meshes,
        point_cloud=point_cloud,
        floor_offset=floor_offset,
      )
    )

  if missing:
    raise FileNotFoundError(
      f"{len(missing)} objects from DFCData split '{split}' are absent from "
      f"{manifest_path}: {missing[:5]}. Rerun preprocessing without --limit."
    )
  if not variants:
    raise ValueError(f"DFCData split '{split}' contains no variants.")
  names = [variant.name for variant in variants]
  if len(names) != len(set(names)):
    raise ValueError(f"DFCData split '{split}' contains duplicate variants.")
  return tuple(variants)


def load_robustdex_variants(dataset_dir: str | Path) -> tuple[DfcVariant, ...]:
  """Load all preprocessed RobustDexGrasp objects."""
  dataset_root = Path(dataset_dir).expanduser().resolve()
  processed_root = dataset_root / "processed" / "v1"
  manifest_path = processed_root / "manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(
      f"RobustDexGrasp manifest not found at {manifest_path}. Run "
      "'uv run scripts/tools/prepare_robustdex_objects.py' first."
    )

  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("format_version") != _EXPECTED_FORMAT_VERSION:
    raise ValueError(
      f"Unsupported RobustDexGrasp manifest version "
      f"{manifest.get('format_version')!r}; expected {_EXPECTED_FORMAT_VERSION}."
    )
  if manifest.get("source") != "RobustDexGrasp":
    raise ValueError(
      f"Unexpected RobustDexGrasp manifest source {manifest.get('source')!r}."
    )

  variants: list[DfcVariant] = []
  for record in manifest.get("objects", []):
    if not isinstance(record, dict):
      continue
    name = str(record["name"])
    visual_mesh = processed_root / record["mesh"]
    collision_meshes = tuple(
      processed_root / relative_path for relative_path in record["collision_meshes"]
    )
    point_cloud = processed_root / record["surface_points"]
    required_paths = (visual_mesh, point_cloud, *collision_meshes)
    absent = [str(path) for path in required_paths if not path.is_file()]
    if absent:
      raise FileNotFoundError(
        f"Processed RobustDexGrasp object '{name}' is incomplete: {absent[:5]}"
      )
    variants.append(
      DfcVariant(
        name=f"robustdex__{name}",
        object_code=name,
        scale=1.0,
        visual_mesh=visual_mesh,
        collision_meshes=collision_meshes,
        point_cloud=point_cloud,
        floor_offset=float(record["floor_offset"]),
        dataset=_ROBUSTDEX_DATASET,
      )
    )

  if not variants:
    raise ValueError("RobustDexGrasp manifest contains no processed objects.")
  variants.sort(key=lambda variant: variant.name)
  return tuple(variants)


def load_training_variants(
  dataset: str,
  dataset_dir: str | Path,
  split: str = "train_set",
) -> tuple[DfcVariant, ...]:
  """Load the configured Stage 2 training catalog."""
  normalized = dataset.lower()
  if normalized == _DFC_DATASET:
    return load_dfc_variants(dataset_dir, split)
  if normalized == _ROBUSTDEX_DATASET:
    return load_robustdex_variants(dataset_dir)
  raise ValueError(
    f"Unsupported Stage 2 dataset '{dataset}'; expected 'dfc' or 'robustdex'."
  )


def select_training_shard(
  variants: tuple[DfcVariant, ...],
  *,
  shard_size_per_rank: int,
  rank: int,
  world_size: int,
  shard_index: int,
  seed: int,
) -> tuple[DfcVariant, ...]:
  """Select a deterministic, rank-disjoint cyclic shard from a catalog."""
  if shard_size_per_rank <= 0:
    raise ValueError("shard_size_per_rank must be positive.")
  if not 0 <= rank < world_size:
    raise ValueError(f"rank {rank} must be in [0, {world_size}).")
  if shard_index < 0:
    raise ValueError("shard_index must be non-negative.")
  if len(variants) <= shard_size_per_rank:
    return variants

  order = list(range(len(variants)))
  random.Random(seed).shuffle(order)
  global_span = shard_size_per_rank * world_size
  global_start = (shard_index * global_span) % len(order)
  rank_start = global_start + rank * shard_size_per_rank
  indices = [
    order[(rank_start + offset) % len(order)] for offset in range(shard_size_per_rank)
  ]
  return tuple(variants[index] for index in indices)


def make_dfc_variant_spec(variant: DfcVariant) -> mujoco.MjSpec:
  """Build one free-body DFC variant at its runtime cfg scale."""
  spec = mujoco.MjSpec()
  spec.compiler.inertiagrouprange[:] = (0, 0)

  visual_mesh = spec.add_mesh(name="visual")
  visual_mesh.file = str(variant.visual_mesh)
  visual_mesh.scale[:] = (variant.scale,) * 3
  for index, path in enumerate(variant.collision_meshes):
    mesh = spec.add_mesh(name=f"collision_{index:03d}")
    mesh.file = str(path)
    mesh.scale[:] = (variant.scale,) * 3

  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_freejoint")
  body.add_geom(
    name="visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="visual",
    group=2,
    contype=0,
    conaffinity=0,
    rgba=(0.75, 0.75, 0.75, 1.0),
  )
  for index in range(len(variant.collision_meshes)):
    body.add_geom(
      name=f"collision_{index:03d}",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"collision_{index:03d}",
      group=0,
      density=_DENSITY,
      rgba=(0.75, 0.75, 0.75, 0.0),
      friction=(1.0, 0.002, 0.0001),
      condim=4,
      solref=(0.01, 1.0),
      solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
      contype=2_097_152,
      conaffinity=2_097_151,
    )
  body.add_site(name="object_center", pos=(0.0, 0.0, 0.0))
  return spec


def make_dfc_object_cfg(
  variants: tuple[DfcVariant, ...],
  init_pos: tuple[float, float, float],
) -> VariantEntityCfg:
  """Create a uniformly assigned DFC ``VariantEntityCfg``."""
  spec_fns: dict[str, Callable[[], mujoco.MjSpec]] = {
    variant.name: partial(make_dfc_variant_spec, variant) for variant in variants
  }
  return VariantEntityCfg(
    variants=spec_fns,
    init_state=VariantEntityCfg.InitialStateCfg(
      pos=init_pos,
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={},
      joint_vel={},
    ),
  )


def dfc_point_cloud_params(variants: tuple[DfcVariant, ...]) -> dict[str, Any]:
  """Return serializable observation parameters aligned to variant order."""
  return {
    "variant_point_cloud_paths": tuple(
      str(variant.point_cloud) for variant in variants
    ),
    "variant_point_cloud_scales": tuple(variant.scale for variant in variants),
  }


def make_dfc_eval_env_cfg(
  training_cfg: ManagerBasedRlEnvCfg,
  dataset_dir: str | Path,
  split: str,
  success_threshold: float,
  *,
  actor_only: bool = True,
  retain_debug_visualizers: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a deterministic-policy evaluation environment for one DFC split."""
  cfg = deepcopy(training_cfg)
  apply_primitive_stage_randomization(cfg, PRIMITIVE_DATASET_STAGE - 1)
  variants = load_dfc_variants(dataset_dir, split)
  original_object_cfg = cfg.scene.entities["object"]
  cfg.scene.entities["object"] = make_dfc_object_cfg(
    variants,
    init_pos=original_object_cfg.init_state.pos,
  )
  cfg.scene.num_envs = len(variants)
  cfg.seed = None
  cfg.auto_reset = True
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 1024)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 4096)
  cfg.curriculum = {}
  point_cloud_debug = cfg.rewards.get("object_point_cloud_debug")
  cfg.rewards = {}
  if retain_debug_visualizers and point_cloud_debug is not None:
    cfg.rewards["object_point_cloud_debug"] = point_cloud_debug
  original_reset_cfg = cfg.events["reset_object_pose"]
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=parahand_mdp.reset_dropped_mesh_object_pose,
    mode="reset",
    params={
      "object_name": "object",
      "position_center": original_reset_cfg.params["position_center"],
      "position_noise": original_reset_cfg.params["position_noise"],
      "drop_height_range": (0.10, 0.15),
      "clearance": 0.003,
      "table_height_event_name": original_reset_cfg.params["table_height_event_name"],
      **dfc_point_cloud_params(variants),
    },
  )

  point_cloud_params = dfc_point_cloud_params(variants)
  for group_cfg in cfg.observations.values():
    group_cfg.terms["object_point_cloud_b"].params.update(point_cloud_params)
  cfg.observations["actor"].enable_corruption = True
  if actor_only:
    cfg.observations = {"actor": cfg.observations["actor"]}
  cfg.metrics = {
    "unseen_success": MetricsTermCfg(
      func=parahand_mdp.final_position_success,
      params={
        "command_name": "object_pose",
        "object_cfg": SceneEntityCfg("object"),
        "threshold": success_threshold,
      },
      reduce="last",
    )
  }
  cfg.commands["object_pose"].debug_vis = False
  return cfg


def make_dataset_train_env_cfg(
  training_cfg: ManagerBasedRlEnvCfg,
  variants: tuple[DfcVariant, ...],
  *,
  drop_height_range: tuple[float, float],
  position_noise: tuple[float, float],
  clearance: float,
) -> ManagerBasedRlEnvCfg:
  """Create a Stage 7 environment for one fixed mesh shard."""
  if drop_height_range[0] < 0.0 or drop_height_range[1] < drop_height_range[0]:
    raise ValueError("drop_height_range must be non-negative and ordered.")
  if clearance < 0.0:
    raise ValueError("clearance must be non-negative.")

  cfg = deepcopy(training_cfg)
  apply_primitive_stage_randomization(cfg, PRIMITIVE_DATASET_STAGE - 1)
  original_object_cfg = cfg.scene.entities["object"]
  cfg.scene.entities["object"] = make_dfc_object_cfg(
    variants,
    init_pos=original_object_cfg.init_state.pos,
  )
  cfg.curriculum = {}
  # Stage 7 keeps thousands of worlds resident, so use a smaller per-world
  # contact budget than the 78-world evaluator while retaining headroom for
  # decomposed collision parts touching the floor and hand simultaneously.
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 2048)

  original_reset_cfg = cfg.events["reset_object_pose"]
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=parahand_mdp.reset_dropped_mesh_object_pose,
    mode="reset",
    params={
      "object_name": "object",
      "position_center": original_reset_cfg.params["position_center"],
      "position_noise": position_noise,
      "drop_height_range": drop_height_range,
      "clearance": clearance,
      "table_height_event_name": original_reset_cfg.params["table_height_event_name"],
      **dfc_point_cloud_params(variants),
    },
  )
  for group_cfg in cfg.observations.values():
    group_cfg.terms["object_point_cloud_b"].params.update(
      dfc_point_cloud_params(variants)
    )
  return cfg
