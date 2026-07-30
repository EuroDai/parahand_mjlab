import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

_TOOL_PATH = Path(__file__).parents[1] / "scripts" / "tools" / "prepare_dfc_objects.py"
_SPEC = importlib.util.spec_from_file_location("prepare_dfc_objects", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


def test_cfg_parser_preserves_repeated_objects_and_concatenated_entries(
  tmp_path: Path,
):
  cfg = tmp_path / "train_set.yaml"
  cfg.write_text(
    "  'core/bottle-a':[0.06],\n  'core/bottle-a':[0.10],  'sem/Mug-b':[0.08],\n"
  )

  entries = _TOOL.parse_cfg(cfg)

  assert entries == [
    _TOOL.CfgEntry("core/bottle-a", 0.06),
    _TOOL.CfgEntry("core/bottle-a", 0.10),
    _TOOL.CfgEntry("sem/Mug-b", 0.08),
  ]


def test_cfg_parser_rejects_unparsed_content(tmp_path: Path):
  cfg = tmp_path / "broken.yaml"
  cfg.write_text("'core/bottle-a':[0.06], broken")

  with pytest.raises(ValueError, match="Invalid DFC cfg syntax"):
    _TOOL.parse_cfg(cfg)


def test_cfg_parser_deduplicates_exact_pairs_with_warning(tmp_path: Path):
  cfg = tmp_path / "test_set.yaml"
  cfg.write_text("'sem/Cat-a':[0.12], 'sem/Cat-a':[0.12], 'sem/Cat-a':[0.08],")

  with pytest.warns(UserWarning, match="Removed 1 exact duplicate"):
    entries = _TOOL.parse_cfg(cfg)

  assert entries == [
    _TOOL.CfgEntry("sem/Cat-a", 0.12),
    _TOOL.CfgEntry("sem/Cat-a", 0.08),
  ]


def test_prepare_dfc_object_pool_keeps_unit_geometry_and_cfg_scales(
  tmp_path: Path,
):
  dataset_dir = tmp_path / "DFCData"
  cfg_dir = dataset_dir / "cfg"
  cfg_dir.mkdir(parents=True)
  (cfg_dir / "train_set.yaml").write_text(
    "'sem/TestObject-abc':[0.06], 'sem/TestObject-abc':[0.10],"
  )
  (cfg_dir / "test_set_seen_cat.yaml").write_text("")
  (cfg_dir / "test_set_unseen_cat.yaml").write_text("")

  source_dir = dataset_dir / "raw" / "meshdatav3" / "sem" / "TestObject-abc"
  coacd_dir = source_dir / "coacd"
  coacd_dir.mkdir(parents=True)
  mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
  mesh.export(coacd_dir / "decomposed.obj")
  mesh.export(coacd_dir / "coacd_convex_piece_0.obj")
  rng = np.random.default_rng(42)
  source_points = rng.normal(size=(3000, 3))
  source_points /= np.linalg.norm(source_points, axis=1, keepdims=True)
  np.save(source_dir / "pc.npy", source_points.astype(np.float32))

  output_dir = dataset_dir / "processed" / "test"
  records, failures, splits = _TOOL.process_dataset(
    _TOOL.Args(dataset_dir=dataset_dir, output_dir=output_dir)
  )

  assert failures == {}
  assert len(records) == 1
  assert len(splits["train_set"]) == 2
  record = records[0]
  assert record.object_code == "sem/TestObject-abc"
  assert record.safe_id == "sem__TestObject-abc"
  assert record.scales_by_split == {"train_set": [0.06, 0.10]}
  assert record.collision_parts == 1
  assert record.point_count == 1024

  object_dir = output_dir / "objects" / record.safe_id
  points = np.load(object_dir / "surface_unit_1024.npy")
  assert points.shape == (1024, 3)
  assert np.isclose(np.linalg.norm(points, axis=1), 1.0).all()
  processed_mesh = trimesh.load_mesh(object_dir / "mesh.obj", force="mesh")
  assert np.allclose(processed_mesh.extents, mesh.extents)

  manifest = json.loads((output_dir / "manifest.json").read_text())
  assert manifest["scale_mode"] == "unit_sphere_runtime_cfg"
  assert [entry["scale"] for entry in manifest["splits"]["train_set"]] == [
    0.06,
    0.10,
  ]
