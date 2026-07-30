import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh

_TOOL_PATH = (
  Path(__file__).parents[1] / "scripts" / "tools" / "prepare_robustdex_objects.py"
)
_SPEC = importlib.util.spec_from_file_location("prepare_robustdex_objects", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


def test_prepare_convex_object_generates_directly_loadable_asset(tmp_path: Path):
  input_dir = tmp_path / "raw"
  source_dir = input_dir / "test_box"
  source_dir.mkdir(parents=True)
  source_mesh = trimesh.creation.box(extents=(0.08, 0.12, 0.20))
  source_mesh.export(source_dir / "top_watertight_tiny.obj")
  output_dir = tmp_path / "processed"

  records, failures = _TOOL.process_dataset(
    _TOOL.Args(
      input_dir=input_dir,
      output_dir=output_dir,
      max_extent=0.11,
      max_grasp_width=0.06,
    )
  )

  assert failures == {}
  assert len(records) == 1
  record = records[0]
  assert record.name == "test_box"
  assert record.collision_parts == 1
  assert max(record.extents) <= 0.11 + 1e-8
  assert sorted(record.extents)[1] <= 0.06 + 1e-8
  assert record.mass > 0.0

  object_dir = output_dir / "test_box"
  points = np.load(object_dir / "surface_1024.npy")
  assert points.shape == (1024, 3)
  assert points.dtype == np.float32
  assert np.isfinite(points).all()

  model = mujoco.MjModel.from_xml_path(str(object_dir / "object.xml"))
  assert model.nq == 7
  assert model.nv == 6
  assert model.ngeom == 2
  assert model.body("object").mass[0] > 0.0

  manifest = json.loads((output_dir / "manifest.json").read_text())
  assert manifest["failures"] == {}
  assert manifest["objects"][0]["name"] == "test_box"
