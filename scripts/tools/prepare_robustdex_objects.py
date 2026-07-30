"""Prepare RobustDexGrasp meshes for direct use as MjLab object variants.

The generated asset for each object contains the centered/scaled source mesh,
VHACD convex collision parts, a deterministic 1024-point surface cloud, metadata,
and a standalone MuJoCo XML with one free body.

Run with:

  uv run scripts/tools/prepare_robustdex_objects.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
import trimesh
import tyro
from trimesh.exchange.obj import export_obj

import mjlab

_DEFAULT_INPUT_DIR = Path("datasets/grasp_objects/RobustDexGrasp/raw")
_DEFAULT_OUTPUT_DIR = Path("datasets/grasp_objects/RobustDexGrasp/processed/v1")


@dataclass(frozen=True)
class Args:
  """RobustDexGrasp preprocessing options."""

  input_dir: Path = _DEFAULT_INPUT_DIR
  output_dir: Path = _DEFAULT_OUTPUT_DIR
  density: float = 500.0
  point_count: int = 1024
  max_extent: float = 0.11
  max_grasp_width: float = 0.06
  max_convex_hulls: int = 16
  vhacd_resolution: int = 200_000
  vhacd_volume_error_percent: float = 0.5
  object_names: tuple[str, ...] = ()
  overwrite: bool = False


@dataclass(frozen=True)
class ObjectRecord:
  """Manifest entry for one processed object."""

  name: str
  xml: str
  mesh: str
  collision_meshes: list[str]
  surface_points: str
  source_mesh: str
  source_sha256: str
  source_watertight: bool
  source_convex: bool
  source_extents: list[float]
  source_center_mass: list[float]
  simulation_scale: float
  extents: list[float]
  floor_offset: float
  density: float
  mass: float
  inertia: list[list[float]]
  mass_properties_source: str
  visual_faces: int
  collision_parts: int
  collision_faces: int


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_source_mesh(path: Path) -> trimesh.Trimesh:
  loaded = trimesh.load_mesh(path, force="mesh", process=True, validate=True)
  if not isinstance(loaded, trimesh.Trimesh):
    raise TypeError(f"Expected one triangular mesh in {path}, got {type(loaded)}")
  if loaded.is_empty or len(loaded.faces) == 0:
    raise ValueError(f"Source mesh contains no faces: {path}")
  if not np.isfinite(loaded.vertices).all():
    raise ValueError(f"Source mesh contains non-finite vertices: {path}")
  loaded.remove_unreferenced_vertices()
  loaded.fix_normals()
  return loaded


def _simulation_scale(
  extents: np.ndarray,
  max_extent: float,
  max_grasp_width: float,
) -> float:
  sorted_extents = np.sort(np.asarray(extents, dtype=np.float64))
  if sorted_extents[1] <= 0.0 or sorted_extents[2] <= 0.0:
    raise ValueError(f"Mesh extents must be positive, got {extents.tolist()}")
  return min(
    1.0,
    max_extent / float(sorted_extents[2]),
    max_grasp_width / float(sorted_extents[1]),
  )


def _center_and_scale(
  source: trimesh.Trimesh,
  max_extent: float,
  max_grasp_width: float,
) -> tuple[trimesh.Trimesh, np.ndarray, float, str]:
  mesh = source.copy()
  if source.is_volume and source.volume > 0.0:
    center_mass = np.asarray(source.center_mass, dtype=np.float64)
    mass_properties_source = "source_mesh"
  else:
    center_mass = np.asarray(source.convex_hull.center_mass, dtype=np.float64)
    mass_properties_source = "convex_hull"
  scale = _simulation_scale(source.extents, max_extent, max_grasp_width)
  mesh.apply_translation(-center_mass)
  mesh.apply_scale(scale)
  return mesh, center_mass, scale, mass_properties_source


def _convex_parts(mesh: trimesh.Trimesh, args: Args) -> list[trimesh.Trimesh]:
  if mesh.is_convex:
    return [mesh.convex_hull]
  raw_parts = trimesh.decomposition.convex_decomposition(
    mesh,
    maxConvexHulls=args.max_convex_hulls,
    resolution=args.vhacd_resolution,
    minimumVolumePercentErrorAllowed=args.vhacd_volume_error_percent,
    maxNumVerticesPerCH=64,
    asyncACD=False,
  )
  parts = [
    trimesh.Trimesh(
      vertices=part["vertices"],
      faces=part["faces"],
      process=True,
      validate=True,
    ).convex_hull
    for part in raw_parts
  ]
  parts = [part for part in parts if not part.is_empty and part.volume > 0.0]
  if not parts:
    raise ValueError("VHACD returned no non-degenerate convex parts.")
  return sorted(
    parts,
    key=lambda part: (
      -float(part.volume),
      *np.asarray(part.centroid, dtype=np.float64).tolist(),
    ),
  )


def _surface_points(
  mesh: trimesh.Trimesh,
  count: int,
  source_sha256: str,
) -> np.ndarray:
  seed = int(source_sha256[:16], 16) % (2**32)
  sample_result = cast(Any, trimesh.sample.sample_surface(mesh, count, seed=seed))
  points = sample_result[0]
  return np.asarray(points, dtype=np.float32)


def _export_obj(mesh: trimesh.Trimesh, path: Path) -> None:
  obj = export_obj(
    mesh,
    include_normals=False,
    include_color=False,
    include_texture=False,
    digits=12,
  )
  if not isinstance(obj, str):
    raise TypeError(f"Expected OBJ export text, got {type(obj)}")
  path.write_text(obj, encoding="utf-8")


def _float_string(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
  return " ".join(f"{float(value):.12g}" for value in values)


def _write_xml(
  output_path: Path,
  object_name: str,
  collision_paths: list[Path],
  mass: float,
  inertia: np.ndarray,
  floor_offset: float,
) -> None:
  root = ET.Element("mujoco", model=f"robustdex_{object_name}")
  ET.SubElement(
    root,
    "compiler",
    angle="radian",
    meshdir=".",
    autolimits="true",
  )
  asset = ET.SubElement(root, "asset")
  ET.SubElement(
    asset,
    "mesh",
    name=f"{object_name}_visual_mesh",
    file="mesh.obj",
  )
  for index, collision_path in enumerate(collision_paths):
    ET.SubElement(
      asset,
      "mesh",
      name=f"{object_name}_collision_mesh_{index:03d}",
      file=collision_path.as_posix(),
    )

  worldbody = ET.SubElement(root, "worldbody")
  body = ET.SubElement(worldbody, "body", name="object")
  ET.SubElement(body, "freejoint", name="object_freejoint")
  full_inertia = (
    inertia[0, 0],
    inertia[1, 1],
    inertia[2, 2],
    inertia[0, 1],
    inertia[0, 2],
    inertia[1, 2],
  )
  ET.SubElement(
    body,
    "inertial",
    pos="0 0 0",
    mass=f"{mass:.12g}",
    fullinertia=_float_string(full_inertia),
  )
  ET.SubElement(
    body,
    "geom",
    name="object_visual",
    type="mesh",
    mesh=f"{object_name}_visual_mesh",
    contype="0",
    conaffinity="0",
    group="2",
    rgba="0.65 0.75 0.9 1",
  )
  for index in range(len(collision_paths)):
    ET.SubElement(
      body,
      "geom",
      name=f"object_collision_{index:03d}",
      type="mesh",
      mesh=f"{object_name}_collision_mesh_{index:03d}",
      contype="2097152",
      conaffinity="2097151",
      condim="4",
      group="3",
      rgba="0 0 0 0",
      friction="1.0 0.1 0.002",
      solref="0.02 1.0",
      solimp="0.95 0.99 0.001 0.5 2.0",
    )
  ET.SubElement(body, "site", name="object_center", pos="0 0 0")
  ET.SubElement(body, "site", name="object_floor", pos=f"0 0 {-floor_offset:.12g}")

  tree = ET.ElementTree(root)
  ET.indent(tree, space="  ")
  tree.write(output_path, encoding="unicode", xml_declaration=True)


def _validate_output(
  object_dir: Path,
  point_count: int,
  expected_collision_parts: int,
) -> None:
  points = np.load(object_dir / "surface_1024.npy")
  if points.shape != (point_count, 3):
    raise ValueError(
      f"Expected point cloud shape {(point_count, 3)}, got {points.shape}"
    )
  if points.dtype != np.float32 or not np.isfinite(points).all():
    raise ValueError("Surface point cloud must contain finite float32 values.")

  collision_paths = sorted((object_dir / "collision").glob("part_*.obj"))
  if len(collision_paths) != expected_collision_parts:
    raise ValueError(
      f"Expected {expected_collision_parts} collision files, "
      f"got {len(collision_paths)}."
    )
  for collision_path in collision_paths:
    part = _load_source_mesh(collision_path)
    if not part.is_convex or not part.is_watertight:
      raise ValueError(
        f"Collision part must be convex and watertight: {collision_path}"
      )

  model = mujoco.MjModel.from_xml_path(str(object_dir / "object.xml"))
  if model.nq != 7 or model.nv != 6:
    raise ValueError(
      f"Expected one free joint (nq=7, nv=6), got nq={model.nq}, nv={model.nv}."
    )
  if model.ngeom != expected_collision_parts + 1:
    raise ValueError(
      f"Expected {expected_collision_parts + 1} geoms, got {model.ngeom}."
    )
  body_id = model.body("object").id
  if not np.isfinite(model.body_inertia[body_id]).all():
    raise ValueError("Compiled object inertia contains non-finite values.")
  if model.body_mass[body_id] <= 0.0:
    raise ValueError("Compiled object mass must be positive.")


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _process_object(source_dir: Path, output_dir: Path, args: Args) -> ObjectRecord:
  object_name = source_dir.name
  source_path = source_dir / "top_watertight_tiny.obj"
  if not source_path.is_file():
    raise FileNotFoundError(f"Missing source mesh: {source_path}")

  source_sha256 = _sha256(source_path)
  source = _load_source_mesh(source_path)
  mesh, source_center_mass, scale, mass_properties_source = _center_and_scale(
    source,
    args.max_extent,
    args.max_grasp_width,
  )
  parts = _convex_parts(mesh, args)

  mass_mesh = mesh if mesh.is_volume and mesh.volume > 0.0 else mesh.convex_hull
  mass = float(mass_mesh.volume * args.density)
  inertia = np.asarray(mass_mesh.moment_inertia * args.density, dtype=np.float64)
  if mass <= 0.0 or not np.isfinite(inertia).all():
    raise ValueError(f"Invalid mass properties for {object_name}.")

  output_dir.mkdir(parents=True, exist_ok=True)
  _export_obj(mesh, output_dir / "mesh.obj")
  collision_dir = output_dir / "collision"
  collision_dir.mkdir()
  collision_paths: list[Path] = []
  for index, part in enumerate(parts):
    relative_path = Path("collision") / f"part_{index:03d}.obj"
    _export_obj(part, output_dir / relative_path)
    collision_paths.append(relative_path)

  points = _surface_points(mesh, args.point_count, source_sha256)
  np.save(output_dir / "surface_1024.npy", points, allow_pickle=False)
  floor_offset = float(-mesh.bounds[0, 2])
  _write_xml(
    output_dir / "object.xml",
    object_name,
    collision_paths,
    mass,
    inertia,
    floor_offset,
  )

  relative_object_dir = Path(object_name)
  record = ObjectRecord(
    name=object_name,
    xml=(relative_object_dir / "object.xml").as_posix(),
    mesh=(relative_object_dir / "mesh.obj").as_posix(),
    collision_meshes=[
      (relative_object_dir / path).as_posix() for path in collision_paths
    ],
    surface_points=(relative_object_dir / "surface_1024.npy").as_posix(),
    source_mesh=str(source_path),
    source_sha256=source_sha256,
    source_watertight=bool(source.is_watertight),
    source_convex=bool(source.is_convex),
    source_extents=cast(list[float], source.extents.tolist()),
    source_center_mass=cast(list[float], source_center_mass.tolist()),
    simulation_scale=scale,
    extents=cast(list[float], mesh.extents.tolist()),
    floor_offset=floor_offset,
    density=args.density,
    mass=mass,
    inertia=cast(list[list[float]], inertia.tolist()),
    mass_properties_source=mass_properties_source,
    visual_faces=len(mesh.faces),
    collision_parts=len(parts),
    collision_faces=sum(len(part.faces) for part in parts),
  )
  _write_json(output_dir / "metadata.json", asdict(record))
  _validate_output(output_dir, args.point_count, len(parts))
  return record


def _load_record(path: Path) -> ObjectRecord:
  return ObjectRecord(**json.loads(path.read_text(encoding="utf-8")))


def process_dataset(args: Args) -> tuple[list[ObjectRecord], dict[str, str]]:
  """Process all RobustDexGrasp object directories and return successes/failures."""
  if args.point_count != 1024:
    raise ValueError("The ParaHand observation contract requires exactly 1024 points.")
  if args.density <= 0.0:
    raise ValueError("density must be positive.")
  if args.max_extent <= 0.0 or args.max_grasp_width <= 0.0:
    raise ValueError("Mesh size limits must be positive.")
  if args.max_convex_hulls < 1:
    raise ValueError("max_convex_hulls must be at least one.")

  source_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
  if not source_dirs:
    raise FileNotFoundError(f"No object directories found in {args.input_dir}")
  args.output_dir.mkdir(parents=True, exist_ok=True)

  records: list[ObjectRecord] = []
  failures: dict[str, str] = {}
  selected_names = set(args.object_names)
  manifest_path = args.output_dir / "manifest.json"
  if selected_names:
    available_names = {path.name for path in source_dirs}
    missing_names = sorted(selected_names - available_names)
    if missing_names:
      raise ValueError(f"Unknown object names: {missing_names}")
    source_dirs = [path for path in source_dirs if path.name in selected_names]
    if manifest_path.is_file():
      previous = json.loads(manifest_path.read_text(encoding="utf-8"))
      records.extend(
        ObjectRecord(**record)
        for record in previous.get("objects", [])
        if record["name"] not in selected_names
      )
      failures.update(
        {
          name: message
          for name, message in previous.get("failures", {}).items()
          if name not in selected_names
        }
      )

  processed_count = len(source_dirs)
  for index, source_dir in enumerate(source_dirs, start=1):
    final_dir = args.output_dir / source_dir.name
    print(f"[{index:02d}/{processed_count:02d}] {source_dir.name}", flush=True)
    if final_dir.exists() and not args.overwrite:
      metadata_path = final_dir / "metadata.json"
      if not metadata_path.is_file():
        failures[source_dir.name] = (
          f"Output exists without metadata: {final_dir}; use --overwrite."
        )
        continue
      try:
        record = _load_record(metadata_path)
        _validate_output(final_dir, args.point_count, record.collision_parts)
        records.append(record)
        print("  reused validated output", flush=True)
      except Exception as exc:
        failures[source_dir.name] = f"{type(exc).__name__}: {exc}"
      continue

    try:
      with tempfile.TemporaryDirectory(
        prefix=f".{source_dir.name}-",
        dir=args.output_dir,
      ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        record = _process_object(source_dir, temp_dir, args)
        if final_dir.exists():
          shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
      records.append(record)
      print(
        f"  scale={record.simulation_scale:.4f}, "
        f"parts={record.collision_parts}, mass={record.mass:.4f} kg",
        flush=True,
      )
    except Exception as exc:
      failures[source_dir.name] = f"{type(exc).__name__}: {exc}"
      print(f"  FAILED: {failures[source_dir.name]}", flush=True)

  manifest = {
    "format_version": 1,
    "source": "RobustDexGrasp",
    "generator": "scripts/tools/prepare_robustdex_objects.py",
    "settings": {
      key: str(value) if isinstance(value, Path) else value
      for key, value in asdict(args).items()
    },
    "objects": [
      asdict(record) for record in sorted(records, key=lambda item: item.name)
    ],
    "failures": failures,
  }
  _write_json(manifest_path, manifest)
  return records, failures


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  records, failures = process_dataset(args)
  print(
    f"Prepared {len(records)} objects in {args.output_dir}; {len(failures)} failed."
  )
  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
