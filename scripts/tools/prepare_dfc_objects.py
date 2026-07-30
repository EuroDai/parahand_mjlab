"""Prepare DFCData unit-sphere objects into one scale-independent object pool.

The source cfg files remain the authority for train/test splits and discrete scales.
Each unique object is copied once; scales are not baked into meshes or point clouds.

Examples:

  uv run scripts/tools/prepare_dfc_objects.py

  uv run scripts/tools/prepare_dfc_objects.py --limit 20

  uv run scripts/tools/prepare_dfc_objects.py --all-objects
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from tqdm import tqdm

import mjlab

_DEFAULT_DATASET_DIR = Path("datasets/grasp_objects/DFCData")
_ALLOWED_SCALES = frozenset((0.06, 0.08, 0.10, 0.12, 0.15))
_CFG_ENTRY_PATTERN = re.compile(
  r"\s*'(?P<object_code>[^']+)'\s*:\s*"
  r"\[(?P<scale>0(?:\.0?6|\.0?8|\.1(?:0)?|\.12|\.15))\]\s*,?"
)
_COLLISION_INDEX_PATTERN = re.compile(r"coacd_convex_piece_(\d+)\.obj$")


@dataclass(frozen=True)
class Args:
  """DFCData preprocessing options."""

  dataset_dir: Path = _DEFAULT_DATASET_DIR
  output_dir: Path | None = None
  point_count: int = 1024
  all_objects: bool = False
  limit: int | None = None
  file_mode: Literal["copy", "hardlink"] = "copy"
  overwrite: bool = False


@dataclass(frozen=True)
class CfgEntry:
  """One ordered object-scale entry from a DFCData cfg file."""

  object_code: str
  scale: float


@dataclass(frozen=True)
class ObjectRecord:
  """Manifest entry for one scale-independent processed object."""

  object_code: str
  safe_id: str
  directory: str
  unit_mesh: str
  collision_meshes: list[str]
  unit_surface_points: str
  source_directory: str
  source_mesh_sha256: str
  source_points_sha256: str
  source_point_count: int
  point_count: int
  point_radius_max: float
  point_bounds: list[list[float]]
  collision_parts: int
  scales_by_split: dict[str, list[float]]


def _parse_cfg(
  path: Path,
) -> tuple[list[CfgEntry], dict[CfgEntry, int]]:
  """Parse a cfg fragment and return unique entries plus duplicate counts."""
  text = path.read_text(encoding="utf-8")
  raw_entries: list[CfgEntry] = []
  position = 0
  while position < len(text):
    match = _CFG_ENTRY_PATTERN.match(text, position)
    if match is None:
      if text[position:].strip() == "":
        break
      snippet = text[position : position + 100].replace("\n", "\\n")
      raise ValueError(
        f"Invalid DFC cfg syntax in {path} at byte {position}: {snippet!r}"
      )
    scale = float(match.group("scale"))
    if scale not in _ALLOWED_SCALES:
      raise ValueError(f"Unsupported DFC scale {scale} in {path}.")
    raw_entries.append(CfgEntry(match.group("object_code"), scale))
    position = match.end()

  counts = _entry_counts(raw_entries)
  duplicates = {entry: count - 1 for entry, count in counts.items() if count > 1}
  entries = list(dict.fromkeys(raw_entries))
  return entries, duplicates


def parse_cfg(path: Path) -> list[CfgEntry]:
  """Parse a DFC cfg fragment and preserve first-occurrence ordering.

  Unlike a YAML mapping parser, this preserves the same object at multiple
  scales. Exact duplicate object-scale pairs are deduplicated with a warning.
  """
  entries, duplicates = _parse_cfg(path)
  if duplicates:
    duplicate_occurrences = sum(duplicates.values())
    warnings.warn(
      f"Removed {duplicate_occurrences} exact duplicate object-scale entries "
      f"from {path}.",
      stacklevel=2,
    )
  return entries


def _load_cfg_splits(
  cfg_dir: Path,
) -> tuple[dict[str, list[CfgEntry]], dict[str, dict[CfgEntry, int]]]:
  """Load DFC cfg fragments with exact-duplicate diagnostics."""
  cfg_paths = sorted(cfg_dir.glob("*.yaml"))
  if not cfg_paths:
    raise FileNotFoundError(f"No DFC cfg files found in {cfg_dir}")
  splits: dict[str, list[CfgEntry]] = {}
  duplicates_by_split: dict[str, dict[CfgEntry, int]] = {}
  for path in cfg_paths:
    entries, duplicates = _parse_cfg(path)
    splits[path.stem] = entries
    duplicates_by_split[path.stem] = duplicates
    if duplicates:
      warnings.warn(
        f"Removed {sum(duplicates.values())} exact duplicate object-scale "
        f"entries from {path}.",
        stacklevel=2,
      )

  pair_to_split: dict[CfgEntry, str] = {}
  for split_name, entries in splits.items():
    for entry in entries:
      previous_split = pair_to_split.get(entry)
      if previous_split is not None:
        raise ValueError(
          f"Object-scale pair {entry} occurs in both "
          f"'{previous_split}' and '{split_name}'."
        )
      pair_to_split[entry] = split_name
  return splits, duplicates_by_split


def load_cfg_splits(cfg_dir: Path) -> dict[str, list[CfgEntry]]:
  """Load cfg fragments and validate cross-split object-scale uniqueness."""
  splits, _ = _load_cfg_splits(cfg_dir)
  return splits


def _entry_counts(entries: list[CfgEntry]) -> dict[CfgEntry, int]:
  counts: dict[CfgEntry, int] = {}
  for entry in entries:
    counts[entry] = counts.get(entry, 0) + 1
  return counts


def _safe_id(object_code: str) -> str:
  parts = object_code.split("/")
  if len(parts) != 2 or not all(parts):
    raise ValueError(
      f"DFC object code must be '<collection>/<name>', got '{object_code}'."
    )
  safe_id = "__".join(parts)
  if "/" in safe_id or safe_id in (".", ".."):
    raise ValueError(f"Unsafe DFC object code: '{object_code}'.")
  return safe_id


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _collision_sort_key(path: Path) -> int:
  match = _COLLISION_INDEX_PATTERN.fullmatch(path.name)
  if match is None:
    raise ValueError(f"Unexpected DFC collision mesh name: {path}")
  return int(match.group(1))


def _source_paths(
  mesh_root: Path,
  object_code: str,
) -> tuple[Path, Path, list[Path]]:
  source_dir = mesh_root / object_code
  unit_mesh = source_dir / "coacd" / "decomposed.obj"
  points = source_dir / "pc.npy"
  collision_meshes = sorted(
    (source_dir / "coacd").glob("coacd_convex_piece_*.obj"),
    key=_collision_sort_key,
  )
  missing = [str(path) for path in (source_dir, unit_mesh, points) if not path.exists()]
  if missing:
    raise FileNotFoundError(f"Missing DFC source paths: {missing}")
  if not collision_meshes:
    raise FileNotFoundError(f"No CoACD collision meshes found in {source_dir}")
  return unit_mesh, points, collision_meshes


def _copy_file(source: Path, destination: Path, mode: str) -> None:
  if mode == "copy":
    shutil.copy2(source, destination)
  elif mode == "hardlink":
    os.link(source, destination)
  else:
    raise ValueError(f"Unsupported file mode: {mode}")


def _sample_unit_points(
  source_path: Path,
  object_code: str,
  point_count: int,
) -> tuple[np.ndarray, int]:
  source = np.load(source_path, allow_pickle=False)
  if source.ndim != 2 or source.shape[1] != 3:
    raise ValueError(
      f"Expected point cloud [N, 3] in {source_path}, got {source.shape}"
    )
  if source.shape[0] < point_count:
    raise ValueError(
      f"Point cloud {source_path} has {source.shape[0]} points, "
      f"fewer than requested {point_count}."
    )
  if not np.isfinite(source).all():
    raise ValueError(f"Point cloud contains non-finite values: {source_path}")

  seed = int(hashlib.sha256(object_code.encode()).hexdigest()[:16], 16) % (2**32)
  rng = np.random.default_rng(seed)
  indices = rng.choice(source.shape[0], size=point_count, replace=False)
  return np.asarray(source[indices], dtype=np.float32), int(source.shape[0])


def _scales_by_object(
  splits: dict[str, list[CfgEntry]],
) -> dict[str, dict[str, list[float]]]:
  result: dict[str, dict[str, list[float]]] = {}
  for split_name, entries in splits.items():
    for entry in entries:
      by_split = result.setdefault(entry.object_code, {})
      by_split.setdefault(split_name, []).append(entry.scale)
  for by_split in result.values():
    for scales in by_split.values():
      scales.sort()
  return result


def _discover_all_objects(mesh_root: Path) -> list[str]:
  return sorted(
    path.relative_to(mesh_root).as_posix()
    for collection_dir in mesh_root.iterdir()
    if collection_dir.is_dir()
    for path in collection_dir.iterdir()
    if path.is_dir()
  )


def _write_json(path: Path, value: object) -> None:
  path.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _validate_output(
  object_dir: Path,
  expected_point_count: int,
  expected_collision_parts: int,
) -> None:
  metadata_path = object_dir / "metadata.json"
  if not metadata_path.is_file():
    raise FileNotFoundError(f"Missing processed metadata: {metadata_path}")
  points = np.load(object_dir / "surface_unit_1024.npy", allow_pickle=False)
  if points.shape != (expected_point_count, 3):
    raise ValueError(
      f"Expected point shape {(expected_point_count, 3)}, got {points.shape}."
    )
  if points.dtype != np.float32 or not np.isfinite(points).all():
    raise ValueError("Processed DFC points must contain finite float32 values.")
  if not (object_dir / "mesh.obj").is_file():
    raise FileNotFoundError(f"Missing processed unit mesh in {object_dir}")
  collision_files = list((object_dir / "collision").glob("part_*.obj"))
  if len(collision_files) != expected_collision_parts:
    raise ValueError(
      f"Expected {expected_collision_parts} collision parts, "
      f"found {len(collision_files)} in {object_dir}."
    )


def _process_object(
  object_code: str,
  mesh_root: Path,
  output_dir: Path,
  scales_by_split: dict[str, list[float]],
  args: Args,
) -> ObjectRecord:
  source_mesh, source_points, source_collisions = _source_paths(
    mesh_root,
    object_code,
  )
  safe_id = _safe_id(object_code)
  _copy_file(source_mesh, output_dir / "mesh.obj", args.file_mode)
  collision_dir = output_dir / "collision"
  collision_dir.mkdir()
  collision_relpaths: list[str] = []
  for index, source_collision in enumerate(source_collisions):
    relative_path = Path("collision") / f"part_{index:03d}.obj"
    _copy_file(source_collision, output_dir / relative_path, args.file_mode)
    collision_relpaths.append(relative_path.as_posix())

  points, source_point_count = _sample_unit_points(
    source_points,
    object_code,
    args.point_count,
  )
  np.save(output_dir / "surface_unit_1024.npy", points, allow_pickle=False)
  point_radius = np.linalg.norm(points, axis=1)
  source_dir = source_mesh.parents[1]
  record = ObjectRecord(
    object_code=object_code,
    safe_id=safe_id,
    directory=(Path("objects") / safe_id).as_posix(),
    unit_mesh="mesh.obj",
    collision_meshes=collision_relpaths,
    unit_surface_points="surface_unit_1024.npy",
    source_directory=str(source_dir),
    source_mesh_sha256=_sha256(source_mesh),
    source_points_sha256=_sha256(source_points),
    source_point_count=source_point_count,
    point_count=args.point_count,
    point_radius_max=float(point_radius.max()),
    point_bounds=[
      points.min(axis=0).astype(float).tolist(),
      points.max(axis=0).astype(float).tolist(),
    ],
    collision_parts=len(source_collisions),
    scales_by_split=scales_by_split,
  )
  _write_json(output_dir / "metadata.json", asdict(record))
  _validate_output(output_dir, args.point_count, len(source_collisions))
  return record


def _load_record(path: Path) -> ObjectRecord:
  return ObjectRecord(**json.loads(path.read_text(encoding="utf-8")))


def process_dataset(
  args: Args,
) -> tuple[list[ObjectRecord], dict[str, str], dict[str, list[CfgEntry]]]:
  """Prepare DFCData and return object records, failures, and parsed cfg splits."""
  if args.point_count != 1024:
    raise ValueError("The ParaHand observation contract requires exactly 1024 points.")
  if args.limit is not None and args.limit < 1:
    raise ValueError("limit must be positive.")

  dataset_dir = args.dataset_dir.resolve()
  cfg_dir = dataset_dir / "cfg"
  mesh_root = dataset_dir / "raw" / "meshdatav3"
  output_root = (
    args.output_dir.resolve()
    if args.output_dir is not None
    else dataset_dir / "processed" / "v1"
  )
  objects_root = output_root / "objects"
  splits, cfg_duplicates = _load_cfg_splits(cfg_dir)
  object_scales = _scales_by_object(splits)
  object_codes = (
    _discover_all_objects(mesh_root) if args.all_objects else sorted(object_scales)
  )
  if args.limit is not None:
    object_codes = object_codes[: args.limit]

  safe_ids: dict[str, str] = {}
  for object_code in object_codes:
    safe_id = _safe_id(object_code)
    previous = safe_ids.get(safe_id)
    if previous is not None:
      raise ValueError(
        f"Flattened object directory collision: '{previous}' and "
        f"'{object_code}' both map to '{safe_id}'."
      )
    safe_ids[safe_id] = object_code

  objects_root.mkdir(parents=True, exist_ok=True)
  records: list[ObjectRecord] = []
  failures: dict[str, str] = {}
  skipped = 0
  progress = tqdm(object_codes, desc="Preparing DFCData", unit="object")
  for object_code in progress:
    safe_id = _safe_id(object_code)
    final_dir = objects_root / safe_id
    try:
      if final_dir.exists() and not args.overwrite:
        record = _load_record(final_dir / "metadata.json")
        expected_scales = object_scales.get(object_code, {})
        if record.scales_by_split != expected_scales:
          raise ValueError(
            f"Cfg scales changed for {object_code}; rerun with --overwrite."
          )
        _validate_output(
          final_dir,
          args.point_count,
          record.collision_parts,
        )
        records.append(record)
        skipped += 1
      else:
        with tempfile.TemporaryDirectory(
          prefix=f".{safe_id}-",
          dir=objects_root,
        ) as temp_dir_name:
          temp_dir = Path(temp_dir_name)
          record = _process_object(
            object_code,
            mesh_root,
            temp_dir,
            object_scales.get(object_code, {}),
            args,
          )
          if final_dir.exists():
            shutil.rmtree(final_dir)
          temp_dir.replace(final_dir)
        records.append(record)
    except Exception as exc:
      failures[object_code] = f"{type(exc).__name__}: {exc}"
      tqdm.write(f"FAILED {object_code}: {failures[object_code]}")
    progress.set_postfix(
      ok=len(records),
      skipped=skipped,
      failed=len(failures),
      refresh=False,
    )

  manifest = {
    "format_version": 1,
    "source": "DFCData / UniDexGrasp",
    "generator": "scripts/tools/prepare_dfc_objects.py",
    "scale_mode": "unit_sphere_runtime_cfg",
    "allowed_scales": sorted(_ALLOWED_SCALES),
    "settings": {
      key: str(value) if isinstance(value, Path) else value
      for key, value in asdict(args).items()
    },
    "splits": {
      split_name: [asdict(entry) for entry in entries]
      for split_name, entries in splits.items()
    },
    "cfg_diagnostics": {
      "exact_duplicates_removed": {
        split_name: [
          {
            **asdict(entry),
            "duplicate_occurrences": duplicate_count,
          }
          for entry, duplicate_count in duplicates.items()
        ]
        for split_name, duplicates in cfg_duplicates.items()
      }
    },
    "objects": [
      asdict(record) for record in sorted(records, key=lambda item: item.object_code)
    ],
    "failures": failures,
  }
  _write_json(output_root / "manifest.json", manifest)
  return records, failures, splits


def main() -> None:
  args = tyro.cli(Args, config=mjlab.TYRO_FLAGS)
  records, failures, splits = process_dataset(args)
  entry_count = sum(len(entries) for entries in splits.values())
  print(
    f"Prepared {len(records)} unit objects for {entry_count} cfg object-scale "
    f"entries; {len(failures)} failed."
  )
  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
