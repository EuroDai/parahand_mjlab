import importlib.util
import sys
from pathlib import Path

import torch

_TOOL_PATH = Path(__file__).parents[1] / "scripts" / "tools" / "parahand_rollouts.py"
_SPEC = importlib.util.spec_from_file_location("parahand_rollouts", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)


def test_rollout_manifest_round_trip_and_termination_counts(tmp_path: Path):
  records = [
    _TOOL.EpisodeRecord(
      episode_id=0,
      path="episodes/episode_000000.pt",
      length=123,
      return_value=4.5,
      termination_reasons=["object_out_of_bounds"],
      shape_id=1,
      size=[0.02, 0.03, 0.025],
    ),
    _TOOL.EpisodeRecord(
      episode_id=1,
      path="episodes/episode_000001.pt",
      length=400,
      return_value=8.0,
      termination_reasons=["time_out"],
      shape_id=2,
      size=[0.025, 0.0, 0.0],
    ),
  ]
  manifest = tmp_path / "manifest.jsonl"
  for record in records:
    _TOOL._append_manifest(manifest, record)

  loaded = _TOOL._read_manifest(tmp_path)

  assert loaded == records
  assert _TOOL._termination_counts(loaded) == {
    "object_out_of_bounds": 1,
    "time_out": 1,
  }


def test_old_rollout_env_origin_is_inferred_from_object_and_target():
  qpos = torch.zeros(2, 10)
  qpos[0, 3:5] = torch.tensor([-5.5521, 4.5782])
  episode = {
    "qpos": qpos,
    "target_pos": torch.tensor([-5.4037, 4.5389, 0.4341]),
  }

  origin = _TOOL._episode_env_origin(episode, torch.arange(3, 10))

  torch.testing.assert_close(origin, torch.tensor([-5.5, 4.5, 0.0]))
