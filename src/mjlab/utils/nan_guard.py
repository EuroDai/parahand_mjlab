"""Lightweight NaN guard for capturing simulation states when NaN/Inf detected."""

import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch


@dataclass
class NanGuardCfg:
  """Configuration for NaN guard."""

  enabled: bool = False
  buffer_size: int = 100
  output_dir: str = "/tmp/mjlab/nan_dumps"
  max_envs_to_dump: int = 5
  dump_cooldown_steps: int = 10_000
  """Minimum physics steps between dumps while non-finite values persist."""
  max_dumps: int = 10
  """Maximum number of dumps per simulation process."""


class NanGuard:
  """Guards against NaN/Inf by buffering states and dumping on detection.

  When enabled, maintains a rolling buffer of simulation states and writes
  them to disk when NaN or Inf is detected. When disabled, all operations
  are no-ops with minimal overhead.
  """

  def __init__(self, cfg: NanGuardCfg, num_envs: int, mj_model: mujoco.MjModel) -> None:
    self.enabled = cfg.enabled
    self.num_envs = num_envs

    if not self.enabled:
      return

    self.buffer_size = cfg.buffer_size
    self.output_dir = Path(cfg.output_dir)
    self.max_envs_to_dump = cfg.max_envs_to_dump
    self.dump_cooldown_steps = cfg.dump_cooldown_steps
    self.max_dumps = cfg.max_dumps
    self.buffer: deque = deque(maxlen=self.buffer_size)
    self.step_counter = 0
    self._dump_count = 0
    self._last_dump_step = -self.dump_cooldown_steps

    self.state_spec = mujoco.mjtState.mjSTATE_PHYSICS.value
    if mj_model.nmocap > 0:
      self.state_spec |= (
        mujoco.mjtState.mjSTATE_MOCAP_POS.value
        | mujoco.mjtState.mjSTATE_MOCAP_QUAT.value
      )
    self.state_size = mujoco.mj_stateSize(mj_model, self.state_spec)
    self.mj_model = mj_model
    self.mj_data = mujoco.MjData(mj_model)

  def capture(self, wp_data: mjwarp.Data) -> None:
    """Capture current simulation state to buffer."""
    if not self.enabled:
      return

    state = {
      "step": self.step_counter,
      "qpos": wp_data.qpos.clone(),
      "qvel": wp_data.qvel.clone(),
    }
    if self.mj_model.na > 0:
      state["act"] = wp_data.act.clone()
    if self.mj_model.nmocap > 0:
      state["mocap_pos"] = wp_data.mocap_pos.clone()
      state["mocap_quat"] = wp_data.mocap_quat.clone()

    self.buffer.append(state)
    self.step_counter += 1

  @contextmanager
  def watch(self, wp_data: mjwarp.Data) -> Iterator[None]:
    """Context manager that captures state before and checks for NaN/Inf after.

    Usage:
      with nan_guard.watch(wp_data):
        mjwarp.step(wp_model, wp_data)
    """
    self.capture(wp_data)
    yield
    self.check_and_dump(wp_data)

  @staticmethod
  def detect_non_finite_fields(data: mjwarp.Data) -> dict[str, torch.Tensor]:
    """Return per-environment non-finite masks for each physics-state field."""
    return {
      name: ~torch.isfinite(getattr(data, name)).all(dim=-1)
      for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")
    }

  @classmethod
  def detect_nans(cls, data: mjwarp.Data) -> torch.Tensor:
    """Detect NaN/Inf values in the complete watched physics state.

    Args:
      data: MuJoCo simulation data containing physics state.

    Returns:
      Boolean tensor where True indicates environments with NaN/Inf values.
    """
    nan_mask = torch.zeros(
      data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device
    )
    for field_mask in cls.detect_non_finite_fields(data).values():
      nan_mask |= field_mask
    return nan_mask

  def check_and_dump(self, data: mjwarp.Data) -> bool:
    """Check for NaN/Inf and dump buffer if detected.

    Returns:
      True if NaN/Inf was detected, even if dump rate limiting suppressed the dump.
    """
    if not self.enabled:
      return False

    field_masks = self.detect_non_finite_fields(data)
    nan_mask = torch.zeros(
      data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device
    )
    for field_mask in field_masks.values():
      nan_mask |= field_mask

    if nan_mask.any():
      cooldown_elapsed = (
        self.step_counter - self._last_dump_step >= self.dump_cooldown_steps
      )
      if self._dump_count < self.max_dumps and cooldown_elapsed:
        nan_env_ids = torch.where(nan_mask)[0].cpu().numpy().tolist()
        non_finite_fields = {
          name: torch.where(mask)[0].cpu().numpy().tolist()
          for name, mask in field_masks.items()
          if mask.any()
        }
        self._dump_buffer(nan_env_ids, non_finite_fields)
        self._dump_count += 1
        self._last_dump_step = self.step_counter
      return True

    return False

  def _dump_buffer(
    self,
    nan_env_ids: list[int],
    non_finite_fields: dict[str, list[int]],
  ) -> None:
    """Write buffered states to disk."""
    self.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"{timestamp}_pid{os.getpid()}_{self._dump_count + 1:03d}"
    filename = self.output_dir / f"nan_dump_{suffix}.npz"
    model_filename = self.output_dir / f"model_{suffix}.mjb"

    envs_to_dump = nan_env_ids[: self.max_envs_to_dump]
    data = {}
    for item in self.buffer:
      step = item["step"]
      qpos = item["qpos"]
      qvel = item["qvel"]
      act = item.get("act", None)
      mocap_pos = item.get("mocap_pos", None)
      mocap_quat = item.get("mocap_quat", None)

      states = np.empty((len(envs_to_dump), self.state_size))
      for idx, env_id in enumerate(envs_to_dump):
        self.mj_data.qpos[:] = qpos[env_id].cpu().numpy()
        self.mj_data.qvel[:] = qvel[env_id].cpu().numpy()
        if act is not None:
          self.mj_data.act[:] = act[env_id].cpu().numpy()
        if mocap_pos is not None:
          self.mj_data.mocap_pos[:] = mocap_pos[env_id].cpu().numpy()
          self.mj_data.mocap_quat[:] = mocap_quat[env_id].cpu().numpy()

        mujoco.mj_getState(self.mj_model, self.mj_data, states[idx], self.state_spec)

      data[f"states_step_{step:06d}"] = states

    data["_metadata"] = np.array(
      {
        "num_envs_total": self.num_envs,
        "num_envs_dumped": len(envs_to_dump),
        "nan_env_ids": nan_env_ids,
        "non_finite_fields": non_finite_fields,
        "dumped_env_ids": list(envs_to_dump),
        "dump_index": self._dump_count + 1,
        "state_spec": self.state_spec,
        "state_size": self.state_size,
        "buffer_size": len(self.buffer),
        "detection_step": self.step_counter,
        "timestamp": timestamp,
        "model_file": model_filename.name,
        "note": "States captured using mj_getState with state_spec. "
        "Use mj_setState with the same spec to restore. "
        "Model saved as MJB for easy reloading.",
      },
      dtype=object,
    )

    np.savez_compressed(filename, **data)
    mujoco.mj_saveModel(self.mj_model, str(model_filename), None)

    # Replace symlinks atomically so multiple ranks cannot leave a broken link.
    latest_dump = self.output_dir / "nan_dump_latest.npz"
    latest_model = self.output_dir / "model_latest.mjb"
    self._replace_symlink(latest_dump, filename.name)
    self._replace_symlink(latest_model, model_filename.name)

    print(
      f"[NanGuard] Detected NaN/Inf at step {self.step_counter}: {non_finite_fields}",
      flush=True,
    )
    print(f"[NanGuard] NaN/Inf found in envs: {nan_env_ids[:10]}...", flush=True)
    print(f"[NanGuard] Dumping {len(envs_to_dump)} envs: {envs_to_dump}", flush=True)
    print(f"[NanGuard] Dumped {len(self.buffer)} states to: {filename}", flush=True)
    print(f"[NanGuard] Saved model to: {model_filename}", flush=True)
    print(f"[NanGuard] Latest dump symlinked at: {latest_dump}", flush=True)

  def _replace_symlink(self, link: Path, target: str) -> None:
    temporary_link = link.with_name(f".{link.name}.{os.getpid()}")
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(target)
    temporary_link.replace(link)
