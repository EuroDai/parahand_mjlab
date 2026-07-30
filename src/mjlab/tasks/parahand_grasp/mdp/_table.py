from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def get_table_heights(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """Return the per-environment tabletop heights owned by a reset event."""
  event_cfg = env.event_manager.get_term_cfg(event_name)
  heights = getattr(event_cfg.func, "heights", None)
  if not isinstance(heights, torch.Tensor):
    raise TypeError(
      f"Event '{event_name}' must expose a tensor-valued 'heights' attribute."
    )
  return heights
