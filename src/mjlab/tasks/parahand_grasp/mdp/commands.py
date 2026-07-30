from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg

from ._table import get_table_heights

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TableRelativeLiftingCommand(LiftingCommand):
  """Sample lifting targets relative to each environment's tabletop."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cfg = cast(TableRelativeLiftingCommandCfg, self.cfg)
    table_heights = get_table_heights(
      self._env,
      cfg.table_height_event_name,
    )
    self.target_pos[env_ids, 2] += table_heights[env_ids]


@dataclass(kw_only=True)
class TableRelativeLiftingCommandCfg(LiftingCommandCfg):
  """Configuration for tabletop-relative lifting targets."""

  table_height_event_name: str = "reset_table_height"

  def build(self, env: ManagerBasedRlEnv) -> TableRelativeLiftingCommand:
    return TableRelativeLiftingCommand(self, env)
