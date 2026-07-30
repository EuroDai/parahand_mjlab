from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.termination_manager import TerminationTermCfg

from ._table import get_table_heights

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class object_out_of_bounds:
  """Terminate when the object leaves the tabletop-relative workspace."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    object_name = cfg.params["object_name"]
    self._object = env.scene[object_name]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    table_height_event_name: str,
    z_lower_offset: float,
    z_upper_offset: float,
  ) -> torch.Tensor:
    del object_name
    position = self._object_position() - env.scene.env_origins
    table_heights = get_table_heights(env, table_height_event_name)
    outside_xy = (
      (position[:, 0] < x_bounds[0])
      | (position[:, 0] > x_bounds[1])
      | (position[:, 1] < y_bounds[0])
      | (position[:, 1] > y_bounds[1])
    )
    outside_z = (position[:, 2] < table_heights + z_lower_offset) | (
      position[:, 2] > table_heights + z_upper_offset
    )
    return outside_xy | outside_z

  def _object_position(self) -> torch.Tensor:
    q_adr = self._object.data.indexing.free_joint_q_adr
    return self._object.data.data.qpos[:, q_adr[:3]]


def abnormal_robot(
  env: ManagerBasedRlEnv,
  max_abs_qvel: float = 500.0,
) -> torch.Tensor:
  """Terminate on non-finite simulation state or excessive velocity."""
  qpos = env.sim.data.qpos
  qvel = env.sim.data.qvel
  non_finite = ~torch.isfinite(qpos).all(dim=-1) | ~torch.isfinite(qvel).all(dim=-1)
  qvel_too_high = (torch.abs(qvel) > max_abs_qvel).any(dim=-1)
  return non_finite | qvel_too_high
