from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class object_out_of_bounds:
  """Terminate when the object leaves its reset-relative workspace."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    object_name = cfg.params["object_name"]
    self._object = env.scene[object_name]
    self._initial_position = torch.zeros(env.num_envs, 3, device=env.device)
    self._lower = torch.tensor(
      (
        cfg.params["x_bounds"][0],
        cfg.params["y_bounds"][0],
        cfg.params["z_bounds"][0],
      ),
      device=env.device,
    )
    self._upper = torch.tensor(
      (
        cfg.params["x_bounds"][1],
        cfg.params["y_bounds"][1],
        cfg.params["z_bounds"][1],
      ),
      device=env.device,
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initial_position[env_ids] = self._object_position()[env_ids]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
  ) -> torch.Tensor:
    del env, object_name
    movement = self._object_position() - self._initial_position
    del x_bounds, y_bounds, z_bounds
    return ((movement < self._lower) | (movement > self._upper)).any(dim=-1)

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
