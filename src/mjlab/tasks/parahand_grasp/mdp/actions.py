from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class _ClippedActionCfg(BaseActionCfg):
  raw_action_limit: float = 1.0


class _ClippedAction(BaseAction):
  """Base action with sanitized policy actions and substep target ramps."""

  raw_action_limit: float
  _ctrl_target: torch.Tensor

  def __init__(self, cfg: _ClippedActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.raw_action_limit = float(cfg.raw_action_limit)
    self._prev_raw_actions = torch.zeros_like(self._raw_actions)
    self._ramp_substeps = env.cfg.decimation
    self._ramp_substep = self._ramp_substeps

  @property
  def prev_raw_action(self) -> torch.Tensor:
    return self._prev_raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._prev_raw_actions.copy_(self._raw_actions)
    safe_actions = torch.nan_to_num(
      actions,
      nan=0.0,
      posinf=self.raw_action_limit,
      neginf=-self.raw_action_limit,
    ).clamp(-self.raw_action_limit, self.raw_action_limit)
    super().process_actions(safe_actions)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    self._prev_raw_actions[env_ids] = 0.0

  def _actuator_ctrl_range(self) -> torch.Tensor:
    actuator_ids, actuator_names = self._entity.find_actuators(
      self.target_names, preserve_order=True
    )
    if actuator_names != self.target_names:
      raise ValueError(
        f"Could not resolve actuator limits for {self.target_names}, "
        f"got {actuator_names}."
      )
    local_actuator_ids = torch.tensor(
      actuator_ids, device=self.device, dtype=torch.long
    )
    global_ctrl_ids = self._entity.indexing.ctrl_ids[local_actuator_ids]
    ctrl_range = torch.as_tensor(
      self._env.sim.mj_model.actuator_ctrlrange,
      device=self.device,
      dtype=torch.float,
    )[global_ctrl_ids]
    return ctrl_range.unsqueeze(0).expand(self.num_envs, -1, -1)

  def _initialize_target_ramp(self) -> None:
    """Initialize ramp state after a subclass creates ``_ctrl_target``."""
    self._ramp_start = self._ctrl_target.clone()
    self._applied_ctrl_target = self._ctrl_target.clone()

  def _set_ctrl_target(self, target: torch.Tensor) -> None:
    """Start a ramp from the last emitted target to a new control target."""
    self._ramp_start.copy_(self._applied_ctrl_target)
    self._ctrl_target.copy_(target)
    self._ramp_substep = 0

  def _ramped_ctrl_target(self) -> torch.Tensor:
    """Return the target for the next physics substep without allocating."""
    self._ramp_substep = min(self._ramp_substep + 1, self._ramp_substeps)
    alpha = self._ramp_substep / self._ramp_substeps
    self._applied_ctrl_target.copy_(self._ramp_start).lerp_(self._ctrl_target, alpha)
    return self._applied_ctrl_target

  def _reset_target_ramp(self, env_ids: torch.Tensor | slice) -> None:
    """Re-anchor reset environments so their next action starts continuously."""
    self._ramp_start[env_ids] = self._ctrl_target[env_ids]
    self._applied_ctrl_target[env_ids] = self._ctrl_target[env_ids]
    self._ramp_substep = self._ramp_substeps


@dataclass(kw_only=True)
class ParaHandRelativeJointPositionActionCfg(_ClippedActionCfg):
  coupled_finger_actuator_names: tuple[str, str, str, str] = (
    "index_mcp_1",
    "middle_mcp_1",
    "ring_mcp_1",
    "little_mcp_1",
  )

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> ParaHandRelativeJointPositionAction:
    return ParaHandRelativeJointPositionAction(self, env)


def apply_mcp_1_constraints(
  target: torch.Tensor,
  limits: torch.Tensor,
  joint_ids: tuple[int, int, int, int],
) -> None:
  """Enforce the ParaHand side-swing ordering within per-joint limits."""
  index_id, middle_id, ring_id, little_id = joint_ids
  middle_lower = torch.maximum(target[:, index_id], limits[:, middle_id, 0])
  middle_upper = torch.minimum(target[:, little_id], limits[:, middle_id, 1])
  target[:, middle_id] = torch.minimum(
    torch.maximum(target[:, middle_id], middle_lower),
    middle_upper,
  )

  ring_lower = torch.maximum(target[:, middle_id], limits[:, ring_id, 0])
  ring_upper = torch.minimum(target[:, little_id], limits[:, ring_id, 1])
  target[:, ring_id] = torch.minimum(
    torch.maximum(target[:, ring_id], ring_lower),
    ring_upper,
  )


class ParaHandRelativeJointPositionAction(_ClippedAction):
  """Persistent joint targets advanced by each policy action.

  Each policy action is added to the previous control target. The resulting
  target is linearly ramped across physics substeps. Resets re-anchor the
  target to the measured joint positions.
  """

  _ctrl_target: torch.Tensor

  def __init__(
    self,
    cfg: ParaHandRelativeJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)
    self._ctrl_range = self._actuator_ctrl_range()
    self._ctrl_target = self._entity.data.joint_pos[:, self.target_ids].clone()
    self._initialize_target_ramp()
    target_index = {name: index for index, name in enumerate(self.target_names)}
    missing_names = set(cfg.coupled_finger_actuator_names) - target_index.keys()
    if missing_names:
      raise ValueError(
        "Coupled finger actuators are missing from the action targets: "
        f"{sorted(missing_names)}."
      )
    (
      self._index_mcp_1_id,
      self._middle_mcp_1_id,
      self._ring_mcp_1_id,
      self._little_mcp_1_id,
    ) = (target_index[name] for name in cfg.coupled_finger_actuator_names)

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    target = self._ctrl_target + self._processed_actions
    target = target.clamp(self._ctrl_range[..., 0], self._ctrl_range[..., 1])
    self._apply_mcp_1_constraints(target)
    self._set_ctrl_target(target)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(
      self._ramped_ctrl_target(), joint_ids=self.target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    current_position = self._entity.data.joint_pos[:, self.target_ids]
    self._ctrl_target[env_ids] = current_position[env_ids]
    self._reset_target_ramp(env_ids)

  def _apply_mcp_1_constraints(self, target: torch.Tensor) -> None:
    apply_mcp_1_constraints(
      target,
      self._ctrl_range,
      (
        self._index_mcp_1_id,
        self._middle_mcp_1_id,
        self._ring_mcp_1_id,
        self._little_mcp_1_id,
      ),
    )


@dataclass(kw_only=True)
class RelativeTendonLengthActionCfg(_ClippedActionCfg):
  reset_target_range: tuple[float, float] = (0.0, 0.0)

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> RelativeTendonLengthAction:
    return RelativeTendonLengthAction(self, env)


class RelativeTendonLengthAction(_ClippedAction):
  """Tendon targets relative to measured lengths, ramped across substeps."""

  _ctrl_target: torch.Tensor

  def __init__(
    self,
    cfg: RelativeTendonLengthActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)
    self._ctrl_range = self._actuator_ctrl_range()
    self._ctrl_target = self._entity.data.tendon_len[:, self.target_ids].clone()
    self._reset_target_center = torch.full_like(self._ctrl_target, torch.nan)
    self._initialize_target_ramp()

  def set_reset_target_center(
    self,
    env_ids: torch.Tensor,
    center: torch.Tensor,
  ) -> None:
    """Set optional absolute reset centers, using NaN to retain relative resets."""
    center = center.to(device=self._ctrl_target.device, dtype=self._ctrl_target.dtype)
    if center.ndim == 1:
      center = center[:, None].expand(-1, self._ctrl_target.shape[1])
    if center.shape != (len(env_ids), self._ctrl_target.shape[1]):
      raise ValueError(
        "Tendon reset center must have shape "
        f"({len(env_ids)}, {self._ctrl_target.shape[1]}), got {tuple(center.shape)}."
      )
    self._reset_target_center[env_ids] = center

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    current_length = self._entity.data.tendon_len[:, self.target_ids]
    target = current_length + self._processed_actions
    target = target.clamp(self._ctrl_range[..., 0], self._ctrl_range[..., 1])
    self._set_ctrl_target(target)

  def apply_actions(self) -> None:
    self._entity.set_tendon_len_target(
      self._ramped_ctrl_target(), tendon_ids=self.target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    current_length = self._entity.data.tendon_len[:, self.target_ids]
    current = current_length[env_ids]
    cfg = cast(RelativeTendonLengthActionCfg, self.cfg)
    offset = torch.empty_like(current).uniform_(*cfg.reset_target_range)
    center = self._reset_target_center[env_ids]
    reset_base = torch.where(torch.isfinite(center), center, current)
    self._ctrl_target[env_ids] = (reset_base + offset).clamp(
      self._ctrl_range[env_ids, :, 0],
      self._ctrl_range[env_ids, :, 1],
    )
    self._reset_target_center[env_ids] = torch.nan
    self._reset_target_ramp(env_ids)
