from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class _ClippedActionCfg(BaseActionCfg):
  raw_action_limit: float = 1.0


class _ClippedAction(BaseAction):
  """Base action that stores sanitized, clipped policy actions."""

  raw_action_limit: float

  def __init__(self, cfg: _ClippedActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.raw_action_limit = float(cfg.raw_action_limit)
    self._prev_raw_actions = torch.zeros_like(self._raw_actions)

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


class ParaHandRelativeJointPositionAction(_ClippedAction):
  """Per-policy-step joint targets relative to current angles.

  The relative target is computed once when the policy action is processed.
  Every physics substep then tracks that same target.
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
    current_position = self._entity.data.joint_pos[:, self.target_ids]
    target = current_position + self._processed_actions
    target = target.clamp(self._ctrl_range[..., 0], self._ctrl_range[..., 1])
    self._apply_mcp_1_constraints(target)
    self._ctrl_target.copy_(target)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._ctrl_target, joint_ids=self.target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    current_position = self._entity.data.joint_pos[:, self.target_ids]
    self._ctrl_target[env_ids] = current_position[env_ids]

  def _apply_mcp_1_constraints(self, target: torch.Tensor) -> None:
    middle_lower = torch.maximum(
      target[:, self._index_mcp_1_id],
      self._ctrl_range[:, self._middle_mcp_1_id, 0],
    )
    middle_upper = torch.minimum(
      target[:, self._little_mcp_1_id],
      self._ctrl_range[:, self._middle_mcp_1_id, 1],
    )
    target[:, self._middle_mcp_1_id] = torch.minimum(
      torch.maximum(target[:, self._middle_mcp_1_id], middle_lower),
      middle_upper,
    )

    ring_lower = torch.maximum(
      target[:, self._middle_mcp_1_id],
      self._ctrl_range[:, self._ring_mcp_1_id, 0],
    )
    ring_upper = torch.minimum(
      target[:, self._little_mcp_1_id],
      self._ctrl_range[:, self._ring_mcp_1_id, 1],
    )
    target[:, self._ring_mcp_1_id] = torch.minimum(
      torch.maximum(target[:, self._ring_mcp_1_id], ring_lower),
      ring_upper,
    )


@dataclass(kw_only=True)
class RelativeTendonLengthActionCfg(_ClippedActionCfg):
  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.TENDON

  def build(self, env: ManagerBasedRlEnv) -> RelativeTendonLengthAction:
    return RelativeTendonLengthAction(self, env)


class RelativeTendonLengthAction(_ClippedAction):
  """Per-policy-step tendon targets relative to measured lengths."""

  _ctrl_target: torch.Tensor

  def __init__(
    self,
    cfg: RelativeTendonLengthActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)
    self._ctrl_range = self._actuator_ctrl_range()
    self._ctrl_target = self._entity.data.tendon_len[:, self.target_ids].clone()

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    current_length = self._entity.data.tendon_len[:, self.target_ids]
    target = current_length + self._processed_actions
    target = target.clamp(self._ctrl_range[..., 0], self._ctrl_range[..., 1])
    self._ctrl_target.copy_(target)

  def apply_actions(self) -> None:
    self._entity.set_tendon_len_target(self._ctrl_target, tendon_ids=self.target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    current_length = self._entity.data.tendon_len[:, self.target_ids]
    self._ctrl_target[env_ids] = current_length[env_ids]
