from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class ObjectPoseCurriculumStage(TypedDict):
  step: int
  position_noise: tuple[float, float]
  yaw_range: tuple[float, float]


class object_pose_curriculum:
  """Apply step-based object reset-pose ranges."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._event_cfg = env.event_manager.get_term_cfg(cfg.params["event_name"])
    self._stages: list[ObjectPoseCurriculumStage] = cfg.params["stages"]
    if not self._stages or self._stages[0]["step"] != 0:
      raise ValueError("Object curriculum stages must start at step 0.")
    if any(
      current["step"] >= following["step"]
      for current, following in zip(self._stages, self._stages[1:], strict=False)
    ):
      raise ValueError("Object curriculum stage steps must be strictly increasing.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    event_name: str,
    stages: list[ObjectPoseCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    stage_index = max(
      index
      for index, stage in enumerate(self._stages)
      if env.common_step_counter >= stage["step"]
    )
    stage = self._stages[stage_index]
    self._event_cfg.params["position_noise"] = stage["position_noise"]
    self._event_cfg.params["yaw_range"] = stage["yaw_range"]
    return {
      "stage": torch.tensor(stage_index, device=env.device, dtype=torch.float32),
      "position_noise_x": torch.tensor(stage["position_noise"][0], device=env.device),
      "position_noise_y": torch.tensor(stage["position_noise"][1], device=env.device),
      "yaw_span": torch.tensor(
        stage["yaw_range"][1] - stage["yaw_range"][0], device=env.device
      ),
    }
