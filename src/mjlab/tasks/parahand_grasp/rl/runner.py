"""ParaHand runner with Stage 2 dataset rehearsal and unseen evaluation."""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from rsl_rl.utils import check_nan
from tensordict import TensorDict
from torch.distributed import ReduceOp
from torch.distributed import distributed_c10d as dist

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.managers.metrics_manager import MetricsManager
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.parahand_grasp.dfc_objects import (
  DfcVariant,
  load_training_variants,
  make_dataset_train_env_cfg,
  make_dfc_eval_env_cfg,
  select_training_shard,
)
from mjlab.tasks.parahand_grasp.mdp.consts import PRIMITIVE_DATASET_STAGE

_UNSEEN_METRIC_NAME = "eval/unseen_success_rate"
_UNSEEN_STEP_METRIC = "unseen_success"


@dataclass
class ParaHandOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """ParaHand runner options."""

  unseen_eval: bool = True
  """Evaluate DFCData unseen variants periodically during training."""
  unseen_eval_interval: int = 300
  """Number of PPO updates between unseen evaluations."""
  unseen_eval_start_stage: int = 2
  """First curriculum stage that records unseen-object success."""
  unseen_eval_dataset_dir: str = "datasets/grasp_objects/DFCData"
  """DFCData root containing ``processed/v1/manifest.json``."""
  unseen_eval_split: str = "test_set_unseen_cat"
  """Manifest split used exclusively for evaluation."""
  unseen_eval_seed: int = 12345
  """Fixed reset seed used for comparable evaluation episodes."""
  unseen_eval_success_threshold: float = 0.05
  """Final object-to-target distance threshold in meters."""
  stage2_enabled: bool = True
  """Enable the runner-managed mesh-object lesson after curriculum promotion."""
  stage2_start_immediately: bool = False
  """Start Stage 2 on the first PPO update, primarily for ablations."""
  stage2_dataset: str = "dfc"
  """Training catalog: ``dfc`` or ``robustdex``."""
  stage2_dfc_dataset_dir: str = "datasets/grasp_objects/DFCData"
  """DFCData root containing the processed manifest."""
  stage2_dfc_split: str = "train_set"
  """DFCData cfg split used for Stage 2 training."""
  stage2_robustdex_dataset_dir: str = "datasets/grasp_objects/RobustDexGrasp"
  """RobustDexGrasp root containing the processed manifest."""
  stage2_primitive_ratio: float = 0.25
  """Fraction of Stage 2 PPO updates collected from analytic Stage 1 primitives."""
  stage2_shard_size_per_rank: int = 128
  """Number of DFC object-scale variants resident on each rank."""
  stage2_shard_update_interval: int = 200
  """Stage 2 PPO updates between DFC shard rotations."""
  stage2_shard_seed: int = 42
  """Deterministic seed for rank-disjoint DFC sharding."""
  stage2_position_noise: tuple[float, float] = (0.08, 0.08)
  """Independent XY reset half-ranges for dataset objects."""
  stage2_drop_height_range: tuple[float, float] = (0.10, 0.15)
  """Vertical gap from the tabletop to the dropped object's lowest point."""
  stage2_floor_clearance: float = 0.003
  """Extra clearance above the floor before applying drop height."""


@contextmanager
def _fixed_eval_rng(seed: int) -> Iterator[None]:
  """Use a fixed eval RNG sequence without consuming training RNG state."""
  python_state = random.getstate()
  numpy_state = np.random.get_state()
  torch_state = torch.random.get_rng_state()
  cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  try:
    yield
  finally:
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
      torch.cuda.set_rng_state_all(cuda_states)


class ParaHandUnseenEvaluator:
  """Persistent one-episode-per-variant evaluator."""

  def __init__(
    self,
    training_env: RslRlVecEnvWrapper,
    dataset_dir: str,
    split: str,
    seed: int,
    success_threshold: float,
    clip_actions: float | None,
  ) -> None:
    self._seed = seed
    eval_cfg = make_dfc_eval_env_cfg(
      training_env.unwrapped.cfg,
      dataset_dir,
      split,
      success_threshold,
    )
    with _fixed_eval_rng(seed):
      raw_env = ManagerBasedRlEnv(cfg=eval_cfg, device=str(training_env.device))
      self.env = RslRlVecEnvWrapper(raw_env, clip_actions=clip_actions)

  def evaluate(self, runner: ManipulationOnPolicyRunner) -> float:
    """Evaluate the current deterministic actor once on every unseen variant."""
    raw_env = self.env.unwrapped
    done_once = torch.zeros(
      raw_env.num_envs,
      dtype=torch.bool,
      device=raw_env.device,
    )
    successes = torch.zeros_like(done_once)
    runner.alg.eval_mode()
    try:
      with _fixed_eval_rng(self._seed), torch.inference_mode():
        observations, _ = self.env.reset()
        policy = runner.alg.get_policy()
        for _ in range(raw_env.max_episode_length):
          actions = policy(observations)
          observations, _, dones, _ = self.env.step(actions)
          done = dones.bool()
          newly_done = done & ~done_once
          if newly_done.any():
            metrics_manager = raw_env.metrics_manager
            if not isinstance(metrics_manager, MetricsManager):
              raise TypeError("Unseen evaluation requires an active MetricsManager.")
            step_success = metrics_manager.get_step_values(_UNSEEN_STEP_METRIC)
            successes[newly_done] = step_success[newly_done].bool()
            done_once |= newly_done
          if done_once.all():
            break
        if not done_once.all():
          incomplete = int((~done_once).sum().item())
          raise RuntimeError(
            f"{incomplete} unseen evaluation environments did not finish within "
            f"{raw_env.max_episode_length} steps."
          )
        return float(successes.float().mean().item())
    finally:
      runner.alg.train_mode()

  def close(self) -> None:
    self.env.close()


class ParaHandOnPolicyRunner(ManipulationOnPolicyRunner):
  """Runner that alternates Stage 2 dataset and primitive PPO updates."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    eval_enabled = bool(train_cfg.get("unseen_eval", False))
    eval_interval = int(train_cfg.get("unseen_eval_interval", 300))
    if eval_interval < 1:
      raise ValueError("unseen_eval_interval must be positive.")
    eval_start_stage = int(train_cfg.get("unseen_eval_start_stage", 2))
    if not 0 <= eval_start_stage <= PRIMITIVE_DATASET_STAGE:
      raise ValueError(
        f"unseen_eval_start_stage must be between 0 and {PRIMITIVE_DATASET_STAGE}."
      )
    self._validate_stage2_cfg(train_cfg)
    super().__init__(env, train_cfg, log_dir, device)

    self._primitive_env = self.env
    self._dataset_env: RslRlVecEnvWrapper | None = None
    self._stage2_catalog: tuple[DfcVariant, ...] | None = None
    self._stage2_active = bool(train_cfg.get("stage2_start_immediately", False))
    self._stage2_update_count = 0
    self._stage2_shard_index = 0
    self._resume_stage2_active = False
    self._active_domain = "primitive"
    self._domain_observations: dict[str, TensorDict] = {}
    self._logger_domain_state = {
      "primitive": self._new_logger_domain_state(copy_current=True)
    }

    self._unseen_eval_interval = eval_interval
    self._unseen_eval_start_stage = eval_start_stage
    self._unseen_evaluator: ParaHandUnseenEvaluator | None = None
    if eval_enabled and self.gpu_global_rank == 0:
      self._unseen_evaluator = ParaHandUnseenEvaluator(
        self.env,
        dataset_dir=str(train_cfg["unseen_eval_dataset_dir"]),
        split=str(train_cfg["unseen_eval_split"]),
        seed=int(train_cfg["unseen_eval_seed"]),
        success_threshold=float(train_cfg["unseen_eval_success_threshold"]),
        clip_actions=train_cfg.get("clip_actions"),
      )

    self._training_log = self.logger.log
    self.logger.log = self._log_with_unseen_eval  # type: ignore[method-assign]

  @staticmethod
  def _validate_stage2_cfg(train_cfg: dict) -> None:
    dataset = str(train_cfg.get("stage2_dataset", "dfc")).lower()
    if dataset not in ("dfc", "robustdex"):
      raise ValueError("stage2_dataset must be 'dfc' or 'robustdex'.")
    if bool(train_cfg.get("stage2_start_immediately", False)) and not bool(
      train_cfg.get("stage2_enabled", True)
    ):
      raise ValueError("stage2_start_immediately requires stage2_enabled.")
    primitive_ratio = float(train_cfg.get("stage2_primitive_ratio", 0.25))
    if not 0.0 <= primitive_ratio <= 1.0:
      raise ValueError("stage2_primitive_ratio must be between zero and one.")
    if int(train_cfg.get("stage2_shard_size_per_rank", 128)) <= 0:
      raise ValueError("stage2_shard_size_per_rank must be positive.")
    if int(train_cfg.get("stage2_shard_update_interval", 200)) <= 0:
      raise ValueError("stage2_shard_update_interval must be positive.")
    position_noise = train_cfg.get("stage2_position_noise", (0.08, 0.08))
    if len(position_noise) != 2 or any(float(value) < 0.0 for value in position_noise):
      raise ValueError("stage2_position_noise must contain two non-negative values.")
    drop_range = train_cfg.get("stage2_drop_height_range", (0.10, 0.15))
    if (
      len(drop_range) != 2
      or float(drop_range[0]) < 0.0
      or float(drop_range[1]) < float(drop_range[0])
    ):
      raise ValueError(
        "stage2_drop_height_range must contain two ordered non-negative values."
      )
    if float(train_cfg.get("stage2_floor_clearance", 0.003)) < 0.0:
      raise ValueError("stage2_floor_clearance must be non-negative.")

  def _new_logger_domain_state(self, *, copy_current: bool) -> dict[str, torch.Tensor]:
    names = ["cur_reward_sum", "cur_episode_length"]
    if self.cfg["algorithm"]["rnd_cfg"]:
      names.extend(("cur_ereward_sum", "cur_ireward_sum"))
    state = {}
    for name in names:
      current = getattr(self.logger, name)
      state[name] = current if copy_current else torch.zeros_like(current)
    return state

  def _switch_domain(self, domain: str) -> tuple[RslRlVecEnvWrapper, TensorDict]:
    if domain == self._active_domain:
      return self.env, self._domain_observations[domain]
    for name, value in self._logger_domain_state[domain].items():
      setattr(self.logger, name, value)
    self._active_domain = domain
    if domain == "primitive":
      self.env = self._primitive_env
    else:
      if self._dataset_env is None:
        raise RuntimeError("Stage 2 dataset environment has not been created.")
      self.env = self._dataset_env
    return self.env, self._domain_observations[domain]

  def _curriculum_term(self):
    manager = self._primitive_env.unwrapped.curriculum_manager
    if not isinstance(manager, CurriculumManager):
      return None
    if "object_lesson" not in manager.active_terms:
      return None
    return manager.get_term_cfg("object_lesson").func

  def _synchronize_stage2(self) -> None:
    if not bool(self.cfg.get("stage2_enabled", True)):
      return
    curriculum = self._curriculum_term()
    local_stage = (
      PRIMITIVE_DATASET_STAGE
      if self._stage2_active
      else int(getattr(curriculum, "stage", 0))
    )
    stage_tensor = torch.tensor(
      local_stage,
      device=self._primitive_env.device,
      dtype=torch.int32,
    )
    if self.is_distributed:
      dist.all_reduce(stage_tensor, op=ReduceOp.MAX)
    synchronized_stage = int(stage_tensor.item())
    if curriculum is not None and synchronized_stage > int(curriculum.stage):
      curriculum.set_stage(synchronized_stage)
    if synchronized_stage >= PRIMITIVE_DATASET_STAGE and self._dataset_env is None:
      self._stage2_active = True
      self._build_dataset_env()
      if self.is_distributed:
        dist.barrier()

  def _load_stage2_catalog(self) -> tuple[DfcVariant, ...]:
    dataset = str(self.cfg["stage2_dataset"]).lower()
    if dataset == "dfc":
      dataset_dir = str(self.cfg["stage2_dfc_dataset_dir"])
      split = str(self.cfg["stage2_dfc_split"])
    else:
      dataset_dir = str(self.cfg["stage2_robustdex_dataset_dir"])
      split = "all"
    return load_training_variants(dataset, dataset_dir, split)

  def _selected_stage2_variants(self) -> tuple[DfcVariant, ...]:
    if self._stage2_catalog is None:
      self._stage2_catalog = self._load_stage2_catalog()
    dataset = str(self.cfg["stage2_dataset"]).lower()
    if dataset == "robustdex":
      return self._stage2_catalog
    return select_training_shard(
      self._stage2_catalog,
      shard_size_per_rank=int(self.cfg["stage2_shard_size_per_rank"]),
      rank=self.gpu_global_rank,
      world_size=self.gpu_world_size,
      shard_index=self._stage2_shard_index,
      seed=int(self.cfg["stage2_shard_seed"]),
    )

  def _build_dataset_env(self) -> None:
    variants = self._selected_stage2_variants()
    dataset_cfg = make_dataset_train_env_cfg(
      self._primitive_env.unwrapped.cfg,
      variants,
      drop_height_range=tuple(self.cfg["stage2_drop_height_range"]),
      position_noise=tuple(self.cfg["stage2_position_noise"]),
      clearance=float(self.cfg["stage2_floor_clearance"]),
    )
    base_seed = int(self.cfg["stage2_shard_seed"])
    dataset_cfg.seed = (
      base_seed + self.gpu_global_rank + 10_000 * (self._stage2_shard_index + 1)
    )
    raw_env = ManagerBasedRlEnv(
      cfg=dataset_cfg,
      device=str(self._primitive_env.device),
    )
    dataset_env = RslRlVecEnvWrapper(
      raw_env,
      clip_actions=self.cfg.get("clip_actions"),
    )
    dataset_obs = dataset_env.get_observations().to(self.device)
    primitive_obs = self._primitive_env.get_observations().to(self.device)
    if set(dataset_obs.keys()) != set(primitive_obs.keys()):
      dataset_env.close()
      raise ValueError("Stage 1 and Stage 2 observation groups do not match.")
    for key in dataset_obs.keys():
      if dataset_obs[key].shape != primitive_obs[key].shape:
        dataset_env.close()
        raise ValueError(
          f"Stage 2 observation shape for '{key}' is {dataset_obs[key].shape}; "
          f"expected {primitive_obs[key].shape}."
        )
    if dataset_env.num_actions != self._primitive_env.num_actions:
      dataset_env.close()
      raise ValueError("Stage 1 and Stage 2 action dimensions do not match.")

    self._dataset_env = dataset_env
    self._domain_observations["dataset"] = dataset_obs
    self._logger_domain_state["dataset"] = self._new_logger_domain_state(
      copy_current=False
    )
    print(
      f"[INFO] Stage 2 rank {self.gpu_global_rank}: loaded "
      f"{len(variants)} {self.cfg['stage2_dataset']} variants "
      f"(shard {self._stage2_shard_index})."
    )

  def _maybe_rotate_dataset_shard(self) -> None:
    if (
      str(self.cfg["stage2_dataset"]).lower() != "dfc" or self._stage2_update_count == 0
    ):
      return
    interval = int(self.cfg["stage2_shard_update_interval"])
    target_index = self._stage2_update_count // interval
    if target_index <= self._stage2_shard_index:
      return
    if self._dataset_env is not None:
      if self._active_domain == "dataset":
        self._switch_domain("primitive")
      self._dataset_env.close()
      self._dataset_env = None
      self._domain_observations.pop("dataset", None)
      self._logger_domain_state.pop("dataset", None)
    self._stage2_shard_index = target_index
    self._build_dataset_env()
    if self.is_distributed:
      dist.barrier()

  def _stage2_domain(self) -> str:
    ratio = float(self.cfg["stage2_primitive_ratio"])
    previous = math.floor(self._stage2_update_count * ratio)
    current = math.floor((self._stage2_update_count + 1) * ratio)
    return "primitive" if current > previous else "dataset"

  def _synchronize_rollout_error(self, error: Exception | None) -> None:
    """Raise a rollout error on every rank before collectives diverge."""
    if not self.is_distributed:
      if error is not None:
        raise error
      return

    failed = torch.tensor(int(error is not None), dtype=torch.int32, device=self.device)
    dist.all_reduce(failed, op=ReduceOp.MAX)
    if not bool(failed.item()):
      return
    if error is not None:
      raise RuntimeError(
        f"Rollout failed on rank {self.gpu_global_rank}: {error}"
      ) from error
    raise RuntimeError(
      "Rollout failed on another distributed rank; see that rank's log "
      "for the original exception."
    )

  def _rollout_step(
    self,
    active_env: RslRlVecEnvWrapper,
    obs: TensorDict,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    """Run one rollout step and synchronize recoverable failures across ranks."""
    result: tuple[TensorDict, torch.Tensor, torch.Tensor, dict] | None = None
    error: Exception | None = None
    try:
      actions = self.alg.act(obs)
      next_obs, rewards, dones, extras = active_env.step(actions.to(active_env.device))
      if self.cfg.get("check_for_nan", True):
        check_nan(next_obs, rewards, dones)
      result = (next_obs, rewards, dones, extras)
    except Exception as exc:
      error = exc

    self._synchronize_rollout_error(error)
    if result is None:
      raise RuntimeError("Rollout failed without an exception.")
    return result

  def _log_with_unseen_eval(
    self,
    it: int,
    start_it: int,
    total_it: int,
    collect_time: float,
    learn_time: float,
    loss_dict: dict,
    learning_rate: float,
    action_std: torch.Tensor,
    rnd_weight: float | None,
    print_minimal: bool = False,
    width: int = 80,
    pad: int = 40,
  ) -> None:
    evaluator = self._unseen_evaluator
    if (
      evaluator is not None
      and self.logger.writer is not None
      and self._logical_curriculum_stage() >= self._unseen_eval_start_stage
      and it % self._unseen_eval_interval == 0
    ):
      eval_start = time.time()
      success_rate = evaluator.evaluate(self)
      learn_time += time.time() - eval_start
      self.logger.writer.add_scalar(_UNSEEN_METRIC_NAME, success_rate, it)

    self._training_log(
      it=it,
      start_it=start_it,
      total_it=total_it,
      collect_time=collect_time,
      learn_time=learn_time,
      loss_dict=loss_dict,
      learning_rate=learning_rate,
      action_std=action_std,
      rnd_weight=rnd_weight,
      print_minimal=print_minimal,
      width=width,
      pad=pad,
    )

  def _logical_curriculum_stage(self) -> int:
    if self._stage2_active:
      return PRIMITIVE_DATASET_STAGE
    curriculum = self._curriculum_term()
    return int(getattr(curriculum, "stage", 0))

  def save(self, path: str, infos=None) -> None:
    """Save Stage 2 routing state while keeping legacy checkpoints readable."""
    stage2_state = {
      "active": self._stage2_active,
      "update_count": self._stage2_update_count,
      "shard_index": self._stage2_shard_index,
    }
    infos = {**(infos or {}), "parahand_stage2": stage2_state}
    active_env = self.env
    self.env = self._primitive_env
    try:
      super().save(path, infos)
    finally:
      self.env = active_env

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Restore optional Stage 2 state from a backward-compatible checkpoint."""
    infos = super().load(path, load_cfg, strict, map_location)
    stage2_state = infos.get("parahand_stage2", {}) if infos else {}
    self._resume_stage2_active = bool(stage2_state.get("active", False))
    self._stage2_active |= self._resume_stage2_active
    self._stage2_update_count = int(stage2_state.get("update_count", 0))
    self._stage2_shard_index = int(stage2_state.get("shard_index", 0))
    return infos

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    """Run PPO, switching complete rollouts between the two Stage 2 domains."""
    if init_at_random_ep_len:
      self._primitive_env.episode_length_buf = torch.randint_like(
        self._primitive_env.episode_length_buf,
        high=int(self._primitive_env.max_episode_length),
      )

    self._domain_observations["primitive"] = self._primitive_env.get_observations().to(
      self.device
    )
    self.alg.train_mode()
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    try:
      for it in range(start_it, total_it):
        self._synchronize_stage2()
        if self._stage2_active:
          self._maybe_rotate_dataset_shard()
          domain = self._stage2_domain()
        else:
          domain = "primitive"
        active_env, obs = self._switch_domain(domain)

        start = time.time()
        with torch.inference_mode():
          for _ in range(self.cfg["num_steps_per_env"]):
            obs, rewards, dones, extras = self._rollout_step(active_env, obs)
            obs, rewards, dones = (
              obs.to(self.device),
              rewards.to(self.device),
              dones.to(self.device),
            )
            self.alg.process_env_step(obs, rewards, dones, extras)
            intrinsic_rewards = (
              self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
            )
            self.logger.process_env_step(
              rewards,
              dones,
              extras,
              intrinsic_rewards,
            )
          self._domain_observations[domain] = obs
          collect_time = time.time() - start
          learn_start = time.time()
          self.alg.compute_returns(obs)

        loss_dict = self.alg.update()
        learn_time = time.time() - learn_start
        self.current_learning_iteration = it
        if self._stage2_active:
          self._stage2_update_count += 1

        self.logger.log(
          it=it,
          start_it=start_it,
          total_it=total_it,
          collect_time=collect_time,
          learn_time=learn_time,
          loss_dict=loss_dict,
          learning_rate=self.alg.learning_rate,
          action_std=self.alg.get_policy().output_std,
          rnd_weight=self.alg.rnd.weight if self.alg.rnd is not None else None,
        )
        if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
          assert self.logger.log_dir is not None
          self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

      if self.logger.writer is not None:
        assert self.logger.log_dir is not None
        self.save(
          os.path.join(
            self.logger.log_dir,
            f"model_{self.current_learning_iteration}.pt",
          )
        )
        self.logger.stop_logging_writer()
    finally:
      if self._dataset_env is not None:
        self._dataset_env.close()
        self._dataset_env = None
      if self._unseen_evaluator is not None:
        self._unseen_evaluator.close()
