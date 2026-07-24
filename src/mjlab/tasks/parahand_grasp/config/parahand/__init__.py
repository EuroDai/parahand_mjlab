from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  parahand_grasp_object_env_cfg,
  parahand_only_grasp_object_env_cfg,
)
from .rl_cfg import (
  parahand_grasp_object_ppo_runner_cfg,
  parahand_only_grasp_object_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-Object-ParaHand",
  env_cfg=parahand_grasp_object_env_cfg(),
  play_env_cfg=parahand_grasp_object_env_cfg(play=True),
  rl_cfg=parahand_grasp_object_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Grasp-Object-ParaHand-Only",
  env_cfg=parahand_only_grasp_object_env_cfg(),
  play_env_cfg=parahand_only_grasp_object_env_cfg(play=True),
  rl_cfg=parahand_only_grasp_object_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
