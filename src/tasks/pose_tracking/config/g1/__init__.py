from mjlab.tasks.registry import register_mjlab_task

from src.tasks.pose_tracking.rl import PoseTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-PoseTracking-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=PoseTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-PoseTracking-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=PoseTrackingOnPolicyRunner,
)
