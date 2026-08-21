from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def reward_threshold_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  trigger_rewards: list[tuple[str, float]],
  target_reward: str,
  target_weight: float,
  ema_alpha: float = 0.05,
  ramp_steps: int = 0,
  state_key: str | None = None,
) -> dict[str, float]:
  """Enable or linearly ramp a reward once all trigger rewards cross thresholds.

  When ``ramp_steps`` is positive, the term's weight goes from zero to
  ``target_weight`` over that many environment control steps after the
  threshold is first crossed. Otherwise it is enabled immediately.

  Persistent EMA/triggered state is stashed on `env` (not on this function,
  since mjlab calls it as a bare function, not an instance), keyed by
  `state_key` (defaults to `target_reward`) so multiple calls of this same
  function for different reward pairs don't clash.

  Note: reads env.reward_manager._episode_sums, a private mjlab attribute
  (pinned to mjlab==1.6.0). If mjlab is upgraded and this breaks, check
  RewardManager.reset() for the new equivalent.
  """
  if not hasattr(env, "_reward_threshold_curriculum_state"):
    env._reward_threshold_curriculum_state = {}
  key = state_key or target_reward
  state = env._reward_threshold_curriculum_state.setdefault(
    key, {"emas": {}, "triggered": False, "trigger_step": None}
  )

  empty = isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0

  result: dict[str, float] = {}
  all_above = True
  for reward_name, threshold in trigger_rewards:
    if empty:
      avg_reward_rate = state["emas"].get(reward_name, 0.0)
    else:
      episode_sums = env.reward_manager._episode_sums[reward_name]
      avg_reward_rate = (
        torch.mean(episode_sums[env_ids]) / env.max_episode_length_s
      ).item()

    prev_ema = state["emas"].get(reward_name)
    new_ema = (
      avg_reward_rate
      if prev_ema is None
      else ema_alpha * avg_reward_rate + (1 - ema_alpha) * prev_ema
    )
    state["emas"][reward_name] = new_ema
    result[f"ema/{reward_name}"] = new_ema
    if new_ema <= threshold:
      all_above = False

  if not state["triggered"] and all_above:
    state["triggered"] = True
    state["trigger_step"] = env.common_step_counter

  if state["triggered"]:
    if ramp_steps > 0:
      elapsed_steps = env.common_step_counter - state["trigger_step"]
      progress = min(max(elapsed_steps / ramp_steps, 0.0), 1.0)
    else:
      progress = 1.0
    env.reward_manager.get_term_cfg(target_reward).weight = target_weight * progress
    result["progress"] = progress
    result["weight"] = target_weight * progress
  else:
    result["progress"] = 0.0
    result["weight"] = 0.0

  result["triggered"] = float(state["triggered"])
  return result
