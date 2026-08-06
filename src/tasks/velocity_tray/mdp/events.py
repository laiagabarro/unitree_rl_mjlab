"""Event functions for tray velocity tasks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Debug flag - set MJLAB_DEBUG_COMMANDS=1 to enable debug prints
_DEBUG = os.getenv("MJLAB_DEBUG_COMMANDS", "0").lower() in ("1", "true", "yes")


def reset_tray_at_hands(
  env,
  env_ids: torch.Tensor | None,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  tray_name: str = "tray",
  site_names: tuple[str, str] = ("left_palm", "right_palm"),
  z_offset: float = 0.0,
) -> None:
  """Reset the tray root pose centered between the robot palms."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  robot = env.scene[robot_cfg.name]
  tray = env.scene[tray_name]

  if isinstance(site_names, str):
    site_names = (site_names,)

  site_ids, resolved_names = robot.find_sites(site_names)
  if len(site_ids) != len(site_names):
    raise ValueError(
      f"Expected sites {site_names} on robot '{robot_cfg.name}', found ids {site_ids}."
    )

  palm_pos = robot.data.site_pos_w[env_ids][:, site_ids, :]
  palm_quat = robot.data.site_quat_w[env_ids][:, site_ids, :]

  mid_pos = palm_pos.mean(dim=1)
  mid_pos[:, 2] += z_offset
  mid_quat = palm_quat[:, 0, :]

  root_pose = torch.cat([mid_pos, mid_quat], dim=-1)
  tray.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
  tray.write_root_link_velocity_to_sim(
    torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
  )
