"""Event functions for tray velocity tasks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import RecomputeLevel
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
  site_names: tuple[str, ...] = ("right_palm",),
  z_offset: float = 0.08,
) -> None:
  """Reset the tray root pose near the robot palm site."""
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

  site_id = site_ids[0]
  if _DEBUG:
    print(
      f"[reset_tray_at_hands] raw palm_pos before recompute="
      f"{robot.data.site_pos_w[env_ids, site_id, :]}"
    )
  # Ensure joint resets are reflected in world-frame site poses.
  env.sim.recompute_constants(RecomputeLevel.set_const_0)
  if _DEBUG:
    print(
      f"[reset_tray_at_hands] raw palm_pos after recompute="
      f"{robot.data.site_pos_w[env_ids, site_id, :]}"
    )

  palm_pos = robot.data.site_pos_w[env_ids, site_id, :].clone()
  palm_quat = robot.data.site_quat_w[env_ids, site_id, :].clone()

  if _DEBUG:
    print(
      f"[reset_tray_at_hands] env_ids={env_ids.tolist()} "
      f"site_names={resolved_names} site_id={site_id}"
    )
    print(f"[reset_tray_at_hands] palm_pos={palm_pos}")
    print(f"[reset_tray_at_hands] palm_quat={palm_quat}")

  palm_pos[:, 2] += z_offset
  root_pose = torch.cat([palm_pos, palm_quat], dim=-1)

  if _DEBUG:
    print(f"[reset_tray_at_hands] writing root_pose={root_pose}")

  tray.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
  tray.write_root_link_velocity_to_sim(
    torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
  )

  if _DEBUG:
    try:
      actual_pose = tray.data.root_link_pose_w[env_ids]
    except Exception as exc:
      actual_pose = f"<unavailable: {exc}>"
    print(f"[reset_tray_at_hands] actual tray root_link_pose_w={actual_pose}")
