"""Event functions for pose tracking tasks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Debug flag - set MJLAB_DEBUG_COMMANDS=1 to enable debug prints
_DEBUG = os.getenv("MJLAB_DEBUG_COMMANDS", "0").lower() in ("1", "true", "yes")


def resample_commands(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
) -> None:
  """Resample commands for reset environments.
  
  This event should be triggered after robot reset events to ensure
  commands are sampled relative to the new robot position.
  
  IMPORTANT: This function forces a robot data update before resampling
  to ensure the latest robot position is used for world-frame transformation.
  
  Args:
    env: The environment instance.
    env_ids: The environment indices to resample commands for.
    command_name: Name of the command to resample.
  """
  command_term = env.command_manager.get_term(command_name)
  
  # Force robot data update to get the latest position after reset events
  # This ensures we sample goals relative to the NEW robot position, not the old one
  robot = env.scene[command_term.cfg.entity_name]
  robot.update(dt=0.0)  # Force data refresh without stepping physics
  
  if _DEBUG and len(env_ids) > 0:
    # Store old targets for comparison
    old_targets = command_term.target_pos_w[env_ids].clone()
    robot_pos_before = robot.data.root_link_pos_w[env_ids].clone()
  
  # Manually trigger command resampling for the reset environments
  command_term._resample_command(env_ids)
  
  # Debug print to verify command is within expected range
  if _DEBUG and len(env_ids) > 0:
    new_targets = command_term.target_pos_w[env_ids]
    robot_pos_after = robot.data.root_link_pos_w[env_ids]
    
    # Calculate distance from robot to new target in xy plane
    delta_xy = new_targets - robot_pos_after[:, :2]
    distances = torch.norm(delta_xy, dim=1)
    
    for i, env_id in enumerate(env_ids):
      print(f"[COMMAND RESAMPLE] env {env_id.item()}:")
      print(f"  Robot pos: [{robot_pos_after[i, 0].item():.3f}, {robot_pos_after[i, 1].item():.3f}]")
      print(f"  Old target: [{old_targets[i, 0].item():.3f}, {old_targets[i, 1].item():.3f}]")
      print(f"  New target: [{new_targets[i, 0].item():.3f}, {new_targets[i, 1].item():.3f}]")
      print(f"  Distance to new target: {distances[i].item():.3f}m")
      print(f"  Expected range: 0.0-0.707m (sqrt of 0.5²+0.5²)")
      if distances[i] > 0.707:  # sqrt(0.5^2 + 0.5^2)
        print(f"  ⚠️  WARNING: Distance exceeds expected range!")
