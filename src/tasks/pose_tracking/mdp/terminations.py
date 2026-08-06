from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.envs import mdp as mjlab_mdp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Debug flag - set MJLAB_DEBUG_TERMINATIONS=1 to enable debug prints
_DEBUG = os.getenv("MJLAB_DEBUG_TERMINATIONS", "0").lower() in ("1", "true", "yes")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    result = (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  else:
    assert data.found is not None
    result = torch.any(data.found, dim=-1)
  
  # Debug print
  if _DEBUG and result.any():
    print(f"[TERMINATION] illegal_contact triggered for {result.sum().item()} envs")
  
  return result


def goal_reached(
  env: ManagerBasedRlEnv,
  command_name: str,
  position_tolerance: float,
  heading_tolerance: float,
  max_base_velocity: float = 0.5,
) -> torch.Tensor:
  """Terminate episode successfully when goal pose is reached.
  
  Args:
    env: The environment instance.
    command_name: Name of the pose command to track.
    position_tolerance: Maximum position error (meters) to consider goal reached.
    heading_tolerance: Maximum heading error (radians) to consider goal reached.
    max_base_velocity: Maximum base linear velocity (m/s) to consider goal reached.
  
  Returns:
    Boolean tensor indicating which environments have reached the goal.
  """
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  pose_error = command.error  # [B, 3] - (x, y, heading)
  
  # Check if position is within tolerance
  position_error = torch.norm(pose_error[:, :2], dim=1)
  position_ok = position_error < position_tolerance
  
  # Check if heading is within tolerance
  heading_error = torch.abs(pose_error[:, 2])
  heading_ok = heading_error < heading_tolerance
  
  # Check if base velocity is below threshold (robot has settled)
  robot = env.scene[command.cfg.entity_name]
  base_lin_vel = robot.data.root_link_lin_vel_w  # [B, 3]
  velocity_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only xy velocity
  velocity_ok = velocity_magnitude < max_base_velocity
  
  # All conditions must be satisfied
  result = position_ok & heading_ok & velocity_ok
  
  # Debug print
  if _DEBUG and result.any():
    for i in result.nonzero(as_tuple=False).squeeze(-1):
      print(f"[TERMINATION] goal_reached triggered for env {i.item()}: "
            f"pos_err={position_error[i].item():.3f}m, "
            f"heading_err={heading_error[i].item():.3f}rad, "
            f"vel={velocity_magnitude[i].item():.3f}m/s")
  
  return result


# Wrapper functions for mjlab terminations with debug prints

def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Wrapper for mjlab time_out with debug print."""
  result = mjlab_mdp.time_out(env)
  if _DEBUG and result.any():
    print(f"[TERMINATION] time_out triggered for {result.sum().item()} envs")
  return result


def bad_orientation(env: ManagerBasedRlEnv, limit_angle: float) -> torch.Tensor:
  """Wrapper for mjlab bad_orientation with debug print."""
  result = mjlab_mdp.bad_orientation(env, limit_angle)
  if _DEBUG and result.any():
    print(f"[TERMINATION] bad_orientation triggered for {result.sum().item()} envs")
  return result


def root_height_below_minimum(env: ManagerBasedRlEnv, minimum_height: float) -> torch.Tensor:
  """Wrapper for mjlab root_height_below_minimum with debug print."""
  result = mjlab_mdp.root_height_below_minimum(env, minimum_height)
  if _DEBUG and result.any():
    print(f"[TERMINATION] root_height_below_minimum triggered for {result.sum().item()} envs")
  return result