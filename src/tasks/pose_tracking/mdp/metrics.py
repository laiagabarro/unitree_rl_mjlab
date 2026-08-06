from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def position_error(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Compute position error from goal in the xy plane (meters)."""
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  # Get pose error [x, y, heading] from the command
  pose_error = command.error  # [B, 3]
  
  # Compute Euclidean distance in xy
  pos_error = torch.norm(pose_error[:, :2], dim=-1)
  return pos_error


def heading_error(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Compute heading error from goal (radians)."""
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  # Get pose error [x, y, heading] from the command
  pose_error = command.error  # [B, 3]
  
  # Return absolute heading error
  heading_error = torch.abs(pose_error[:, 2])
  return heading_error


def pose_total_error(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Compute total pose error (linear + angular) from goal.
  
  Combines position error (Euclidean distance in xy) with heading error
  to give a single scalar representing total deviation from goal pose.
  """
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  # Get pose error [x, y, heading] from the command
  pose_error = command.error  # [B, 3]
  
  # Linear error (position in xy)
  linear_error = torch.norm(pose_error[:, :2], dim=1)
  
  # Angular error (heading)
  angular_error = torch.abs(pose_error[:, 2])
  
  # Total error
  total_error = linear_error + angular_error
  return total_error


def base_lin_vel(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot/imu_lin_vel",
) -> torch.Tensor:
  """Compute magnitude of base linear velocity (m/s)."""
  sensor: BuiltinSensor = env.scene[sensor_name]
  lin_vel = sensor.data  # [B, 3]
  
  # Compute velocity magnitude
  vel_magnitude = torch.norm(lin_vel, dim=-1)
  return vel_magnitude


def base_ang_vel(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot/imu_ang_vel",
) -> torch.Tensor:
  """Compute magnitude of base angular velocity (rad/s)."""
  sensor: BuiltinSensor = env.scene[sensor_name]
  ang_vel = sensor.data  # [B, 3]
  
  # Compute angular velocity magnitude
  ang_vel_magnitude = torch.norm(ang_vel, dim=-1)
  return ang_vel_magnitude


def joint_deviation_from_stand(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Compute sum of squared deviations of joint angles from default standing pose."""
  asset: Entity = env.scene[asset_cfg.name]
  diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  error = torch.sum(torch.square(diff_angle), dim=1)
  return error


def mean_action_acc(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Compute mean action acceleration (rate of change of actions)."""
  # Get current and previous actions
  curr_actions = env.action_manager.action
  prev_actions = env.action_manager.prev_action
  
  # Compute acceleration (difference between current and previous action differences)
  action_diff = curr_actions - prev_actions
  
  # Compute mean absolute acceleration
  action_acc = torch.mean(torch.abs(action_diff), dim=-1)
  
  return action_acc
