from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_pose2d(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded 2D position (x, y) in base frame.
  
  Uses the pose error from the Pose2dCommand, which represents the target
  position relative to the robot's current base frame.
  """
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  # Get pose error [x, y, heading] from the command manager
  # The error property should be available on the Pose2dCommand
  pose_error = command.error  # [B, 3]
  
  # Position error in xy
  xy_error = torch.sum(torch.square(pose_error[:, :2]), dim=1)
  
  return torch.exp(-xy_error / std**2)


def track_angular_pose2d(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded heading in base frame.
  
  Uses the heading error from the Pose2dCommand.
  """
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  # Get pose error [x, y, heading] from the command manager
  pose_error = command.error  # [B, 3]
  
  # Heading error
  heading_error = torch.square(pose_error[:, 2])
  
  return torch.exp(-heading_error / std**2)


def goal_reached_bonus(
  env: ManagerBasedRlEnv,
  position_tolerance: float,
  heading_tolerance: float,
  command_name: str,
  max_base_velocity: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sparse bonus reward for reaching and maintaining the goal pose.
  
  Gives a constant reward when position, heading, and optionally velocity are within tolerance.
  This encourages the robot to actually reach and settle at the goal.
  
  Args:
    env: The environment instance.
    position_tolerance: Maximum position error (meters).
    heading_tolerance: Maximum heading error (radians).
    command_name: Name of the pose command.
    max_base_velocity: Optional maximum base velocity (m/s). If None, velocity is not checked.
    asset_cfg: Scene entity configuration.
  """
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  pose_error = command.error  # [B, 3]
  
  # Check if position is within tolerance
  position_error = torch.norm(pose_error[:, :2], dim=1)
  position_ok = position_error < position_tolerance
  
  # Check if heading is within tolerance
  heading_error = torch.abs(pose_error[:, 2])
  heading_ok = heading_error < heading_tolerance
  
  # Check if velocity is below threshold (if specified)
  if max_base_velocity is not None:
    robot = env.scene[command.cfg.entity_name]
    base_lin_vel = robot.data.root_link_lin_vel_w  # [B, 3]
    velocity_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only xy velocity
    velocity_ok = velocity_magnitude < max_base_velocity
    at_goal = (position_ok & heading_ok & velocity_ok).float()
  else:
    at_goal = (position_ok & heading_ok).float()
  
  return at_goal


def velocity_toward_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  distance_threshold: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for moving toward the goal position.
  
  Projects the robot's velocity onto the direction toward goal.
  Positive reward when moving toward goal, negative when moving away.
  This provides immediate feedback for making progress.
  
  Args:
    env: The environment instance.
    command_name: Name of the pose command.
    distance_threshold: Distance threshold (meters) below which the reward is disabled.
      Prevents conflict with goal stabilization rewards.
    asset_cfg: Scene entity configuration.
  """
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get pose error (direction to goal in base frame)
  pose_error = command.error  # [B, 3]
  direction_to_goal = pose_error[:, :2]  # [B, 2] (x, y)
  
  # Get current velocity in base frame
  lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]  # [B, 2] (x, y)
  
  # Normalize direction to goal (avoid division by zero)
  goal_distance = torch.norm(direction_to_goal, dim=1, keepdim=True)
  goal_distance = torch.clamp(goal_distance, min=0.01)
  direction_to_goal_normalized = direction_to_goal / goal_distance
  
  # Compute velocity component toward goal (dot product)
  velocity_toward = torch.sum(lin_vel_b * direction_to_goal_normalized, dim=1)
  
  # Linearly scale reward from 0 at goal to 1 at distance_threshold
  # Prevents conflict with goal stabilization rewards while providing smooth transition
  scale = torch.clamp(goal_distance.squeeze(1) / distance_threshold, min=0.0, max=1.0)

  # Zero out reward when within distance threshold to avoid conflict with goal stabilization rewards.
  # scale = (goal_distance.squeeze(1) > distance_threshold).float()

  return velocity_toward * scale


def base_velocity_at_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for base velocity when at the goal.
  
  Strongly penalizes any linear or angular velocity when the robot
  is within the goal threshold. This prevents drifting, stepping,
  or rotating when the robot should be standing still.
  
  Args:
    env: The environment instance.
    command_name: Name of the pose command.
    command_threshold: Distance threshold for "at goal" (meters).
    asset_cfg: Scene entity configuration.
    
  Returns:
    Squared magnitude of linear + angular velocity, scaled by proximity to goal.
  """
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  asset: Entity = env.scene[asset_cfg.name]
  
  # Check if at goal
  pose_error = command.error  # [B, 3]
  position_error = torch.norm(pose_error[:, :2], dim=1)
  heading_error = torch.abs(pose_error[:, 2])
  total_error = position_error + heading_error
  
  # Only penalize velocity when at goal
  at_goal_mask = (total_error < command_threshold).float()
  
  # Penalize both linear and angular velocity
  lin_vel = asset.data.root_link_lin_vel_b[:, :2]  # [B, 2] xy velocity in base frame
  ang_vel = asset.data.root_link_ang_vel_b[:, 2]   # [B] yaw velocity in base frame
  
  lin_vel_penalty = torch.sum(torch.square(lin_vel), dim=1)
  ang_vel_penalty = torch.square(ang_vel)
  
  total_penalty = lin_vel_penalty + ang_vel_penalty
  
  return total_penalty * at_goal_mask


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def base_height_deviation(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target base height.
  
  Encourages the robot to maintain a consistent base height during locomotion.
  Helps prevent crouching too low or standing too tall.
  
  Args:
    env: The environment instance.
    target_height: Desired height of the base (meters).
    asset_cfg: Scene entity configuration.
  
  Returns:
    Squared deviation from target height.
  """
  asset: Entity = env.scene[asset_cfg.name]
  current_height = asset.data.root_link_pos_w[:, 2]  # z-position
  height_error = current_height - target_height
  return torch.square(height_error)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_term(command_name)
    if command is not None:
      pose_error = command.error
      linear_error = torch.norm(pose_error[:, :2], dim=1)
      angular_error = torch.abs(pose_error[:, 2])
      total_error = linear_error + angular_error
      scale = (total_error > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_term(command_name)
    if command is not None:
      pose_error = command.error
      linear_error = torch.norm(pose_error[:, :2], dim=1)
      angular_error = torch.abs(pose_error[:, 2])
      total_error = linear_error + angular_error
      active = (total_error > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_term(command_name)
        if command is not None:
            pose_error = command.error
            linear_error = torch.norm(pose_error[:, :2], dim=1)
            angular_error = torch.abs(pose_error[:, 2])
            total_error = linear_error + angular_error
            scale = (total_error > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_term(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    pose_error = command.error
    linear_error = torch.norm(pose_error[:, :2], dim=1)
    angular_error = torch.abs(pose_error[:, 2])
    total_error = linear_error + angular_error
    active = (total_error > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_term(command_name)
  assert command is not None
  pose_error = command.error
  linear_error = torch.norm(pose_error[:, :2], dim=1)
  angular_error = torch.abs(pose_error[:, 2])
  total_error = linear_error + angular_error
  active = (total_error > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_term(command_name)
    if command is not None:
      pose_error = command.error
      linear_error = torch.norm(pose_error[:, :2], dim=1)
      angular_error = torch.abs(pose_error[:, 2])
      total_error = linear_error + angular_error
      active = (total_error > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    assert command is not None

    # Use pose error magnitude to determine activity level
    pose_error = command.error
    linear_speed = torch.norm(pose_error[:, :2], dim=1)
    angular_speed = torch.abs(pose_error[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still_at_goal(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        std: float = 0.75,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Reward for maintaining default joint positions when at goal.
    
    Returns a positive reward (0 to 1) that is higher when joints are closer
    to their default positions. Only active when within command_threshold of goal.
    
    Args:
        env: The environment instance.
        command_name: Name of the pose command.
        command_threshold: Distance threshold for "at goal" (meters).
        std: Standard deviation for exponential reward scaling.
        asset_cfg: Scene entity configuration.
    
    Returns:
        Exponential reward based on joint deviation from default pose.
    """
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(diff_angle), dim=1)
    reward = torch.exp(-error / std**2)
    
    # Only apply reward when at goal (per-environment masking)
    if command_name is not None:
        command = env.command_manager.get_term(command_name)
        if command is not None:
            pose_error = command.error
            linear_error = torch.norm(pose_error[:, :2], dim=1)
            angular_error = torch.abs(pose_error[:, 2])
            total_error = linear_error + angular_error
            at_goal = (total_error <= command_threshold).float()
            reward = reward * at_goal
    
    return reward


def arm_joint_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize arm joint deviation from default positions.
  
  Encourages arms to stay close to their default positions,
  reducing excessive arm swing or unnatural arm movements.
  """
  asset: Entity = env.scene[asset_cfg.name]
  current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  deviation = current_joint_pos - default_joint_pos
  return torch.sum(torch.square(deviation), dim=1)


def waist_joint_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize waist joint deviation from default positions.
  
  Encourages the waist to stay close to its default position,
  promoting upright posture and stability.
  """
  asset: Entity = env.scene[asset_cfg.name]
  current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  deviation = current_joint_pos - default_joint_pos
  return torch.sum(torch.square(deviation), dim=1)

