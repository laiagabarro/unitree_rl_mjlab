from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformPose2dCommand(CommandTerm):
  cfg: UniformPose2dCommandCfg

  def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Pose command: [x, y, heading] in base frame
    self.pose_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    
    # Target pose in world frame [x, y, heading]
    self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.target_heading_w = torch.zeros(self.num_envs, device=self.device)
    
    # Tracking errors
    self.pose_error_b = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["error_pos_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    """Returns the pose command [x, y, heading] in base frame."""
    return self.pose_command_b
  
  @property
  def error(self) -> torch.Tensor:
    """Returns the pose error [x, y, heading] in base frame."""
    return self.pose_error_b

  def _update_metrics(self) -> None:
    """Update pose tracking error metrics."""
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    
    # Position error in xy
    self.metrics["error_pos_xy"] += (
      torch.norm(self.pose_error_b[:, :2], dim=-1) / max_command_step
    )
    
    # Heading error
    self.metrics["error_heading"] += (
      torch.abs(self.pose_error_b[:, 2]) / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample pose targets for the given environments."""
    # Sample target position (x, y) in base frame - create independent samples
    target_x_b = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_x)
    target_y_b = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_y)
    
    # Sample target heading
    target_heading_b = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.heading)
    
    # Transform target from current base frame to world frame
    base_pos_w = self.robot.data.root_link_pos_w[env_ids]
    base_quat_w = self.robot.data.root_link_quat_w[env_ids]
    
    # Transform position using quaternion rotation
    target_pos_b = torch.stack([target_x_b, target_y_b, torch.zeros_like(target_x_b)], dim=-1)
    target_pos_w_3d = quat_apply(base_quat_w, target_pos_b) + base_pos_w
    
    # Store target in world frame
    self.target_pos_w[env_ids] = target_pos_w_3d[:, :2]
    
    # Transform heading to world frame (extract yaw from base quaternion and add offset)
    # base_quat_w represents rotation from base to world
    # Quaternion format: [w, x, y, z] (scalar first)
    # We need to add the relative heading to the base's current yaw
    base_yaw = torch.atan2(
      2.0 * (base_quat_w[:, 0] * base_quat_w[:, 3] + base_quat_w[:, 1] * base_quat_w[:, 2]),
      1.0 - 2.0 * (base_quat_w[:, 2]**2 + base_quat_w[:, 3]**2)
    )
    self.target_heading_w[env_ids] = wrap_to_pi(base_yaw + target_heading_b)

  def _update_command(self) -> None:
    """Update pose error by transforming world target to current base frame."""
    # Get current robot pose in world frame
    base_pos_w = self.robot.data.root_link_pos_w
    base_quat_w = self.robot.data.root_link_quat_w
    
    # Transform target position from world to current base frame
    # target_pos_w is 2D, but we need 3D for rotation
    target_pos_w_3d = torch.cat([
      self.target_pos_w, 
      torch.zeros(self.num_envs, 1, device=self.device)
    ], dim=-1)
    
    # Vector from robot to target in world frame
    delta_pos_w = target_pos_w_3d - base_pos_w
    
    # Rotate to base frame (inverse transform)
    # quat_apply_inverse rotates vector from world to base
    from mjlab.utils.lab_api.math import quat_apply_inverse
    delta_pos_b = quat_apply_inverse(base_quat_w, delta_pos_w)
    
    # Position error in base frame (x, y)
    self.pose_error_b[:, :2] = delta_pos_b[:, :2]
    
    # Heading error: difference between target heading and current heading in world, wrapped to [-pi, pi]
    # Quaternion format: [w, x, y, z] (scalar first)
    base_yaw = torch.atan2(
      2.0 * (base_quat_w[:, 0] * base_quat_w[:, 3] + base_quat_w[:, 1] * base_quat_w[:, 2]),
      1.0 - 2.0 * (base_quat_w[:, 2]**2 + base_quat_w[:, 3]**2)
    )
    heading_error = wrap_to_pi(self.target_heading_w - base_yaw)
    self.pose_error_b[:, 2] = heading_error
    
    # Update command to reflect current error (for observation)
    self.pose_command_b[:] = self.pose_error_b

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create pose joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("pos_x", ranges.pos_x),
      ("pos_y", ranges.pos_y),
      ("heading", ranges.heading),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, (min_val, max_val) in axes:
        slider = server.gui.add_slider(
          label,
          min=min_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )
        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    # Update joystick target BEFORE computing errors to avoid one-frame delay
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      
      # Joystick provides values in base frame, convert to world frame
      base_pos_w = self.robot.data.root_link_pos_w[idx]
      base_quat_w = self.robot.data.root_link_quat_w[idx]
      
      # Get slider values
      target_x_b = self._joystick_sliders[0].value
      target_y_b = self._joystick_sliders[1].value
      target_heading_b = self._joystick_sliders[2].value
      
      # Transform to world frame
      target_pos_b = torch.tensor([target_x_b, target_y_b, 0.0], device=self.device)
      target_pos_w_3d = quat_apply(base_quat_w.unsqueeze(0), target_pos_b.unsqueeze(0)) + base_pos_w.unsqueeze(0)
      self.target_pos_w[idx] = target_pos_w_3d[0, :2]
      
      # Transform heading
      # Quaternion format: [w, x, y, z] (scalar first)
      base_yaw = torch.atan2(
        2.0 * (base_quat_w[0] * base_quat_w[3] + base_quat_w[1] * base_quat_w[2]),
        1.0 - 2.0 * (base_quat_w[2]**2 + base_quat_w[3]**2)
      )
      self.target_heading_w[idx] = wrap_to_pi(base_yaw + target_heading_b)
    
    # Now compute command with updated target
    super().compute(dt)

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw pose command visualization: target position and heading arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    target_pos_w = self.target_pos_w.cpu().numpy()
    target_heading_w = self.target_heading_w.cpu().numpy()

    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      target_xy = target_pos_w[batch]
      target_heading = target_heading_w[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Current position
      current_pos = np.array([base_pos_w[0], base_pos_w[1], base_pos_w[2] + z_offset])

      # Target position in world frame
      target_pos = np.array([target_xy[0], target_xy[1], base_pos_w[2] + z_offset])

      # 1. Arrow from current position to target position (blue).
      visualizer.add_arrow(
        current_pos, target_pos, color=(0.2, 0.2, 0.8, 0.7), width=0.02
      )

      # 2. Target heading arrow (green).
      # The heading shows the desired orientation at the target
      heading_length = 0.3
      heading_to = target_pos + np.array([
        np.cos(target_heading) * heading_length,
        np.sin(target_heading) * heading_length,
        0
      ])
      visualizer.add_arrow(
        target_pos, heading_to, color=(0.2, 0.8, 0.2, 0.8), width=0.025
      )


@dataclass(kw_only=True)
class UniformPose2dCommandCfg(CommandTermCfg):
  """Configuration for uniform 2D pose command."""
  entity_name: str

  @dataclass
  class Ranges:
    """Ranges for sampling pose targets in base frame."""
    pos_x: tuple[float, float]  # X position range in meters
    pos_y: tuple[float, float]  # Y position range in meters
    heading: tuple[float, float]  # Heading range in radians

  ranges: Ranges

  @dataclass
  class VizCfg:
    """Visualization configuration."""
    z_offset: float = 0.0  # Height offset for arrows
    scale: float = 1.0  # Scale for position vectors

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformPose2dCommand:
    return UniformPose2dCommand(self, env)
