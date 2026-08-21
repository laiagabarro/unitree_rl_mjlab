from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def cube_fallen(
  env: ManagerBasedRlEnv,
  tray_name: str = "tray",
  cube_names: tuple[str, ...] = ("cube_0", "cube_1", "cube_2", "cube_3"),
  height_threshold: float = 0.1,
  activation_reward: str = "cube_inside_tray",
  activation_weight: float = 1.0,
) -> torch.Tensor:
  """Terminate once any cube has fallen, after the cube curriculum is complete.

  This stays disabled until the cube curriculum has reached
  ``activation_weight`` so that early walking and the reward ramp keep full
  episodes.
  """
  if env.reward_manager.get_term_cfg(activation_reward).weight < activation_weight:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  tray: Entity = env.scene[tray_name]
  site_id = tray.find_sites(("tray_center",))[0][0]
  tray_pos = tray.data.site_pos_w[:, site_id, :]
  tray_quat_inv = quat_inv(tray.data.site_quat_w[:, site_id, :])

  fallen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for cube_name in cube_names:
    cube: Entity = env.scene[cube_name]
    half_size = env.sim.model.geom_size[:, cube.indexing.geom_ids[0], 0]
    local_pos = quat_apply(tray_quat_inv, cube.data.root_link_pos_w - tray_pos)
    at_resting_height = torch.abs(local_pos[:, 1] - half_size) < height_threshold
    fallen |= ~at_resting_height
  return fallen


def is_terminated_without_cube_fall(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Keep the fall-over penalty, but not for the no-penalty cube termination."""
  return env.termination_manager.terminated & ~env.termination_manager.get_term(
    "cube_fallen"
  )
