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
  activation_state_key: str = "cube_inside_tray",
) -> torch.Tensor:
  """Terminate once any cube has fallen, ramping in with the cube curriculum.

  Instead of a hard on/off switch tied to a reward weight threshold, this
  reads the same ``progress`` (0..1) that ``reward_threshold_curriculum``
  computes for ``activation_state_key`` and terminates each env with that
  probability when a cube has fallen. At progress=0 the termination never
  fires (early walking and the reward ramp keep full episodes); at
  progress=1 it always fires, same as before. In between, only a growing
  fraction of envs with a fallen cube are actually cut short each step,
  instead of every env across the whole batch flipping to "terminate on
  fall" in the same env.common_step_counter tick.
  """
  from .curriculums import get_curriculum_progress

  progress = get_curriculum_progress(env, activation_state_key)
  if progress <= 0.0:
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

  if progress >= 1.0:
    return fallen

  roll = torch.rand(env.num_envs, device=env.device)
  return fallen & (roll < progress)


def is_terminated_without_cube_fall(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Keep the fall-over penalty, but not for the no-penalty cube termination."""
  return env.termination_manager.terminated & ~env.termination_manager.get_term(
    "cube_fallen"
  )