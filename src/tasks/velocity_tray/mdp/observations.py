from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def debug_obs(func, name: str):
    def wrapped(env, *args, **kwargs):
        value = func(env, *args, **kwargs)

        if not torch.isfinite(value).all():
            bad = ~torch.isfinite(value)

            print("\n================ OBSERVATION NaN/INF ================")
            print(f"Observation: {name}")
            print(f"Shape: {tuple(value.shape)}")
            print(f"Number of bad values: {bad.sum().item()}")
            print(
                "Bad indices:",
                torch.nonzero(bad, as_tuple=False)[:20]
                .detach()
                .cpu(),
            )
            print(
                "Values:",
                value.flatten()[:20].detach().cpu(),
            )
            print("======================================================\n")

        return value

    return wrapped


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

def cube_state_relative_to_tray(
  env: ManagerBasedRlEnv,
  tray_name: str = "tray",
  cube_names: tuple[str, ...] = ("cube_0", "cube_1", "cube_2", "cube_3"),
) -> torch.Tensor:
  """Per-cube position relative to the tray frame, plus tilt cosine.
  Privileged (critic-only) — not observable from robot proprioception."""
  tray: Entity = env.scene[tray_name]
  site_id = tray.find_sites(("tray_center",))[0][0]
  tray_pos = tray.data.site_pos_w[:, site_id, :]
  tray_quat_inv = quat_inv(tray.data.site_quat_w[:, site_id, :])
  z_tray_w = tray.data.root_link_quat_w  # reutilitzem el fix ja aplicat

  feats = []
  for cube_name in cube_names:
    cube: Entity = env.scene[cube_name]
    rel_pos = quat_apply(tray_quat_inv, cube.data.root_link_pos_w - tray_pos)
    z_cube_w = quat_apply(cube.data.root_link_quat_w, torch.tensor(
      [0.0, 0.0, 1.0], device=env.device
    ).expand(env.num_envs, 3))
    z_tray_up = quat_apply(z_tray_w, torch.tensor(
      [0.0, 0.0, 1.0], device=env.device
    ).expand(env.num_envs, 3))
    cos_theta = torch.sum(z_cube_w * z_tray_up, dim=-1, keepdim=True)
    feats.append(torch.cat([rel_pos, cos_theta], dim=-1))  # (N, 4) per cub
  return torch.cat(feats, dim=-1)  # (N, 16)
