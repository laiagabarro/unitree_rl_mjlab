"""Event functions for tray velocity tasks."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

from src.assets.objects.cube.cube_constants import (
    CUBE_DEFAULT_MASS,
    CUBE_HALF_SIZE_MAX,
    CUBE_HALF_SIZE_MIN,
    CUBE_MASS_MAX,
    CUBE_MASS_MIN,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Set MJLAB_DEBUG_COMMANDS=1 to enable debug prints.
_DEBUG = os.getenv(
    "MJLAB_DEBUG_COMMANDS",
    "0",
).lower() in ("1", "true", "yes")


def reset_tray_at_hands(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_name: str = "tray",
    site_names: tuple[str, ...] = ("right_palm",),
    z_offset: float = 0.08,
) -> None:
    """Reset tray so tray_center is exactly at right_palm."""

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.int,
        )

    robot = env.scene[robot_cfg.name]
    tray = env.scene[tray_name]

    site_ids, resolved_names = robot.find_sites(("right_palm",))

    if len(site_ids) != 1:
        raise ValueError(
            "Could not find exactly one right_palm site. "
            f"Found: {resolved_names}"
        )

    palm_site_id = site_ids[0]

    env.sim.forward()

    palm_pos = robot.data.site_pos_w[env_ids, palm_site_id, :].clone()
    palm_quat = robot.data.site_quat_w[env_ids, palm_site_id, :].clone()

    tray_site_ids, tray_site_names = tray.find_sites(("tray_center",))

    if len(tray_site_ids) != 1:
        raise ValueError(
            "Could not find exactly one tray_center site. "
            f"Found: {tray_site_names}"
        )

    tray_site_id = tray_site_ids[0]

    tray_root_pos = tray.data.root_link_pos_w[env_ids].clone()
    tray_root_quat = tray.data.root_link_quat_w[env_ids].clone()

    tray_site_pos_w = tray.data.site_pos_w[env_ids, tray_site_id, :].clone()
    tray_site_quat_w = tray.data.site_quat_w[env_ids, tray_site_id, :].clone()

    tray_center_pos_local = quat_apply(
        quat_inv(tray_root_quat),
        tray_site_pos_w - tray_root_pos,
    )

    tray_center_quat_local = quat_mul(
        quat_inv(tray_root_quat),
        tray_site_quat_w,
    )

    if _DEBUG:
        print("[reset_tray_at_hands] RIGHT PALM")
        print(f"    pos  = {palm_pos}")
        print(f"    quat = {palm_quat}")

        print("[reset_tray_at_hands] TRAY CENTER LOCAL")
        print(f"    pos  = {tray_center_pos_local}")
        print(f"    quat = {tray_center_quat_local}")

    root_quat = quat_mul(
        palm_quat,
        quat_inv(tray_center_quat_local),
    )

    site_offset_world = quat_apply(
        root_quat,
        tray_center_pos_local,
    )

    root_pos = palm_pos - site_offset_world

    if _DEBUG:
        reconstructed_center = root_pos + site_offset_world
        reconstructed_quat = quat_mul(
            root_quat,
            tray_center_quat_local,
        )

        print("[reset_tray_at_hands] SOLVED TRAY ROOT")
        print(f"    root_pos  = {root_pos}")
        print(f"    root_quat = {root_quat}")

        print("[reset_tray_at_hands] RECONSTRUCTED CENTER")
        print(f"    pos  = {reconstructed_center}")
        print(f"    quat = {reconstructed_quat}")

        print("[reset_tray_at_hands] ERRORS")
        print(f"    position = {reconstructed_center - palm_pos}")
        print(f"    quaternion = {reconstructed_quat - palm_quat}")

    root_pose = torch.cat([root_pos, root_quat], dim=-1)

    tray.write_root_link_pose_to_sim(
        root_pose,
        env_ids=env_ids,
    )

    tray.write_root_link_velocity_to_sim(
        torch.zeros(
            (len(env_ids), 6),
            device=env.device,
            dtype=root_pose.dtype,
        ),
        env_ids=env_ids,
    )

    env.sim.forward()

    if _DEBUG:
        actual_tray_root = tray.data.root_link_pose_w[env_ids]
        actual_center_pos = tray.data.site_pos_w[env_ids, tray_site_id, :]
        actual_center_quat = tray.data.site_quat_w[env_ids, tray_site_id, :]

        print("[reset_tray_at_hands] AFTER FORWARD")
        print(f"    tray root = {actual_tray_root}")
        print(f"    center pos = {actual_center_pos}")
        print(f"    center quat = {actual_center_quat}")

        print("[reset_tray_at_hands] CENTER ERROR")
        print(f"    pos = {actual_center_pos - palm_pos}")
        print(f"    quat = {actual_center_quat - palm_quat}")


def randomize_cubes_physics(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    cube_names: tuple[str, ...] = (
        "cube_0",
        "cube_1",
        "cube_2",
        "cube_3",
    ),
) -> None:
    """Randomize cube size and mass at every environment reset."""

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.int,
        )

    if not cube_names:
        return

    alpha_min = 0.5 * math.log(
        CUBE_MASS_MIN / CUBE_DEFAULT_MASS
    )
    alpha_max = 0.5 * math.log(
        CUBE_MASS_MAX / CUBE_DEFAULT_MASS
    )

    for cube_name in cube_names:
        dr.geom_size(
            env,
            env_ids,
            ranges=(
                CUBE_HALF_SIZE_MIN,
                CUBE_HALF_SIZE_MAX,
            ),
            asset_cfg=SceneEntityCfg(
                name=cube_name,
                geom_names=["cube_geom"],
            ),
            distribution="uniform",
            operation="abs",
        )

        # dr.pseudo_inertia(
        #     env,
        #     env_ids,
        #     alpha_range=(
        #         alpha_min,
        #         alpha_max,
        #     ),
        #     distribution="uniform",
        #     asset_cfg=SceneEntityCfg(
        #         name=cube_name,
        #     ),
        # )

def reset_cubes_on_tray(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    tray_name: str = "tray",
    cube_names: tuple[str, ...] = (
        "cube_0",
        "cube_1",
        "cube_2",
        "cube_3",
    ),
    z_offset: float = 0.04,
    num_cubes_min: int = 0,
    num_cubes_max: int = 4,
    x_center: float = 0.10,
    x_half_range: float = 0.07,
    z_half_range: float = 0.10,
    min_separation: float = 0.01,
) -> None:
    """Randomize cube count and placement on the tray."""

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.int,
        )

    tray = env.scene[tray_name]

    if not cube_names:
        return

    if not 0 <= num_cubes_min <= num_cubes_max <= len(cube_names):
        raise ValueError(
            "Invalid cube count range: "
            f"{num_cubes_min}..{num_cubes_max} "
            f"for {len(cube_names)} cube entities."
        )

    batch_size = len(env_ids)

    env.sim.forward()

    site_ids, resolved_names = tray.find_sites(("tray_center",))

    if len(site_ids) != 1:
        raise ValueError(
            "Could not find exactly one tray_center site. "
            f"Found: {resolved_names}"
        )

    site_id = site_ids[0]

    tray_center_pos_w = tray.data.site_pos_w[env_ids, site_id, :].clone()
    tray_center_quat_w = tray.data.site_quat_w[env_ids, site_id, :].clone()

    num_cubes = torch.randint(
        num_cubes_min,
        num_cubes_max + 1,
        (batch_size,),
        device=env.device,
    )

    placed_positions = torch.zeros(
        (batch_size, len(cube_names), 3),
        device=env.device,
        dtype=tray_center_pos_w.dtype,
    )

    placed_mask = torch.zeros(
        (batch_size, len(cube_names)),
        device=env.device,
        dtype=torch.bool,
    )

    for cube_idx, cube_name in enumerate(cube_names):
        cube = env.scene[cube_name]
        active = num_cubes > cube_idx

        geom_ids = cube.indexing.geom_ids

        if len(geom_ids) != 1:
            raise ValueError(
                f"Expected exactly one geom for {cube_name}, "
                f"found {len(geom_ids)}."
            )

        geom_id = geom_ids[0]

        cube_half_size = env.sim.model.geom_size[
            env_ids, geom_id, 0
        ].clone()

        candidate_x = (
            x_center
            + (
                torch.rand(
                    batch_size,
                    device=env.device,
                    dtype=tray_center_pos_w.dtype,
                )
                * 2.0
                - 1.0
            )
            * x_half_range
        )

        candidate_z = (
            (
                torch.rand(
                    batch_size,
                    device=env.device,
                    dtype=tray_center_pos_w.dtype,
                )
                * 2.0
                - 1.0
            )
            * z_half_range
        )

        for _ in range(30):
            valid = active.clone()

            if cube_idx > 0:
                for previous_idx in range(cube_idx):
                    previous_active = placed_mask[:, previous_idx]
                    previous_half_size = placed_positions[
                        :, previous_idx, 2
                    ]

                    dx = (
                        candidate_x
                        - placed_positions[:, previous_idx, 0]
                    )

                    dz = (
                        candidate_z
                        - placed_positions[:, previous_idx, 1]
                    )

                    distance = torch.sqrt(dx * dx + dz * dz)

                    required_distance = (
                        cube_half_size
                        + previous_half_size
                        + min_separation
                    )

                    valid &= (
                        ~previous_active
                        | (distance >= required_distance)
                    )

            if torch.all(~active | valid):
                break

            resample_mask = active & ~valid

            new_x = (
                x_center
                + (
                    torch.rand(
                        batch_size,
                        device=env.device,
                        dtype=tray_center_pos_w.dtype,
                    )
                    * 2.0
                    - 1.0
                )
                * x_half_range
            )

            new_z = (
                (
                    torch.rand(
                        batch_size,
                        device=env.device,
                        dtype=tray_center_pos_w.dtype,
                    )
                    * 2.0
                    - 1.0
                )
                * z_half_range
            )

            candidate_x = torch.where(
                resample_mask,
                new_x,
                candidate_x,
            )

            candidate_z = torch.where(
                resample_mask,
                new_z,
                candidate_z,
            )

        placed_positions[:, cube_idx, 0] = candidate_x
        placed_positions[:, cube_idx, 1] = candidate_z
        placed_positions[:, cube_idx, 2] = cube_half_size
        placed_mask[:, cube_idx] = active

        offset_local = torch.stack(
            (
                candidate_x,
                cube_half_size + z_offset,
                candidate_z,
            ),
            dim=-1,
        )

        offset_world = quat_apply(
            tray_center_quat_w,
            offset_local,
        )

        cube_pos = tray_center_pos_w + offset_world
        cube_quat = tray_center_quat_w.clone()

        cube_pose = torch.cat(
            (cube_pos, cube_quat),
            dim=-1,
        )

        # hidden_pose = torch.zeros_like(cube_pose)
        # hidden_pose[:, 0] = 100.0 + cube_idx
        # hidden_pose[:, 1:] = torch.tensor(
        #     [100.0, 100.0, 0.5, 0.5, 0.5, 0.5],
        #     device=env.device,
        #     dtype=cube_pose.dtype,
        # )

        # final_pose = torch.where(
        #     active.unsqueeze(-1),
        #     cube_pose,
        #     hidden_pose,
        # )

        cube.write_root_link_pose_to_sim(
            cube_pose,
            env_ids=env_ids,
        )

        cube.write_root_link_velocity_to_sim(
            torch.zeros(
                (batch_size, 6),
                device=env.device,
                dtype=cube_pose.dtype,
            ),
            env_ids=env_ids,
        )

    env.sim.forward()
    

    if _DEBUG:
        print(
            "[reset_cubes_on_tray] "
            f"num_cubes={num_cubes.tolist()}"
        )
        print(
            "[reset_cubes_on_tray] "
            f"cube_positions_local={placed_positions}"
        )