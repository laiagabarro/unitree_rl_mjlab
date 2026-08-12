"""Event functions for tray velocity tasks."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

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
    """Reset tray so tray_center is positioned at the right palm."""

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.int,
        )

    robot = env.scene[robot_cfg.name]
    tray = env.scene[tray_name]

    if isinstance(site_names, str):
        site_names = (site_names,)

    # =========================================================
    # ONLY RIGHT HAND
    # =========================================================

    site_ids, resolved_names = robot.find_sites(("right_palm",))

    if len(site_ids) != 1:
        raise ValueError(
            f"Could not find exactly one right_palm site. "
            f"Found: {resolved_names}"
        )

    site_id = site_ids[0]

    # Update FK/site poses AFTER robot joint reset
    env.sim.forward()

    palm_pos = robot.data.site_pos_w[
        env_ids, site_id, :
    ].clone()

    palm_quat = robot.data.site_quat_w[
        env_ids, site_id, :
    ].clone()

    # =========================================================
    # DEBUG: RIGHT PALM
    # =========================================================

    if _DEBUG:
        print(
            f"[reset_tray_at_hands] env_ids={env_ids.tolist()} "
            f"site_names={resolved_names} site_id={site_id}"
        )

        print(
            f"[reset_tray_at_hands] palm_pos={palm_pos}"
        )

        print(
            f"[reset_tray_at_hands] palm_quat={palm_quat}"
        )

    # =========================================================
    # DEBUG AXES
    #
    # These are the local X/Y/Z axes expressed in world frame.
    # =========================================================

    axis_x = torch.tensor(
        [1.0, 0.0, 0.0],
        device=env.device,
        dtype=palm_quat.dtype,
    ).unsqueeze(0).expand(
        palm_quat.shape[0],
        -1,
    )

    axis_y = torch.tensor(
        [0.0, 1.0, 0.0],
        device=env.device,
        dtype=palm_quat.dtype,
    ).unsqueeze(0).expand(
        palm_quat.shape[0],
        -1,
    )

    axis_z = torch.tensor(
        [0.0, 0.0, 1.0],
        device=env.device,
        dtype=palm_quat.dtype,
    ).unsqueeze(0).expand(
        palm_quat.shape[0],
        -1,
    )

    if _DEBUG:
        palm_x_world = quat_apply(
            palm_quat,
            axis_x,
        )

        palm_y_world = quat_apply(
            palm_quat,
            axis_y,
        )

        palm_z_world = quat_apply(
            palm_quat,
            axis_z,
        )

        print(
            "[reset_tray_at_hands] RIGHT PALM WORLD AXES:"
        )

        print(
            f"    X = {palm_x_world}"
        )

        print(
            f"    Y = {palm_y_world}"
        )

        print(
            f"    Z = {palm_z_world}"
        )

    # =========================================================
    # TRAY CENTER LOCAL TRANSFORM
    #
    # food_tray.xml:
    #
    #     <site
    #         name="tray_center"
    #         pos="0 0 0.02"
    #         ...
    #     />
    #
    # The site itself has no local quaternion in the XML (identity),
    # but the tray needs to be offset and rotated relative to the
    # palm frame so it sits flat and faces the robot correctly.
    # tray_center_pos / tray_center_quat below encode that offset:
    #
    #     tray_center_world_pos  = root_pos  + root_quat * tray_center_pos
    #     tray_center_world_quat = root_quat * tray_center_quat
    #
    # We want tray_center to align with the right palm, so we solve
    # for root_pos / root_quat given palm_pos / palm_quat and this
    # fixed local offset.
    # =========================================================

    tray_center_pos = torch.tensor(
        [0.0, -0.1, -0.03],
        device=env.device,
        dtype=palm_pos.dtype,
    )

    # Base rotation: lay the tray flat (90 degrees around X).
    angle_x = math.pi / 2
    quat_x = torch.tensor(
        [math.cos(angle_x / 2), math.sin(angle_x / 2), 0.0, 0.0],
        device=env.device,
        dtype=palm_quat.dtype,
    )

    # Additional rotation: spin the tray around its own (now flat)
    # axis so its long side faces the robot.
    angle_z = math.pi / 2
    quat_z = torch.tensor(
        [math.cos(angle_z / 2), 0.0, 0.0, math.sin(angle_z / 2)],
        device=env.device,
        dtype=palm_quat.dtype,
    )

    tray_center_quat_single = quat_mul(quat_z, quat_x)

    tray_center_quat = tray_center_quat_single.unsqueeze(0).expand(
        palm_quat.shape[0],
        -1,
    )

    root_quat = quat_mul(
        palm_quat,
        quat_inv(tray_center_quat),
    )

    # =========================================================
    # DEBUG: TRAY AXES
    # =========================================================

    if _DEBUG:
        tray_x_world = quat_apply(
            root_quat,
            axis_x,
        )

        tray_y_world = quat_apply(
            root_quat,
            axis_y,
        )

        tray_z_world = quat_apply(
            root_quat,
            axis_z,
        )

        print(
            "[reset_tray_at_hands] TRAY CENTER WORLD AXES:"
        )

        print(
            f"    X = {tray_x_world}"
        )

        print(
            f"    Y = {tray_y_world}"
        )

        print(
            f"    Z = {tray_z_world}"
        )

    # =========================================================
    # POSITION
    #
    # tray_center is located at:
    #
    #     root_pos + root_quat * tray_center_pos
    #
    # We want:
    #
    #     tray_center == right_palm
    #
    # Therefore:
    #
    #     root_pos = palm_pos - rotated_offset
    #
    # =========================================================

    site_offset_world = quat_apply(
        root_quat,
        tray_center_pos.unsqueeze(0).expand(
            palm_pos.shape[0],
            -1,
        ),
    )

    root_pos = palm_pos - site_offset_world

    # =========================================================
    # DEBUG: POSITIONS
    # =========================================================

    if _DEBUG:
        reconstructed_tray_center_pos = (
            root_pos + site_offset_world
        )

        print(
            "[reset_tray_at_hands] POSITIONS:"
        )

        print(
            f"    right_palm  = {palm_pos}"
        )

        print(
            f"    tray_root   = {root_pos}"
        )

        print(
            f"    tray_center = {reconstructed_tray_center_pos}"
        )

        print(
            f"    center_error = "
            f"{reconstructed_tray_center_pos - palm_pos}"
        )

    # =========================================================
    # ROOT POSE
    # =========================================================

    root_pose = torch.cat(
        [root_pos, root_quat],
        dim=-1,
    )

    # =========================================================
    # DEBUG: QUATERNION VERIFICATION
    # =========================================================

    if _DEBUG:
        # Verify:
        #
        #     root_quat * tray_center_local_quat
        #         == palm_quat
        #
        reconstructed_palm_quat = quat_mul(
            root_quat,
            tray_center_quat,
        )

        print(
            "[reset_tray_at_hands] "
            "tray_center_local_quat="
            f"{tray_center_quat}"
        )

        print(
            f"[reset_tray_at_hands] root_pos={root_pos}"
        )

        print(
            f"[reset_tray_at_hands] root_quat={root_quat}"
        )

        print(
            "[reset_tray_at_hands] "
            f"reconstructed_palm_quat="
            f"{reconstructed_palm_quat}"
        )

        print(
            f"[reset_tray_at_hands] palm_quat="
            f"{palm_quat}"
        )

        print(
            f"[reset_tray_at_hands] writing root_pose="
            f"{root_pose}"
        )

    # =========================================================
    # WRITE TRAY POSE
    # =========================================================

    tray.write_root_link_pose_to_sim(
        root_pose,
        env_ids=env_ids,
    )

    # =========================================================
    # RESET TRAY VELOCITY
    # =========================================================

    tray.write_root_link_velocity_to_sim(
        torch.zeros(
            (len(env_ids), 6),
            device=env.device,
        ),
        env_ids=env_ids,
    )

    # Update simulation
    env.sim.forward()

    # =========================================================
    # DEBUG: ACTUAL TRAY POSE AFTER FORWARD
    # =========================================================

    if _DEBUG:
        actual_pose = tray.data.root_link_pose_w[
            env_ids
        ]

        print(
            "[reset_tray_at_hands] "
            "actual tray root_link_pose_w "
            f"AFTER forward={actual_pose}"
        )