"""Event functions for tray velocity tasks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Set MJLAB_DEBUG_COMMANDS=1 to enable debug prints.
_DEBUG = os.getenv("MJLAB_DEBUG_COMMANDS", "0").lower() in (
    "1",
    "true",
    "yes",
)


def reset_tray_at_hands(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_name: str = "tray",
    site_names: tuple[str, ...] = ("right_palm",),
    z_offset: float = 0.08,
) -> None:
    """Reset tray so its tray_center site is exactly at right_palm.

    The tray_center local position and orientation are taken from the
    MuJoCo site defined in food_tray.xml. Nothing is hard-coded here
    regarding the site's rotation.

    The desired constraint is:

        tray_center_world == right_palm_world

    for both position and orientation.
    """

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.int,
        )

    robot = env.scene[robot_cfg.name]
    tray = env.scene[tray_name]

    # ---------------------------------------------------------
    # RIGHT PALM
    # ---------------------------------------------------------

    site_ids, resolved_names = robot.find_sites(("right_palm",))

    if len(site_ids) != 1:
        raise ValueError(
            "Could not find exactly one right_palm site. "
            f"Found: {resolved_names}"
        )

    palm_site_id = site_ids[0]

    # Make sure FK is up to date after the robot reset.
    env.sim.forward()

    palm_pos = robot.data.site_pos_w[
        env_ids, palm_site_id, :
    ].clone()

    palm_quat = robot.data.site_quat_w[
        env_ids, palm_site_id, :
    ].clone()

    # ---------------------------------------------------------
    # TRAY CENTER SITE
    #
    # IMPORTANT:
    # Read the actual site transform from MuJoCo.
    #
    # This already includes:
    #
    #   pos="0 -0.1 -0.03"
    #   quat="0.5 0.5 0.5 0.5"
    #
    # from food_tray.xml.
    # ---------------------------------------------------------

    tray_site_ids, tray_site_names = tray.find_sites(("tray_center",))

    if len(tray_site_ids) != 1:
        raise ValueError(
            "Could not find exactly one tray_center site. "
            f"Found: {tray_site_names}"
        )

    tray_site_id = tray_site_ids[0]

    # At this point the tray is still at its current/default pose.
    # The site position/orientation relative to the tray root is
    # therefore obtained from the MuJoCo model itself.
    #
    # site_pos_w/site_quat_w are world-frame values, so we first
    # obtain the local transform from the tray root.
    tray_root_pos = tray.data.root_link_pos_w[
        env_ids
    ].clone()

    tray_root_quat = tray.data.root_link_quat_w[
        env_ids
    ].clone()

    tray_site_pos_w = tray.data.site_pos_w[
        env_ids, tray_site_id, :
    ].clone()

    tray_site_quat_w = tray.data.site_quat_w[
        env_ids, tray_site_id, :
    ].clone()

    # Convert the current world-frame site pose into the tray-root
    # local frame.
    tray_center_pos_local = quat_apply(
        quat_inv(tray_root_quat),
        tray_site_pos_w - tray_root_pos,
    )

    tray_center_quat_local = quat_mul(
        quat_inv(tray_root_quat),
        tray_site_quat_w,
    )

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------

    if _DEBUG:
        print(
            "[reset_tray_at_hands] RIGHT PALM"
        )
        print(f"    pos  = {palm_pos}")
        print(f"    quat = {palm_quat}")

        print(
            "[reset_tray_at_hands] TRAY CENTER LOCAL"
        )
        print(f"    pos  = {tray_center_pos_local}")
        print(f"    quat = {tray_center_quat_local}")

    # ---------------------------------------------------------
    # SOLVE ROOT ORIENTATION
    #
    # root_quat * tray_center_local_quat = palm_quat
    #
    # therefore:
    #
    # root_quat = palm_quat * inverse(tray_center_local_quat)
    # ---------------------------------------------------------

    root_quat = quat_mul(
        palm_quat,
        quat_inv(tray_center_quat_local),
    )

    # ---------------------------------------------------------
    # SOLVE ROOT POSITION
    #
    # tray_center_world =
    #
    #     root_pos + root_quat * tray_center_pos_local
    #
    # We want:
    #
    #     tray_center_world = palm_pos
    #
    # therefore:
    #
    #     root_pos =
    #         palm_pos -
    #         root_quat * tray_center_pos_local
    # ---------------------------------------------------------

    site_offset_world = quat_apply(
        root_quat,
        tray_center_pos_local,
    )

    root_pos = palm_pos - site_offset_world

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------

    if _DEBUG:
        reconstructed_center = (
            root_pos + site_offset_world
        )

        reconstructed_quat = quat_mul(
            root_quat,
            tray_center_quat_local,
        )

        print(
            "[reset_tray_at_hands] SOLVED TRAY ROOT"
        )
        print(f"    root_pos  = {root_pos}")
        print(f"    root_quat = {root_quat}")

        print(
            "[reset_tray_at_hands] RECONSTRUCTED CENTER"
        )
        print(f"    pos  = {reconstructed_center}")
        print(f"    quat = {reconstructed_quat}")

        print(
            "[reset_tray_at_hands] ERRORS"
        )
        print(
            f"    position = "
            f"{reconstructed_center - palm_pos}"
        )
        print(
            f"    quaternion = "
            f"{reconstructed_quat - palm_quat}"
        )

    # ---------------------------------------------------------
    # WRITE TRAY ROOT POSE
    # ---------------------------------------------------------

    root_pose = torch.cat(
        [root_pos, root_quat],
        dim=-1,
    )

    tray.write_root_link_pose_to_sim(
        root_pose,
        env_ids=env_ids,
    )

    # Reset tray velocity.
    tray.write_root_link_velocity_to_sim(
        torch.zeros(
            (len(env_ids), 6),
            device=env.device,
            dtype=root_pose.dtype,
        ),
        env_ids=env_ids,
    )

    env.sim.forward()

    # ---------------------------------------------------------
    # FINAL DEBUG
    # ---------------------------------------------------------

    if _DEBUG:
        actual_tray_root = tray.data.root_link_pose_w[
            env_ids
        ]

        actual_center_pos = tray.data.site_pos_w[
            env_ids, tray_site_id, :
        ]

        actual_center_quat = tray.data.site_quat_w[
            env_ids, tray_site_id, :
        ]

        print(
            "[reset_tray_at_hands] AFTER FORWARD"
        )
        print(f"    tray root = {actual_tray_root}")
        print(f"    center pos = {actual_center_pos}")
        print(f"    center quat = {actual_center_quat}")

        print(
            "[reset_tray_at_hands] CENTER ERROR"
        )
        print(
            f"    pos = {actual_center_pos - palm_pos}"
        )
        print(
            f"    quat = {actual_center_quat - palm_quat}"
        )


def reset_cube_on_tray(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    tray_name: str = "tray",
    cube_name: str = "cube",
    cube_half_size: float = 0.03,
    z_offset: float = 0.05,
) -> None:
    """Place a free cube on top of the tray.

    The cube is not welded to the tray.

    Its XY position is the tray_center position and its bottom face
    is placed on top of the tray surface.

    `cube_half_size` is the half-size of the cube geometry.

    For the current cube:

        half-size = 0.03 m
        full size = 0.06 m

    The cube center is therefore placed one half-size above the
    tray surface.
    """

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.int,
        )

    tray = env.scene[tray_name]
    cube = env.scene[cube_name]

    # ---------------------------------------------------------
    # TRAY CENTER
    # ---------------------------------------------------------

    env.sim.forward()

    site_ids, resolved_names = tray.find_sites(("tray_center",))

    if len(site_ids) != 1:
        raise ValueError(
            "Could not find exactly one tray_center site. "
            f"Found: {resolved_names}"
        )

    site_id = site_ids[0]

    tray_center_pos_w = tray.data.site_pos_w[
        env_ids, site_id, :
    ].clone()

    tray_center_quat_w = tray.data.site_quat_w[
        env_ids, site_id, :
    ].clone()

    # ---------------------------------------------------------
    # CUBE POSITION
    #
    # tray_center is the reference point on the tray.
    #
    # The cube is placed one half-size above that point along
    # the tray_center local +Z axis.
    #
    # This is important because the cube geometry is centered
    # on its root body.
    # ---------------------------------------------------------

    offset_local = torch.tensor(
        [0.1, cube_half_size + z_offset, 0.0],
        device=env.device,
        dtype=tray_center_pos_w.dtype,
    ).unsqueeze(0).expand(
        len(env_ids),
        -1,
    )

    offset_world = quat_apply(
        tray_center_quat_w,
        offset_local,
    )

    cube_pos = tray_center_pos_w + offset_world

    # Same orientation as tray_center.
    cube_quat = tray_center_quat_w.clone()

    cube_pose = torch.cat(
        [cube_pos, cube_quat],
        dim=-1,
    )

    if _DEBUG:
        print(
            "[reset_cube_on_tray]"
        )
        print(
            f"    tray_center = {tray_center_pos_w}"
        )
        print(
            f"    tray_center_quat = "
            f"{tray_center_quat_w}"
        )
        print(
            f"    cube_half_size = "
            f"{cube_half_size}"
        )
        print(
            f"    cube_pos = {cube_pos}"
        )
        print(
            f"    cube_quat = {cube_quat}"
        )

    # ---------------------------------------------------------
    # WRITE CUBE POSE
    # ---------------------------------------------------------

    cube.write_root_link_pose_to_sim(
        cube_pose,
        env_ids=env_ids,
    )

    cube.write_root_link_velocity_to_sim(
        torch.zeros(
            (len(env_ids), 6),
            device=env.device,
            dtype=cube_pose.dtype,
        ),
        env_ids=env_ids,
    )

    env.sim.forward()

    if _DEBUG:
        actual_cube_pose = cube.data.root_link_pose_w[
            env_ids
        ]

        print(
            "[reset_cube_on_tray] "
            f"actual cube pose = {actual_cube_pose}"
        )
