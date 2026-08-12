"""Unitree G1-23DOF velocity environment configurations."""

from dataclasses import replace

import mujoco

from src.assets.objects.cube.cube_constants import get_cube_cfg
from src.assets.objects.tray.tray_constants import get_tray_cfg
from src.assets.robots import (
    G1_23DOF_ACTION_SCALE,
    get_g1_23dof_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_23dof_constants import HOME_KEYFRAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity_tray.mdp.events import reset_cube_on_tray
from src.tasks.velocity_tray.velocity_tray_env_cfg import make_velocity_env_cfg


_ROBOT_PALM_SITE = "robot/right_palm"
_TRAY_CENTER_SITE = "tray/tray_center"


def add_tray_weld(spec: mujoco.MjSpec) -> None:
    """Rigidly attach the tray center to the robot's right palm."""

    required_sites = {
        _ROBOT_PALM_SITE,
        _TRAY_CENTER_SITE,
    }

    available_sites = {
        site.name
        for site in spec.sites
    }

    missing_sites = required_sites - available_sites

    if missing_sites:
        raise ValueError(
            "Cannot create the robot-tray weld; missing scene site(s): "
            f"{sorted(missing_sites)}. "
            f"Available sites: {sorted(available_sites)}"
        )

    spec.add_equality(
        name="robot_tray_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        name1=_ROBOT_PALM_SITE,
        name2=_TRAY_CENTER_SITE,
    )


def unitree_g1_23dof_rough_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1-23DOF rough terrain velocity configuration."""

    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 48

    cfg.scene.entities = {
        "robot": get_g1_23dof_robot_cfg(),
        "tray": get_tray_cfg(),
    }

    cfg.scene.spec_fn = add_tray_weld

    # Set raycast sensor frame to G1-23DOF pelvis.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(
                sensor,
                RayCastSensorCfg,
            )
            sensor.frame.name = "pelvis"

    site_names = (
        "left_foot",
        "right_foot",
    )

    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in range(1, 8)
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (
        cfg.scene.sensors or ()
    ) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]

    assert isinstance(
        joint_pos_action,
        JointPositionActionCfg,
    )

    joint_pos_action.scale = G1_23DOF_ACTION_SCALE

    cfg.viewer.body_name = "torso_link"

    twist_cmd = cfg.commands["twist"]

    assert isinstance(
        twist_cmd,
        UniformVelocityCommandCfg,
    )

    twist_cmd.viz.z_offset = 1.15

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = geom_names

    cfg.events["base_com"].params[
        "asset_cfg"
    ].body_names = ("torso_link",)

    # Rationale for std values:
    #
    # - Knees/hip_pitch get the loosest std to allow natural leg
    #   bending during stride.
    # - Hip roll/yaw stay tighter to prevent excessive lateral sway.
    # - Ankle roll is very tight for balance.
    # - Shoulders/elbows get moderate freedom for natural arm swing.
    # - Wrists are loose since they don't affect balance much.

    cfg.rewards["pose"].params["std_standing"] = {
        ".*": 0.05,
    }

    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.15,
        r".*ankle_roll.*": 0.1,

        # Waist.
        r".*waist_yaw.*": 0.15,

        # Arms.
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }

    cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.25,
        r".*hip_yaw.*": 0.25,
        r".*knee.*": 0.5,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,

        # Waist.
        r".*waist_yaw.*": 0.25,

        # Arms.
        r".*shoulder_pitch.*": 0.25,
        r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.1,
        r".*wrist.*": 0.1,
    }

    cfg.rewards["body_orientation_l2"].params[
        "asset_cfg"
    ].body_names = ("torso_link",)

    cfg.rewards["body_ang_vel"].params[
        "asset_cfg"
    ].body_names = ("torso_link",)

    cfg.rewards["foot_clearance"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.rewards["foot_slip"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={
            "sensor_name": self_collision_cfg.name,
            "force_threshold": 10.0,
        },
    )

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)

        cfg.observations[
            "actor"
        ].enable_corruption = False

        cfg.events.pop(
            "push_robot",
            None,
        )

        cfg.curriculum = {}

        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if (
                cfg.scene.terrain.terrain_generator
                is not None
            ):
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


# Tray arm pose.
_TRAY_ARM_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_wrist_roll_joint": 1.57079632679,
}


def unitree_g1_23dof_flat_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1-23DOF flat terrain velocity configuration."""

    cfg = unitree_g1_23dof_rough_env_cfg(
        play=play,
    )

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # =========================================================
    # TRAY ARM INITIAL POSE
    # =========================================================

    _ARM_KEYWORDS = (
        "shoulder",
        "elbow",
        "wrist",
    )

    home_non_arm = {
        k: v
        for k, v in HOME_KEYFRAME.joint_pos.items()
        if not any(
            kw in k
            for kw in _ARM_KEYWORDS
        )
    }

    tray_init_state = replace(
        HOME_KEYFRAME,
        joint_pos={
            **home_non_arm,
            **_TRAY_ARM_JOINT_POS,
        },
    )

    robot_cfg = get_g1_23dof_robot_cfg()

    robot_cfg = replace(
        robot_cfg,
        init_state=tray_init_state,
    )

    # =========================================================
    # SCENE
    #
    # The cube is only added to the flat tray task.
    # =========================================================

    cfg.scene.entities = {
        "robot": robot_cfg,
        "tray": get_tray_cfg(),
        "cube": get_cube_cfg(),
    }

    # =========================================================
    # CUBE RESET
    #
    # This is intentionally inserted AFTER reset_tray.
    # Python dictionaries preserve insertion order, so the tray
    # is positioned first and the cube is then placed using the
    # final tray_center pose.
    # =========================================================

    assert "reset_tray" in cfg.events

    cfg.events["reset_cube"] = EventTermCfg(
        func=reset_cube_on_tray,
        mode="reset",
        params={
            "tray_name": "tray",
            "cube_name": "cube",
            "z_offset": 0.04,
        },
    )

    # =========================================================
    # FLAT TERRAIN
    # =========================================================

    assert cfg.scene.terrain is not None

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan.
    cfg.scene.sensors = tuple(
        s
        for s in (cfg.scene.sensors or ())
        if s.name != "terrain_scan"
    )

    del cfg.observations[
        "actor"
    ].terms["height_scan"]

    del cfg.observations[
        "critic"
    ].terms["height_scan"]

    # Disable terrain curriculum.
    cfg.curriculum.pop(
        "terrain_levels",
        None,
    )

    if play:
        twist_cmd = cfg.commands["twist"]

        assert isinstance(
            twist_cmd,
            UniformVelocityCommandCfg,
        )

        twist_cmd.ranges.lin_vel_x = (
            -0.5,
            1.0,
        )

        twist_cmd.ranges.lin_vel_y = (
            -0.5,
            0.5,
        )

        twist_cmd.ranges.ang_vel_z = (
            -0.5,
            0.5,
        )

    return cfg


# VLA arm pose.
_VLA_ARM_JOINT_POS = {
    "left_shoulder_pitch_joint": -0.55,
    "left_shoulder_roll_joint": 0.25,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.25,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": -0.55,
    "right_shoulder_roll_joint": -0.25,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.25,
    "right_wrist_roll_joint": 0.0,
}


def unitree_g1_23dof_arms_up_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """G1-23DOF velocity config where the robot walks with arms in VLA pose."""

    cfg = unitree_g1_23dof_flat_env_cfg(
        play=play,
    )

    _ARM_KEYWORDS = (
        "shoulder",
        "elbow",
        "wrist",
    )

    home_non_arm = {
        k: v
        for k, v in HOME_KEYFRAME.joint_pos.items()
        if not any(
            kw in k
            for kw in _ARM_KEYWORDS
        )
    }

    vla_init_state = replace(
        HOME_KEYFRAME,
        joint_pos={
            **home_non_arm,
            **_VLA_ARM_JOINT_POS,
        },
    )

    robot_cfg = get_g1_23dof_robot_cfg()

    robot_cfg = replace(
        robot_cfg,
        init_state=vla_init_state,
    )

    # Preserve the tray and cube scene.
    cfg.scene.entities = {
        "robot": robot_cfg,
        "tray": get_tray_cfg(),
        "cube": get_cube_cfg(),
    }

    # Tighten arm pose reward stds.
    for std_key in (
        "std_walking",
        "std_running",
    ):
        cfg.rewards["pose"].params[std_key] = {
            **cfg.rewards["pose"].params[std_key],
            r".*shoulder_pitch.*": 0.05,
            r".*shoulder_roll.*": 0.05,
            r".*shoulder_yaw.*": 0.05,
            r".*elbow.*": 0.05,
            r".*wrist.*": 0.05,
        }

    return cfg