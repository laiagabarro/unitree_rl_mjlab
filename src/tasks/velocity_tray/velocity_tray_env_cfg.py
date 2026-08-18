"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.velocity_tray.mdp as mdp


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.debug_obs(mdp.builtin_sensor,"base_ang_vel"),
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.debug_obs(mdp.projected_gravity,"projected_gravity"),
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.debug_obs(mdp.generated_commands,"command"),
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.debug_obs(mdp.phase, "phase"),
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.debug_obs(mdp.joint_pos_rel,"joint_pos"),
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.debug_obs(mdp.joint_vel_rel,"joint_vel"),
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.debug_obs(mdp.last_action,"actions")),
    "height_scan": ObservationTermCfg(
      func=mdp.debug_obs(envs_mdp.height_scan, "height_scan"),
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.debug_obs(mdp.builtin_sensor, "base_lin_vel"),
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": ObservationTermCfg(
      func=mdp.debug_obs(envs_mdp.height_scan,"height_scan"),
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.debug_obs(mdp.foot_height,"foot_height"),
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.debug_obs(mdp.foot_air_time,"foot_air_time"),
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.debug_obs(mdp.foot_contact,"foot_contact"),
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.debug_obs(mdp.foot_contact_forces,"foot_contact_forces"),
      params={"sensor_name": "feet_ground_contact"},
    ),
    "cube_state": ObservationTermCfg(
      func=mdp.debug_obs(mdp.cube_state_relative_to_tray, "cube_state"),
      params={"tray_name": "tray", "cube_names": ("cube_0","cube_1","cube_2","cube_3")},
      ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.05,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_tray": EventTermCfg(
        func=mdp.reset_tray_at_hands,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "tray_name": "tray",
            "site_names": ("right_palm",),
            "z_offset": 0.02,
        },
    ),
    "randomize_cubes_physics": EventTermCfg(
        func=mdp.randomize_cubes_physics,
        mode="reset",
    ),
    "reset_cubes": EventTermCfg(
        func=mdp.reset_cubes_on_tray,
        mode="reset",
        params={
            "tray_name": "tray",
            "cube_names": (
                "cube_0",
                "cube_1",
                "cube_2",
                "cube_3",
            ),
            # The tray mesh top is about 0.0175 m above its local origin.
            # Keep the cube bottoms just above it instead of spawning them
            # slightly inside the mesh.
            "z_offset": 0.05,
            "num_cubes_min": 4,
            "num_cubes_max": 4,
            "x_center": 0.10,
            "x_half_range": 0.07,
            "z_half_range": 0.10,
            "min_separation": 0.01,
        },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(
      func=mdp.is_terminated_without_cube_fall,
      weight=-200.0,
    ),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.005),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "tray_orientation": RewardTermCfg(
      func=mdp.tray_orientation,
      weight=0.0,  # Ramped up via curriculum.
      params={"tray_name": "tray", "k": 20.0},
    ),
    # "cube_upright": RewardTermCfg(
    #   func=mdp.cube_upright,
    #   weight=0.0,  # Ramped up via curriculum.
    #   params={
    #     "tray_name": "tray",
    #     "cube_names": ("cube_0", "cube_1", "cube_2", "cube_3"),
    #     "k": 8.0,
    #   },
    # ),
    "cube_inside_tray": RewardTermCfg(
      func=mdp.cube_inside_tray,
      weight=0.0,  # Ramped up via curriculum.
      params={
        "tray_name": "tray",
        "cube_names": ("cube_0", "cube_1", "cube_2", "cube_3"),
        "height_threshold": 0.1,
      },
    ),
    "tray_angular_velocity": RewardTermCfg(
      func=mdp.tray_angular_velocity_penalty,
      weight=0.0,  # Ramped up via curriculum, same trigger as tray_orientation.
      params={"tray_name": "tray"},
    ),
    "tray_vertical_velocity": RewardTermCfg(
      func=mdp.tray_vertical_velocity_penalty,
      weight=0.0,  # Ramped up via curriculum, same trigger as tray_orientation.
      params={"tray_name": "tray"},
    ),
    "cube_linear_velocity": RewardTermCfg(
      func=mdp.cube_linear_velocity_penalty,
      weight=0.0,  # Ramped up via curriculum.
      params={"tray_name": "tray", "cube_names": ("cube_0","cube_1","cube_2","cube_3")},
    ),
    "cube_angular_velocity": RewardTermCfg(
      func=mdp.cube_angular_velocity_penalty,
      weight=0.0,  # Ramped up via curriculum.
      params={"tray_name": "tray", "cube_names": ("cube_0","cube_1","cube_2","cube_3")},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "cube_fallen": TerminationTermCfg(
      func=mdp.cube_fallen,
      params={
        "tray_name": "tray",
        "cube_names": ("cube_0", "cube_1", "cube_2", "cube_3"),
        "height_threshold": 0.1,
        "activation_reward": "cube_inside_tray",
      },
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
    "tray_orientation_curriculum": CurriculumTermCfg(
        func=mdp.reward_threshold_curriculum,
        params={
        "trigger_rewards": [
            ("track_linear_velocity", 0.6),
            ("track_angular_velocity", 0.5),
        ],
        "target_reward": "tray_orientation",
        "target_weight": 1.0,
        },
    ),
    # "cube_upright_curriculum": CurriculumTermCfg(
    #     func=mdp.reward_threshold_curriculum,
    #     params={
    #     "trigger_rewards": [("tray_orientation", 0.8)],
    #     "target_reward": "cube_upright",
    #     "target_weight": 0.5,
    #     },
    # ),
    "cube_lin_vel_curriculum": CurriculumTermCfg(
        func=mdp.reward_threshold_curriculum,
        params={
            "trigger_rewards": [("tray_orientation", 0.8)],
            "target_reward": "cube_linear_velocity",
            "target_weight": -0.02,
        },
    ),
        "cube_ang_vel_curriculum": CurriculumTermCfg(
        func=mdp.reward_threshold_curriculum,
        params={
            "trigger_rewards": [("tray_orientation", 0.8)],
            "target_reward": "cube_angular_velocity",
            "target_weight": -0.02,
        },
    ),
    "cube_inside_tray_curriculum": CurriculumTermCfg(
        func=mdp.reward_threshold_curriculum,
        params={
        "trigger_rewards": [("tray_orientation", 0.8)],
        "target_reward": "cube_inside_tray",
        "target_weight": 1.0,
        },
    ),
    "tray_ang_vel_curriculum": CurriculumTermCfg(
        func=mdp.reward_threshold_curriculum,
        params={
            "trigger_rewards": [
            ("track_linear_velocity", 0.6),
            ("track_angular_velocity", 0.5),
            ],
            "target_reward": "tray_angular_velocity",
            "target_weight": -0.15,
        },
        ),
        "tray_lin_vel_curriculum": CurriculumTermCfg(
        func=mdp.reward_threshold_curriculum,
        params={
            "trigger_rewards": [
            ("track_linear_velocity", 0.6),
            ("track_angular_velocity", 0.5),
            ],
            "target_reward": "tray_vertical_velocity",
            "target_weight": -0.15,
        },
        ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
