"""Pose tracking task configuration.

This module provides a factory function to create a base pose tracking task config.
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.pose_tracking.mdp as mdp
from src.tasks.pose_tracking.mdp import UniformPose2dCommandCfg


def make_pose_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base pose_tracking tracking task configuration."""

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
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "pose_2d"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "pose_2d"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
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
    "position_error": MetricsTermCfg(
      func=mdp.position_error,
      params={"command_name": "pose_2d"},
    ),
    "heading_error": MetricsTermCfg(
      func=mdp.heading_error,
      params={"command_name": "pose_2d"},
    ),
    "pose_total_error": MetricsTermCfg(
      func=mdp.pose_total_error,
      params={"command_name": "pose_2d"},
    ),
    "base_lin_vel": MetricsTermCfg(
      func=mdp.base_lin_vel,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": MetricsTermCfg(
      func=mdp.base_ang_vel,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_deviation_from_stand": MetricsTermCfg(
      func=mdp.joint_deviation_from_stand,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
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
    "pose_2d": UniformPose2dCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),  # Disable automatic resampling; only resample on reset
      debug_vis=True,
      ranges=UniformPose2dCommandCfg.Ranges(
        pos_x=(-0.5, 0.5),
        pos_y=(-0.5, 0.5),
        heading=(-math.pi / 4, math.pi / 4),
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
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
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
    "resample_commands": EventTermCfg(
      func=mdp.resample_commands,
      mode="reset",
      params={
        "command_name": "pose_2d",
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
    "track_heading": RewardTermCfg(
      func=mdp.track_angular_pose2d,
      weight=5.0,
      params={"command_name": "pose_2d", "std": 0.5},
    ),
    "velocity_toward_goal": RewardTermCfg(
      func=mdp.velocity_toward_goal,
      weight=8.0,
      params={
        "command_name": "pose_2d",
        "distance_threshold": 0.2,
      },
    ),
    "base_velocity_at_goal": RewardTermCfg(
      func=mdp.base_velocity_at_goal,
      weight=-2.0,
      params={
        "command_name": "pose_2d",
        "command_threshold": 0.1,
      },
    ),
    "stand_still_at_goal": RewardTermCfg(
      func=mdp.stand_still_at_goal,
      weight=30.0,
      params={
        "command_name": "pose_2d",
        "command_threshold": 0.1,
        "std": 0.75,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.25,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=3.0,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.50,
        "command_threshold": 0.15,
        "command_name": "pose_2d",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-5.0,
      params={
        "target_height": 0.10,
        "command_name": "pose_2d",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "pose_2d",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "pose_2d",
        "command_threshold": 0.1,
      },
    ),
    "arm_joint_deviation": RewardTermCfg(
      func=mdp.arm_joint_deviation,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(
          ".*shoulder_pitch.*",
          ".*shoulder_roll.*",
          ".*shoulder_yaw.*",
          ".*elbow.*",
          ".*wrist.*",
        )),
      },
    ),
    "waist_joint_deviation": RewardTermCfg(
      func=mdp.waist_joint_deviation,
      weight=-2.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(
          ".*waist_yaw.*",
        )),
      },
    ),
    "base_height_deviation": RewardTermCfg(
      func=mdp.base_height_deviation,
      weight=-5.0,
      params={
        "target_height": 0.75,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "goal_reached": TerminationTermCfg(
      func=mdp.goal_reached,
      params={
        "command_name": "pose_2d",
        "position_tolerance": 0.05,
        "heading_tolerance": 0.05,
        "max_base_velocity": 0.05,
      },
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(40.0)},
    ),
    "too_low": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.4},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    # "action_smoothness": CurriculumTermCfg(
    #   func=mdp.reward_weight,
    #   params={
    #     "reward_name": "action_rate_l2",
    #     "weight_stages": [
    #       {"step": 0, "weight": -0.01},
    #       {"step": 200, "weight": -0.02},
    #       {"step": 400, "weight": -0.04},
    #     ],
    #   },
    # ),
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
    episode_length_s=10.0,
  )
