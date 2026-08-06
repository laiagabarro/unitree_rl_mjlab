"""Unitree G1-23DOF velocity environment configurations."""

from dataclasses import replace

from src.assets.objects import get_tray_cfg
from src.assets.robots import (
  G1_23DOF_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_23dof_constants import HOME_KEYFRAME
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_23dof_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1-23DOF rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {
    "robot": get_g1_23dof_robot_cfg(),
    "tray": get_tray_cfg(),
  }

  # Set raycast sensor frame to G1-23DOF pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23DOF_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
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

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_23dof_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1-23DOF flat terrain velocity configuration."""
  cfg = unitree_g1_23dof_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


# VLA arm pose: joints not listed default to 0.0 (remaining entries come from HOME_KEYFRAME).
_VLA_ARM_JOINT_POS = {
  "left_shoulder_pitch_joint":  -0.55,
  "left_shoulder_roll_joint":    0.25,
  "left_shoulder_yaw_joint":     0.0,
  "left_elbow_joint":           -0.25,
  "left_wrist_roll_joint":       0.0,
  "right_shoulder_pitch_joint": -0.55,
  "right_shoulder_roll_joint":  -0.25,
  "right_shoulder_yaw_joint":    0.0,
  "right_elbow_joint":          -0.25,
  "right_wrist_roll_joint":      0.0,
}


def unitree_g1_23dof_arms_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1-23DOF velocity config where the robot walks with arms in the VLA ready pose."""
  cfg = unitree_g1_23dof_flat_env_cfg(play=play)

  # Build a new initial state merging the HOME_KEYFRAME leg defaults with VLA arm positions.
  # Drop any HOME_KEYFRAME patterns that match arm joints to avoid them overriding the
  # explicit VLA entries (regex patterns could otherwise take precedence over exact names).
  _ARM_KEYWORDS = ("shoulder", "elbow", "wrist")
  home_non_arm = {
    k: v for k, v in HOME_KEYFRAME.joint_pos.items()
    if not any(kw in k for kw in _ARM_KEYWORDS)
  }
  vla_init_state = replace(
    HOME_KEYFRAME,
    joint_pos={**home_non_arm, **_VLA_ARM_JOINT_POS},
  )

  # Override the robot entity with the new default arm positions.
  robot_cfg = get_g1_23dof_robot_cfg()
  robot_cfg = replace(robot_cfg, init_state=vla_init_state)
  cfg.scene.entities = {"robot": robot_cfg}

  # Tighten arm pose reward stds so the policy learns to hold the VLA position.
  for std_key in ("std_walking", "std_running"):
    cfg.rewards["pose"].params[std_key] = {
      **cfg.rewards["pose"].params[std_key],
      r".*shoulder_pitch.*": 0.05,
      r".*shoulder_roll.*":  0.05,
      r".*shoulder_yaw.*":   0.05,
      r".*elbow.*":          0.05,
      r".*wrist.*":          0.05,
    }

  # _arm_cfg = SceneEntityCfg("robot", joint_names=[
  #   "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  #   "left_elbow_joint", "left_wrist_roll_joint",
  #   "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  #   "right_elbow_joint", "right_wrist_roll_joint",
  # ])

  # # L2 penalty on arm joint velocity: discourages any swinging motion.
  # cfg.rewards["arm_vel_l2"] = RewardTermCfg(
  #   func=mdp.joint_vel_l2,
  #   weight=-0.5,
  #   params={"asset_cfg": _arm_cfg},
  # )

  # # L2 penalty on arm joint acceleration: discourages jerky motion.
  # cfg.rewards["arm_acc_l2"] = RewardTermCfg(
  #   func=mdp.joint_acc_l2,
  #   weight=-0.01,
  #   params={"asset_cfg": _arm_cfg},
  # )

  return cfg
