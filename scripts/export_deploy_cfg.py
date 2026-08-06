from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv


# G1-23DOF SDK motor index mapping
# The G1-23DOF model has 23 joints (MuJoCo indices 0-22) that map to specific SDK motor IDs
# SDK motor IDs: 0-12 (legs+waist), 15-19 (left arm), 22-26 (right arm)
# Missing IDs: 13, 14 (waist roll/pitch not in 23DOF), 20, 21 (right wrist pitch/yaw not in 23DOF)
G1_23DOF_JOINT_IDS_MAP = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  # Legs and waist (0-12)
    15, 16, 17, 18, 19,  # Left arm (15-19)
    22, 23, 24, 25, 26   # Right arm (22-26)
]

# Motor specs for action scale calculation
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)

ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))

EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_5020 = 25.0


def _reflected_inertia(rotors: tuple[float, ...], gears: tuple[float, ...]) -> float:
    """Calculate reflected inertia for a motor configuration."""
    result = 0.0
    for i in range(3):
        gear_ratio = 1.0
        for j in range(i, 3):
            gear_ratio *= gears[j]
        result += rotors[i] * gear_ratio ** 2
    return result


def _calculate_motor_specs() -> dict[str, dict[str, float]]:
    """Calculate armature, stiffness, and action scale for each motor type."""
    natural_freq = 10 * 2.0 * math.pi
    
    specs = {}
    for name, rotors, gears, effort in [
        ("7520_14", ROTOR_INERTIAS_7520_14, GEARS_7520_14, EFFORT_7520_14),
        ("7520_22", ROTOR_INERTIAS_7520_22, GEARS_7520_22, EFFORT_7520_22),
        ("5020", ROTOR_INERTIAS_5020, GEARS_5020, EFFORT_5020),
    ]:
        armature = _reflected_inertia(rotors, gears)
        stiffness = armature * natural_freq**2
        scale = 0.25 * effort / stiffness
        specs[name] = {"armature": armature, "stiffness": stiffness, "scale": scale}
    
    # Ankle uses 2x 5020 motors
    specs["ankle"] = {
        "armature": specs["5020"]["armature"] * 2,
        "stiffness": specs["5020"]["stiffness"] * 2,
        "scale": 0.25 * (EFFORT_5020 * 2) / (specs["5020"]["stiffness"] * 2),
    }
    
    return specs


def _get_action_scales_g1_23dof() -> list[float]:
    """Calculate action scales for G1-23DOF based on motor specs."""
    specs = _calculate_motor_specs()
    
    # Joint order with motor types
    motor_types = [
        "7520_14",  # left_hip_pitch
        "7520_22",  # left_hip_roll
        "7520_14",  # left_hip_yaw
        "7520_22",  # left_knee
        "ankle",    # left_ankle_pitch
        "ankle",    # left_ankle_roll
        "7520_14",  # right_hip_pitch
        "7520_22",  # right_hip_roll
        "7520_14",  # right_hip_yaw
        "7520_22",  # right_knee
        "ankle",    # right_ankle_pitch
        "ankle",    # right_ankle_roll
        "7520_14",  # waist_yaw
        "5020",     # left_shoulder_pitch
        "5020",     # left_shoulder_roll
        "5020",     # left_shoulder_yaw
        "5020",     # left_elbow
        "5020",     # left_wrist_roll
        "5020",     # right_shoulder_pitch
        "5020",     # right_shoulder_roll
        "5020",     # right_shoulder_yaw
        "5020",     # right_elbow
        "5020",     # right_wrist_roll
    ]
    
    return [round(specs[motor_type]["scale"], 2) for motor_type in motor_types]


class _InlineListDumper(yaml.SafeDumper):
    """Force all lists to be serialized in YAML inline style ([a, b, c])."""


def _represent_list_inline(dumper: _InlineListDumper, data: list[Any]) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_InlineListDumper.add_representer(list, _represent_list_inline)


def _to_plain_value(value: Any) -> Any:
    """Recursively convert torch/tuple/custom objects to YAML-serializable basic types."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return _to_plain_value(value.detach().cpu().tolist())
    if isinstance(value, tuple):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_plain_value(v) for k, v in value.items()}
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return _to_plain_value(vars(value))
    if isinstance(value, float):
        # Align with reference deploy.yaml style:
        # - Most scales in (0, 1) range use 2 decimal places
        # - Larger values (like stiffness/damping) use 1 decimal place
        if abs(value) < 1.0:
            return float(f"{value:.2f}")
        return float(f"{value:.1f}")
    return value


def _obs_export_name(train_name: str, params: dict[str, Any]) -> str:
    """Map training-side observation names to deployment-side registered names."""
    if train_name == "command":
        cmd_name = params.get("command_name")
        if cmd_name == "twist":
            return "velocity_commands"
        if cmd_name == "pose_2d":
            return "pose_command"
        if cmd_name == "motion":
            return "motion_command"
    if train_name == "phase":
        return "gait_phase"
    if train_name == "joint_pos":
        return "joint_pos_rel"
    if train_name == "joint_vel":
        return "joint_vel_rel"
    if train_name == "actions":
        return "last_action"
    return train_name


def _obs_export_params(train_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Correct parameter keys for deployment, avoiding training/deployment naming differences."""
    out = dict(params)
    if train_name == "command":
        cmd_name = out.get("command_name")
        if cmd_name == "twist":
            out["command_name"] = "base_velocity"
        elif cmd_name == "pose_2d":
            out["command_name"] = "pose_command"
    if train_name == "phase":
        # For gait_phase, we may need to pass command_name for pose tracking
        if "command_name" in out and out["command_name"] == "pose_2d":
            out["command_name"] = "pose_command"
        out = {k: v for k, v in out.items() if k in {"period", "command_name"}}
    if train_name in {"joint_pos", "joint_vel", "actions"}:
        out = {}
    return out


def _build_joint_pd_from_cfg(env: ManagerBasedRlEnv) -> tuple[list[float], list[float]]:
    """Extract stiffness and damping for each simulation joint from robot articulation config."""
    robot = env.scene["robot"]
    num_joints = len(robot.joint_names)
    stiffness = [0.0] * num_joints
    damping = [0.0] * num_joints

    # Actuator runtime objects record the matched joint IDs; their cfg provides PD parameters
    for actuator in robot.actuators:
        cfg = actuator.cfg
        joint_ids = actuator._target_ids.tolist()  # noqa: SLF001
        for jid in joint_ids:
            stiffness[jid] = float(cfg.stiffness)
            damping[jid] = float(cfg.damping)
    return stiffness, damping


def export_deploy_cfg(env: ManagerBasedRlEnv, log_dir: Path):
    """Export deploy.yaml for deployment from the training environment."""
    output_path = Path(log_dir)
    if output_path.suffix.lower() != ".yaml":
        output_path = output_path / "params" / "deploy.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robot = env.scene["robot"]

    # Determine joint_ids_map based on robot type
    action_term = next(iter(env.action_manager._terms.values()))
    num_actions = int(action_term.action_dim)
    
    # For G1-23DOF, use the predefined SDK motor index mapping
    if num_actions == 23:
        joint_ids_map = G1_23DOF_JOINT_IDS_MAP
    else:
        # For other robots, use sequential mapping
        target_ids = action_term.target_ids.detach().cpu().tolist()
        joint_ids_map = [int(i) for i in target_ids]

    stiffness_sim, damping_sim = _build_joint_pd_from_cfg(env)
    default_joint_pos_sim = (
        robot.data.default_joint_pos[0].detach().cpu().tolist()
    )

    # For deployment, we need arrays indexed by action index (0 to num_actions-1)
    # These arrays will be sent in action order to the robot controller
    stiffness = [float(stiffness_sim[i]) for i in range(num_actions)]
    damping = [float(damping_sim[i]) for i in range(num_actions)]
    default_joint_pos = [float(default_joint_pos_sim[i]) for i in range(num_actions)]

    # For G1-23DOF, pad stiffness and damping arrays with 0.0 for missing SDK motor IDs
    if num_actions == 23:
        # SDK expects 29 motor indices (0-28)
        # Missing IDs: 13, 14, 20, 21, 27, 28
        sdk_array_size = 29  # Fixed size for G1-23DOF SDK
        padded_stiffness = [0.0] * sdk_array_size
        padded_damping = [0.0] * sdk_array_size
        
        for i, sdk_id in enumerate(joint_ids_map):
            padded_stiffness[sdk_id] = stiffness[i]
            padded_damping[sdk_id] = damping[i]
        
        stiffness = padded_stiffness
        damping = padded_damping

    cfg: dict[str, Any] = {
        "joint_ids_map": joint_ids_map,
        "step_dt": float(env.step_dt),
        "stiffness": stiffness,
        "damping": damping,
        "default_joint_pos": default_joint_pos,
    }

    # Export command ranges for velocity/pose tasks; keep empty dict for imitation tasks
    commands: dict[str, Any] = {}
    
    # Velocity command (twist -> base_velocity)
    if "twist" in env.cfg.commands:
        cmd_cfg = env.cfg.commands["twist"]
        ranges_cfg = getattr(cmd_cfg, "ranges", None)
        if ranges_cfg is None:
            raise ValueError("twist command config missing 'ranges', cannot export deploy commands")
        ranges = {
            "lin_vel_x": list(ranges_cfg.lin_vel_x),
            "lin_vel_y": list(ranges_cfg.lin_vel_y),
            "ang_vel_z": list(ranges_cfg.ang_vel_z),
            "heading": None,
        }
        commands["base_velocity"] = {"ranges": ranges}
    
    # Pose command (pose_2d -> pose_command)
    if "pose_2d" in env.cfg.commands:
        cmd_cfg = env.cfg.commands["pose_2d"]
        ranges_cfg = getattr(cmd_cfg, "ranges", None)
        if ranges_cfg is None:
            raise ValueError("pose_2d command config missing 'ranges', cannot export deploy commands")
        ranges = {
            "pos_x": list(ranges_cfg.pos_x),
            "pos_y": list(ranges_cfg.pos_y),
            "heading": list(ranges_cfg.heading),
        }
        commands["pose_command"] = {"ranges": ranges}
    
    cfg["commands"] = commands

    # Action terms
    cfg["actions"] = {}
    for term in env.action_manager._terms.values():
        action_name = term.__class__.__name__
        term_cfg = term.cfg
        action_dim = int(term.action_dim)

        # For G1-23DOF, calculate scales based on motor specs instead of using env values
        if num_actions == 23:
            scale = _get_action_scales_g1_23dof()
        else:
            scale = term._scale[0].detach().cpu().tolist()  # noqa: SLF001
        
        offset = term._offset[0].detach().cpu().tolist()  # noqa: SLF001
        clip = getattr(term_cfg, "clip", None)

        cfg["actions"][action_name] = {
            "clip": _to_plain_value(clip),
            "joint_names": list(getattr(term_cfg, "actuator_names", (".*",))),
            "scale": scale if isinstance(scale, list) else [float(scale)] * action_dim,
            "offset": offset if isinstance(offset, list) else [float(offset)] * action_dim,
            "joint_ids": None,
        }

    # Observation terms: prioritize exporting actor/policy network input group
    obs_group_name = "policy"
    if obs_group_name not in env.observation_manager.active_terms:
        obs_group_name = "actor"
    if obs_group_name not in env.observation_manager.active_terms:
        obs_group_name = next(iter(env.observation_manager.active_terms.keys()))

    obs_names = env.observation_manager.active_terms[obs_group_name]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs[obs_group_name]
    cfg["observations"] = {}

    for train_name, obs_cfg in zip(obs_names, obs_cfgs, strict=True):
        params = dict(obs_cfg.params)
        export_name = _obs_export_name(train_name, params)
        export_params = _obs_export_params(train_name, params)

        obs_sample = obs_cfg.func(env, **params)
        obs_dim = int(obs_sample.shape[1]) if obs_sample.ndim > 1 else int(obs_sample.shape[0])

        scale = obs_cfg.scale
        if scale is None:
            scale_list = [1.0] * obs_dim
        else:
            plain_scale = _to_plain_value(scale)
            if isinstance(plain_scale, list):
                scale_list = plain_scale
            else:
                scale_list = [float(plain_scale)] * obs_dim

        clip = _to_plain_value(obs_cfg.clip)
        if clip is not None and not isinstance(clip, list):
            clip = list(clip)

        history_length = int(obs_cfg.history_length) if obs_cfg.history_length else 1
        cfg["observations"][export_name] = {
            "params": _to_plain_value(export_params),
            "clip": clip,
            "scale": _to_plain_value(scale_list),
            "history_length": history_length,
        }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            _to_plain_value(cfg),
            f,
            Dumper=_InlineListDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=False,
            width=120,
        )

    print(f"[INFO] Deploy config exported to: {output_path}")