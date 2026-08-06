#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to deploy policy from WandB run or local checkpoint.

Usage examples:

  # Download and deploy policy from a WandB run (policy_type auto-detected from run config):
  python scripts/deploy_policy.py --run_path rodrigo55-tc/mjlab/m1eyjdec

  # Download with explicit policy_type and custom policy name:
  python scripts/deploy_policy.py --run_path rodrigo55-tc/mjlab/m1eyjdec \\
    --policy_type mimic \\
    --policy_name my_policy_name

  # Export and deploy from a local checkpoint (task and policy_type auto-detected from agent.yaml):
  python scripts/deploy_policy.py --checkpoint_file logs/rsl_rl/g1_23dof_mimic/2026-04-21_12-33-46/model_700.pt

  # Export from a local checkpoint with explicit task and policy_type:
  python scripts/deploy_policy.py \\
    --checkpoint_file logs/rsl_rl/g1_23dof_pose_tracking/2026-04-21_12-33-46/model_700.pt \\
    --task Unitree-G1-23Dof-Tracking-No-State-Estimation \\
    --policy_type pose_tracking \\
    --policy_name my_pose_tracking_policy
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
import wandb
import yaml


def extract_task_and_policy_type_from_checkpoint(checkpoint_path):
    """Extract task name and policy_type from agent.yaml in checkpoint directory.
    
    Returns:
        tuple: (task, policy_type) or (None, None) if extraction fails
    """
    checkpoint_path = Path(checkpoint_path)
    agent_yaml_path = checkpoint_path.parent / "params" / "agent.yaml"
    
    if not agent_yaml_path.exists():
        print(f"[WARNING] agent.yaml not found at: {agent_yaml_path}")
        return None, None
    
    try:
        # Read file as text and use regex to extract values
        # This avoids issues with !!python/tuple tags
        with open(agent_yaml_path, 'r') as f:
            content = f.read()
        
        # Extract wandb_group (task)
        task_match = re.search(r'^wandb_group:\s*(.+)$', content, re.MULTILINE)
        if not task_match:
            print(f"[WARNING] 'wandb_group' not found in agent.yaml")
            return None, None
        task = task_match.group(1).strip()
        
        # Extract experiment_name and derive policy_type
        exp_match = re.search(r'^experiment_name:\s*(.+)$', content, re.MULTILINE)
        if not exp_match:
            print(f"[WARNING] 'experiment_name' not found in agent.yaml")
            return task, None
        
        experiment_name = exp_match.group(1).strip()
        policy_match = re.search(r'g1_23dof_(.+)', experiment_name)
        if policy_match:
            policy_type = policy_match.group(1)
        else:
            print(f"[WARNING] Could not extract policy_type from experiment_name: {experiment_name}")
            return task, None
        
        print(f"[INFO] Extracted from agent.yaml: task='{task}', policy_type='{policy_type}'")
        return task, policy_type
        
    except Exception as e:
        print(f"[WARNING] Failed to read agent.yaml: {e}")
        traceback.print_exc()
        return None, None


def extract_task_and_policy_type_from_wandb_run(run):
    """Extract task name and policy_type from WandB run config.
    
    Args:
        run: wandb.Api().run object
        
    Returns:
        tuple: (task, policy_type) or (None, None) if extraction fails
    """
    try:
        config = run.config
        
        # Debug: print available top-level keys
        print(f"[DEBUG] Available config keys: {list(config.keys())}")
        
        # Try multiple ways to get experiment_name
        experiment_name = None
        if 'train_cfg' in config:
            train_cfg = config['train_cfg']
            print(f"[DEBUG] train_cfg type: {type(train_cfg)}")
            if isinstance(train_cfg, dict):
                # Check if it's wrapped in 'value'
                if 'value' in train_cfg:
                    experiment_name = train_cfg['value'].get('experiment_name')
                else:
                    experiment_name = train_cfg.get('experiment_name')

        if not experiment_name:
            print(f"[WARNING] 'experiment_name' not found in run config under 'train_cfg'")
            return None, None

        print(f"[DEBUG] Found experiment_name: {experiment_name}")
        
        # Extract policy_type from experiment_name (substring after "g1_23dof_")
        match = re.search(r'g1_23dof_(.+)', experiment_name)
        if not match:
            print(f"[WARNING] Could not extract policy_type from experiment_name: {experiment_name}")
            return None, None
        
        policy_type = match.group(1)
        
        # Also try to get task from wandb_group if available
        task = config.get('wandb_group')
        if not task and 'train_cfg' in config:
            train_cfg = config['train_cfg']
            if isinstance(train_cfg, dict):
                if 'value' in train_cfg:
                    task = train_cfg['value'].get('wandb_group')
                else:
                    task = train_cfg.get('wandb_group')
        
        print(f"[INFO] Extracted from WandB run: policy_type='{policy_type}'")
        if task:
            print(f"[INFO] Task: '{task}'")
        
        return task, policy_type
        
    except Exception as e:
        print(f"[WARNING] Failed to read WandB run config: {e}")
        traceback.print_exc()
        return None, None


def get_used_keys(config_data):
    """Extract all used key bindings from the config."""
    used_keys = set()
    if 'all_transitions' in config_data:
        for state_name, binding in config_data['all_transitions'].items():
            if isinstance(binding, str) and 'key_' in binding:
                # Extract key from "key_x.on_pressed"
                match = re.search(r'key_(\w+)\.', binding)
                if match:
                    used_keys.add(match.group(1))
    return used_keys


def get_next_fsm_id(config_data, id_range_start):
    """Find the next available FSM ID in the given range."""
    used_ids = set()
    if 'FSM' in config_data and '_' in config_data['FSM']:
        for state_name, state_config in config_data['FSM']['_'].items():
            if isinstance(state_config, dict) and 'id' in state_config:
                used_ids.add(state_config['id'])
    
    # Find next available ID starting from id_range_start
    next_id = id_range_start
    while next_id in used_ids:
        next_id += 1
    return next_id


def get_motion_duration(npz_path):
    """Return duration in seconds from a motion .npz file."""
    import numpy as np
    d = np.load(npz_path)
    fps = float(d['fps'].flat[0])
    num_frames = d['joint_pos'].shape[0]
    return num_frames / fps


def update_config_yaml(config_path, run_name, policy_type, motion_npz=None, motion_npz_path=None):
    """Update config.yaml to add new FSM state for the deployed policy."""
    print(f"\n[INFO] Updating config.yaml to add FSM state for '{run_name}'...")
    
    # Read the YAML file preserving structure
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Parse YAML
    config_data = yaml.safe_load(config_content)
    
    # Check if state already exists in all three places
    fsm_exists = (
        'FSM' in config_data and 
        '_' in config_data['FSM'] and 
        run_name in config_data['FSM']['_']
    )
    transition_exists = (
        'all_transitions' in config_data and 
        run_name in config_data['all_transitions']
    )
    state_config_exists = (
        'FSM' in config_data and 
        run_name in config_data['FSM']
    )
    
    if fsm_exists and transition_exists and state_config_exists:
        print(f"[INFO] FSM state '{run_name}' is already configured in config.yaml")
        print("[INFO] Skipping config.yaml update")
        return
    
    # Get used keys
    used_keys = get_used_keys(config_data)
    print(f"[INFO] Currently used keys: {sorted(used_keys)}")
    
    # Ask user for key binding
    while True:
        key = input(f"Enter an alphanumeric key for '{run_name}' transition: ").strip().lower()
        if not key or not re.match(r'^[a-z0-9]$', key):
            print("[ERROR] Please enter a single alphanumeric character (a-z, 0-9)")
            continue
        if key in used_keys:
            print(f"[ERROR] Key '{key}' is already assigned. Please choose another key.")
            continue
        break
    
    # Get next available ID (for pose_tracking, use 2XX range)
    if policy_type == "velocity":
        next_id = get_next_fsm_id(config_data, 101)
    elif policy_type == "pose_tracking":
        next_id = get_next_fsm_id(config_data, 201)
    elif policy_type == "mimic":
        next_id = get_next_fsm_id(config_data, 301)
    else:
        next_id = get_next_fsm_id(config_data, 401)
    
    print(f"[INFO] Assigning ID: {next_id}, Key: {key}")
    
    # Update config content by inserting the new entries
    # 1. Add to all_transitions at the end of the list
    # Find the last line of all_transitions (before the next YAML section)
    transitions_pattern = r'(all_transitions: &all_transitions\n(?:  \w+:.*\n)+)'
    
    def add_transition(match):
        existing = match.group(1)
        new_transition = f'  {run_name}: key_{key}.on_pressed\n'
        return existing + new_transition
    
    config_content = re.sub(transitions_pattern, add_transition, config_content)
    
    # 2. Add to FSM._ definitions at the end
    # Find the FSM._ section and add at the end before the first state definition (before next line that doesn't have indent)
    type_mapping = {
        "pose_tracking": "PoseTracking",
        "velocity": "RLBase",
        "mimic": "Mimic"
    }
    state_type = type_mapping.get(policy_type, "Unknown")
    
    # Pattern to find end of FSM._ section (before the next top-level key that starts without 4 spaces)
    fsm_defs_pattern = r'(  _: # enabled fsms\n(?:    \w+:\n(?:      \w+:.*\n)+)+)'
    
    def add_fsm_def(match):
        existing = match.group(1)
        new_fsm_def = f'''    {run_name}:
      id: {next_id}
      type: {state_type}
'''
        return existing + new_fsm_def
    
    config_content = re.sub(fsm_defs_pattern, add_fsm_def, config_content)
    
    # 3. Add state configuration at the end of the file
    if policy_type == "pose_tracking":
        new_state_config = f'''\n
  {run_name}:
    transitions: *all_transitions
    policy_dir: config/policy/{policy_type}/{run_name}
    goal_pose_topic: rt/goal_pose
    # debug_print: true
'''
    elif policy_type == "mimic" and motion_npz is not None:
        if motion_npz_path is not None:
            try:
                time_end = get_motion_duration(motion_npz_path)
                print(f"[INFO] Motion duration: {time_end:.3f}s")
            except Exception as e:
                print(f"[WARNING] Could not read motion duration: {e}")
                time_end = 500.0
        else:
            time_end = 500.0
        new_state_config = f'''\n
  {run_name}:
    transitions: *all_transitions
    motion_file: config/policy/mimic/{run_name}/params/{motion_npz}
    policy_dir: config/policy/mimic/{run_name}/
    time_start: 0.0
    time_end: {time_end:.3f}
'''
    else:
        new_state_config = f'''\n
  {run_name}:
    transitions: *all_transitions
    policy_dir: config/policy/{policy_type}/{run_name}
'''
    config_content = config_content.rstrip() + new_state_config
    
    # Write back to file
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"[SUCCESS] Updated config.yaml with new FSM state '{run_name}'")
    print(f"[SUCCESS] Key binding: {key}, FSM ID: {next_id}")


def derive_policy_name_from_checkpoint(checkpoint_path):
    """Derive policy name from checkpoint path.
    
    Example: logs/rsl_rl/g1_23dof_pose_tracking/2026-04-21_12-33-46/model_700.pt
             -> 2026-04-21_12-33-46_step_700
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Get the step number from filename (e.g., model_700.pt -> 700)
    filename = checkpoint_path.stem  # e.g., "model_700"
    match = re.search(r'model_(\d+)', filename)
    if not match:
        raise ValueError(f"Cannot extract step number from checkpoint filename: {checkpoint_path.name}")
    step = match.group(1)
    
    # Get the log directory name (parent directory of checkpoint)
    log_dir_name = checkpoint_path.parent.name  # e.g., "2026-04-21_12-33-46"
    
    policy_name = f"{log_dir_name}_step_{step}"
    return policy_name


def export_from_checkpoint(task, checkpoint_file, policy_type, policy_name):
    """Export policy.onnx and deploy.yaml from a checkpoint file.
    
    Returns:
        tuple: (policy_onnx_path, deploy_yaml_path) or (None, None) on failure
    """
    print(f"[INFO] Exporting policy from checkpoint: {checkpoint_file}")
    
    # Create a temporary directory for export
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run export_onnx script
        script_path = Path(__file__).parent / "export_onnx.py"
        cmd = [
            sys.executable,
            str(script_path),
            task,
            "--checkpoint_file", checkpoint_file,
            "--output_dir", temp_dir
        ]
        
        print(f"[INFO] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Export failed with exit code {result.returncode}")
            return None, None
        
        # Check if files were created
        policy_onnx_src = Path(temp_dir) / "policy.onnx"
        deploy_yaml_src = Path(temp_dir) / "params" / "deploy.yaml"
        
        if not policy_onnx_src.exists():
            print(f"[ERROR] policy.onnx not found at: {policy_onnx_src}")
            return None, None
        
        if not deploy_yaml_src.exists():
            print(f"[ERROR] deploy.yaml not found at: {deploy_yaml_src}")
            return None, None
        
        # Create output directories
        policy_dir = 'deploy/robots/g1_23dof/config/policy'
        output_base = os.path.join(policy_dir, policy_type, policy_name)
        
        exported_dir = os.path.join(output_base, "exported")
        params_dir = os.path.join(output_base, "params")
        
        os.makedirs(exported_dir, exist_ok=True)
        os.makedirs(params_dir, exist_ok=True)
        
        # Copy files to destination
        policy_onnx_dest = os.path.join(exported_dir, "policy.onnx")
        deploy_yaml_dest = os.path.join(params_dir, "deploy.yaml")
        
        shutil.copy2(policy_onnx_src, policy_onnx_dest)
        shutil.copy2(deploy_yaml_src, deploy_yaml_dest)
        
        print(f"[SUCCESS] Policy saved to: {policy_onnx_dest}")
        print(f"[SUCCESS] Deploy config saved to: {deploy_yaml_dest}")
        
        return policy_onnx_dest, deploy_yaml_dest


def main():
    """Deploy policy from WandB run or local checkpoint."""
    parser = argparse.ArgumentParser(description="Deploy policy from WandB run or local checkpoint.")
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--run_path",
        type=str,
        help="WandB run path (format: entity/project/run_id) to download policy from.",
    )
    source_group.add_argument(
        "--checkpoint_file",
        type=str,
        help="Path to local checkpoint file (.pt) to export policy from.",
    )
    
    # Common options
    parser.add_argument(
        "--policy_type",
        type=str,
        default=None,
        choices=["velocity", "mimic", "pose_tracking"],
        help="Type of policy to deploy (velocity, mimic, or pose_tracking). "
             "If not provided, will be extracted from agent.yaml (for --checkpoint_file) "
             "or from run config (for --run_path).",
    )
    parser.add_argument(
        "--policy_name",
        type=str,
        default=None,
        help="Name for the policy subdirectory. If not provided, will be derived from "
             "checkpoint path (for --checkpoint_file) or run name (for --run_path).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name. If not provided with --checkpoint_file, will be extracted from agent.yaml.",
    )
    
    args = parser.parse_args()
    
    # Validate and extract arguments for checkpoint_file path
    if args.checkpoint_file:
        checkpoint_path = Path(args.checkpoint_file)
        if not checkpoint_path.exists():
            print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
            return 1
        
        # Extract task and policy_type from agent.yaml if not provided
        extracted_task, extracted_policy_type = extract_task_and_policy_type_from_checkpoint(checkpoint_path)
        
        # Use extracted values if user didn't provide them
        task = args.task if args.task else extracted_task
        policy_type = args.policy_type if args.policy_type else extracted_policy_type
        
        # Validate we have all required values
        if not task:
            parser.error("Could not determine task. Please provide --task explicitly.")
        if not policy_type:
            parser.error("Could not determine policy_type. Please provide --policy_type explicitly.")
        
        # Determine policy name
        if args.policy_name:
            policy_name = args.policy_name
        else:
            try:
                policy_name = derive_policy_name_from_checkpoint(checkpoint_path)
                print(f"[INFO] Using derived policy name: {policy_name}")
            except ValueError as e:
                print(f"[ERROR] {e}")
                print("[INFO] Please provide --policy_name explicitly")
                return 1
        
        # Export and deploy
        onnx_path, yaml_path = export_from_checkpoint(
            task,
            args.checkpoint_file,
            policy_type,
            policy_name
        )
        
        if onnx_path is None or yaml_path is None:
            return 1
        
        # Update config.yaml
        config_path = "deploy/robots/g1_23dof/config/config.yaml"
        if not os.path.exists(config_path):
            print(f"[WARNING] Config file not found: {config_path}")
            print("[WARNING] Skipping config.yaml update")
        else:
            update_config_yaml(config_path, policy_name, policy_type)
        
        return 0
    
    # WandB path requires policy_type to be specified
    if args.run_path:
        print(f"[INFO] Downloading policy from WandB run: {args.run_path}")
        return download_from_wandb(args.run_path, args.policy_type, args.policy_name)


def download_from_wandb(run_path, policy_type=None, policy_name_override=None):
    """Download policy.onnx from WandB run to local policies directory."""
    
    try:
        # Initialize WandB API
        api = wandb.Api()
        
        # Get the run
        run = api.run(run_path)
        print(f"[INFO] Connected to run: {run.name}")
        
        # Extract task and policy_type from run config if not provided
        if not policy_type:
            _, extracted_policy_type = extract_task_and_policy_type_from_wandb_run(run)
            policy_type = extracted_policy_type
            
            if not policy_type:
                print("[ERROR] Could not determine policy_type from run config.")
                print("[ERROR] Please provide --policy_type explicitly.")
                return 1
        
        # Determine policy name
        policy_name = policy_name_override if policy_name_override else run.name
    
        # Create output directory
        policy_dir = 'deploy/robots/g1_23dof/config/policy'
        output_dir = os.path.join(policy_dir, policy_type, policy_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory: {output_dir}")
        
        # List all artifacts for this run
        artifacts = list(run.logged_artifacts())
        
        # Find all exported_policy artifacts
        exported_policy_artifacts = []
        for artifact in artifacts:
            if artifact.type == "model" and "exported_policy" in artifact.name:
                exported_policy_artifacts.append(artifact)
        
        if not exported_policy_artifacts:
            print("[ERROR] No exported_policy artifact found in this run.")
            print("[INFO] Available artifacts:")
            for artifact in artifacts:
                print(f"  - {artifact.name} (type: {artifact.type})")
            return 1
        
        # Sort by creation time and get the latest
        exported_policy_artifacts.sort(key=lambda x: x.created_at, reverse=True)
        exported_policy_artifact = exported_policy_artifacts[0]
        
        print(f"[INFO] Found {len(exported_policy_artifacts)} exported_policy artifact(s)")
        print(f"[INFO] Using latest artifact: {exported_policy_artifact.name} (created: {exported_policy_artifact.created_at})")
        
        # Download the artifact
        artifact_dir = exported_policy_artifact.download()
        print(f"[INFO] Downloaded to: {artifact_dir}")
        
        # Download files to output directory
        exported_dir = os.path.join(output_dir, "exported")
        params_dir = os.path.join(output_dir, "params")
        os.makedirs(exported_dir, exist_ok=True)
        os.makedirs(params_dir, exist_ok=True)
        
        # Copy policy.onnx from artifact
        onnx_source = os.path.join(artifact_dir, "policy.onnx")
        onnx_dest = os.path.join(exported_dir, "policy.onnx")
        
        if os.path.exists(onnx_source):
            shutil.copy2(onnx_source, onnx_dest)
            print(f"[SUCCESS] Policy saved to: {onnx_dest}")
        else:
            print(f"[ERROR] policy.onnx not found in artifact directory: {artifact_dir}")
            print("[INFO] Files in artifact:")
            for file in os.listdir(artifact_dir):
                print(f"  - {file}")
            return 1
        
        # Copy deploy.yaml from artifact
        yaml_source = os.path.join(artifact_dir, "deploy.yaml")
        yaml_dest = os.path.join(params_dir, "deploy.yaml")
        
        if os.path.exists(yaml_source):
            shutil.copy2(yaml_source, yaml_dest)
            print(f"[SUCCESS] Deploy config saved to: {yaml_dest}")
        else:
            print(f"[ERROR] deploy.yaml not found in artifact directory: {artifact_dir}")
            print("[INFO] Files in artifact:")
            for file in os.listdir(artifact_dir):
                print(f"  - {file}")
            return 1

        # Copy any .npz motion files from artifact
        npz_files = [f for f in os.listdir(artifact_dir) if f.endswith(".npz")]
        for npz_file in npz_files:
            npz_source = os.path.join(artifact_dir, npz_file)
            npz_dest = os.path.join(params_dir, npz_file)
            shutil.copy2(npz_source, npz_dest)
            print(f"[SUCCESS] Motion file saved to: {npz_dest}")
        if not npz_files:
            print("[INFO] No .npz motion file found in artifact (skipping)")
        motion_npz = npz_files[0] if npz_files else None
        motion_npz_path = os.path.join(params_dir, motion_npz) if motion_npz else None

        # Modify config.yaml to add new FSM state for the deployed policy
        config_path = "deploy/robots/g1_23dof/config/config.yaml"
        if not os.path.exists(config_path):
            print(f"[WARNING] Config file not found: {config_path}")
            print("[WARNING] Skipping config.yaml update")
        else:
            update_config_yaml(config_path, policy_name, policy_type, motion_npz=motion_npz, motion_npz_path=motion_npz_path)

        return 0
        
    except Exception as e:
        print(f"[ERROR] Failed to download from WandB: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
