#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to upload exported policy to WandB run."""

import argparse
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
import wandb


def extract_task_from_checkpoint(checkpoint_path):
    """Extract task name from agent.yaml in checkpoint directory.
    
    Returns:
        str: task name or None if extraction fails
    """
    import re
    
    checkpoint_path = Path(checkpoint_path)
    agent_yaml_path = checkpoint_path.parent / "params" / "agent.yaml"
    
    if not agent_yaml_path.exists():
        print(f"[WARNING] agent.yaml not found at: {agent_yaml_path}")
        return None
    
    try:
        # Read file as text and use regex to extract values
        with open(agent_yaml_path, 'r') as f:
            content = f.read()
        
        # Extract wandb_group (task)
        task_match = re.search(r'^wandb_group:\s*(.+)$', content, re.MULTILINE)
        if not task_match:
            print(f"[WARNING] 'wandb_group' not found in agent.yaml")
            return None
        task = task_match.group(1).strip()
        
        print(f"[INFO] Extracted task from agent.yaml: '{task}'")
        return task
        
    except Exception as e:
        print(f"[WARNING] Failed to read agent.yaml: {e}")
        traceback.print_exc()
        return None


def export_policy_to_onnx(task, checkpoint_file, output_dir):
    """Export policy.onnx and deploy.yaml from a checkpoint file.
    
    Returns:
        tuple: (policy_onnx_path, deploy_yaml_path) or (None, None) on failure
    """
    print(f"[INFO] Exporting policy from checkpoint: {checkpoint_file}")
    
    # Run export_onnx script
    script_path = Path(__file__).parent / "export_onnx.py"
    cmd = [
        sys.executable,
        str(script_path),
        task,
        "--checkpoint_file", checkpoint_file,
        "--output_dir", output_dir
    ]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Export failed with exit code {result.returncode}")
        return None, None
    
    # Check if files were created
    policy_onnx_path = Path(output_dir) / "policy.onnx"
    deploy_yaml_path = Path(output_dir) / "params" / "deploy.yaml"
    
    if not policy_onnx_path.exists():
        print(f"[ERROR] policy.onnx not found at: {policy_onnx_path}")
        return None, None
    
    if not deploy_yaml_path.exists():
        print(f"[ERROR] deploy.yaml not found at: {deploy_yaml_path}")
        return None, None
    
    return str(policy_onnx_path), str(deploy_yaml_path)


def upload_to_wandb(run_path, policy_onnx_path, deploy_yaml_path, checkpoint_file):
    """Upload policy files to WandB run as artifact.
    
    Args:
        run_path: WandB run path (format: entity/project/run_id)
        policy_onnx_path: Path to policy.onnx file
        deploy_yaml_path: Path to deploy.yaml file
        checkpoint_file: Path to checkpoint file (used to derive step number)
        
    Returns:
        int: 0 on success, 1 on failure
    """
    try:
        import re
        
        # Extract step number from checkpoint filename
        checkpoint_path = Path(checkpoint_file)
        filename = checkpoint_path.stem  # e.g., "model_600"
        match = re.search(r'model_(\d+)', filename)
        if not match:
            print(f"[ERROR] Cannot extract step number from checkpoint filename: {checkpoint_path.name}")
            return 1
        step = match.group(1)
        
        # Initialize WandB API
        api = wandb.Api()
        
        # Get the run
        run = api.run(run_path)
        print(f"[INFO] Connected to run: {run.name} ({run_path})")
        
        policy_onnx_path = Path(policy_onnx_path)
        deploy_yaml_path = Path(deploy_yaml_path)
        
        # Create artifact with step-based name
        artifact_name = f"step_{step}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Exported ONNX policy and deployment config at step {step}"
        )
        
        # Add files to artifact
        artifact.add_file(str(policy_onnx_path), name="policy.onnx")
        artifact.add_file(str(deploy_yaml_path), name="deploy.yaml")
        
        print(f"[INFO] Uploading artifact '{artifact_name}' to {run_path}...")
        
        # Log artifact to the run
        with wandb.init(
            entity=run.entity,
            project=run.project,
            id=run.id,
            resume="allow"
        ) as active_run:
            # Log the artifact (this will create a new version if it exists)
            active_run.log_artifact(artifact)
        
        print(f"[SUCCESS] Uploaded artifact '{artifact_name}' to {run_path}")
        print(f"[INFO] Files in artifact:")
        print(f"  - policy.onnx")
        print(f"  - deploy.yaml")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Failed to upload to WandB: {e}")
        traceback.print_exc()
        return 1


def main():
    """Upload exported policy to WandB run."""
    parser = argparse.ArgumentParser(
        description="Upload exported policy (ONNX + deploy.yaml) to WandB run."
    )
    
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="Path to local checkpoint file (.pt) to export policy from.",
    )
    parser.add_argument(
        "--run_path",
        type=str,
        required=True,
        help="WandB run path (format: entity/project/run_id) to upload policy to.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name. If not provided, will be extracted from agent.yaml.",
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint file exists
    checkpoint_path = Path(args.checkpoint_file)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        return 1
    
    # Extract task from agent.yaml if not provided
    task = args.task
    if not task:
        task = extract_task_from_checkpoint(checkpoint_path)
        if not task:
            print("[ERROR] Could not determine task. Please provide --task explicitly.")
            return 1
    
    # Create temporary directory for export
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export policy to ONNX
        policy_onnx_path, deploy_yaml_path = export_policy_to_onnx(
            task,
            args.checkpoint_file,
            temp_dir
        )
        
        if policy_onnx_path is None or deploy_yaml_path is None:
            return 1
        
        # Upload to WandB
        result = upload_to_wandb(
            args.run_path,
            policy_onnx_path,
            deploy_yaml_path,
            args.checkpoint_file
        )
        
        return result


if __name__ == "__main__":
    exit(main())
