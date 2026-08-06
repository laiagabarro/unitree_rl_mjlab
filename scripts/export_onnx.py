"""Script to export ONNX policy from a checkpoint."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# Import the deploy config exporter
sys.path.insert(0, str(Path(__file__).parent))
from export_deploy_cfg import export_deploy_cfg
sys.path.pop(0)


@dataclass
class ExportConfig:
    checkpoint_file: str
    """Path to the .pt checkpoint file"""
    
    output_dir: str | None = None
    """Output directory for policy.onnx (defaults to checkpoint directory)"""


def export_onnx(task_id: str, cfg: ExportConfig):
    """Export ONNX policy from a checkpoint."""
    configure_torch_backends()
    
    checkpoint_path = Path(cfg.checkpoint_file).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Default output to checkpoint directory
    if cfg.output_dir is None:
        output_dir = str(checkpoint_path.parent)
    else:
        output_dir = str(Path(cfg.output_dir).resolve())
    
    print(f"[INFO] Task: {task_id}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Load configurations
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    
    # Create environment
    device = "cpu"  # Use CPU for export
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load runner
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        from mjlab.rl import MjlabOnPolicyRunner
        runner_cls = MjlabOnPolicyRunner
    
    from dataclasses import asdict
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint...")
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
    
    # Export to ONNX
    print(f"[INFO] Exporting to ONNX...")
    os.makedirs(output_dir, exist_ok=True)
    runner.export_policy_to_onnx(output_dir, "policy.onnx")
    
    # Export deployment config
    print(f"[INFO] Exporting deployment config...")
    try:
        export_deploy_cfg(env.unwrapped, output_dir)
    except Exception as e:
        print(f"[WARNING] Failed to export deploy.yaml: {e}")
    
    onnx_path = Path(output_dir) / "policy.onnx"
    deploy_path = Path(output_dir) / "params" / "deploy.yaml"
    
    if onnx_path.exists():
        print(f"[SUCCESS] ONNX exported to: {onnx_path}")
        print(f"[INFO] File size: {onnx_path.stat().st_size / 1024:.2f} KB")
    else:
        print(f"[ERROR] Failed to export ONNX")
        sys.exit(1)
    
    if deploy_path.exists():
        print(f"[SUCCESS] Deploy config exported to: {deploy_path}")
    
    env.close()


def main():
    # Import tasks to populate the registry
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    
    cfg = tyro.cli(
        ExportConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    
    export_onnx(chosen_task, cfg)


if __name__ == "__main__":
    main()
