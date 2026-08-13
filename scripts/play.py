"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import wandb

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

sys.path.insert(0, str(Path(__file__).parent))
from export_deploy_cfg import export_deploy_cfg  # noqa: E402
sys.path.pop(0)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  wandb_run_path: str | None = None
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # --- DEBUG: manually re-enable tray/cube rewards to inspect them in the
  # viewer, since play mode clears the curriculum and would otherwise leave
  # them at weight=0 forever. Remove this before using the script for
  # anything other than visual debugging.
  
  for name, w in [
      ("tray_orientation", 1.0),
      ("cube_upright", 0.5),
      ("cube_inside_tray", 1.0),
      ("tray_angular_velocity", -0.05),
      ("tray_vertical_velocity", -0.05),
  ]:
      if name in env_cfg.rewards:
          env_cfg.rewards[name].weight = w

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    # Export policy to ONNX and upload to WandB.
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    export_model_dir = str(log_dir / "exported")
    print(f"[INFO]: Exporting policy to ONNX: {export_model_dir}")
    try:
      runner.export_policy_to_onnx(export_model_dir, "policy.onnx")
    except Exception as e:
      print(f"[WARNING]: Failed to export ONNX: {e}")

    print(f"[INFO]: Exporting deployment config")
    try:
      export_deploy_cfg(env.unwrapped, log_dir)
    except Exception as e:
      print(f"[WARNING]: Failed to export deploy.yaml: {e}")

    if cfg.wandb_run_path is not None:
      print("[INFO]: Uploading exported policy to WandB...")
      try:
        onnx_path = log_dir / "exported" / "policy.onnx"
        deploy_yaml_path = log_dir / "params" / "deploy.yaml"
        api = wandb.Api()
        run = api.run(cfg.wandb_run_path)
        # Delete existing exported_policy artifacts so W&B doesn't deduplicate and skip the upload.
        try:
          for old_artifact in run.logged_artifacts():
            if old_artifact.type == "model" and "exported_policy" in old_artifact.name:
              old_artifact.delete(delete_aliases=True)
              print(f"[INFO]: Deleted existing artifact: {old_artifact.name} (v{old_artifact.version})")
        except Exception as del_e:
          print(f"[WARNING]: Could not delete existing artifacts: {del_e}")
        artifact = wandb.Artifact(
          name="exported_policy",
          type="model",
          description=f"Exported policy for {task_id}",
        )
        artifact.add_file(str(onnx_path), name="policy.onnx")
        artifact.add_file(str(deploy_yaml_path), name="deploy.yaml")
        if cfg.motion_file is not None and Path(cfg.motion_file).exists():
          motion_file_name = Path(cfg.motion_file).name
          artifact.add_file(str(cfg.motion_file), name=motion_file_name)
          print(f"[INFO]: Adding motion file to artifact: {motion_file_name}")
        with wandb.init(
          entity=run.entity,
          project=run.project,
          id=run.id,
          resume="allow",
        ) as active_run:
          logged_artifact = active_run.log_artifact(artifact)
          print("[INFO]: Waiting for artifact upload to complete...")
          logged_artifact.wait()
        print("[INFO]: Successfully uploaded exported policy to WandB")
      except Exception as e:
        print(f"[WARNING]: Failed to upload to WandB: {e}")

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
