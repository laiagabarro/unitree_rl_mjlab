### Launch PoseTracking training
```bash
python scripts/train.py Unitree-G1-23Dof-PoseTracking-Flat --env.scene.num-envs=4096
```

The logs are uploaded in real time to WandB and recorded locally in `unitree_rl_mjlab/logs/rsl_rl/g1_23dof_pose_tracking/2026-xx-xx_xx-xx-xx`

### Play checkpoint file
You can use the viser web viewer to see the Reward graphs in realtime
```bash
python scripts/play.py Unitree-G1-23Dof-PoseTracking-Flat --wandb-run-path=<WANDB_RUN_PATH> --viewer=viser
```
Or you can play the policy in Mujoco
```bash
python scripts/play.py Unitree-G1-23Dof-PoseTracking-Flat --wandb-run-path=<WANDB_RUN_PATH>
```

### Deploying a policy
Copy the run path from WandB and use this script to deploy
```bash
python scripts/deploy_policy.py --run_path=<WANDB_RUN_PATH>
```
Assign a keypress to activate the policy and press Enter to confirm

This script edits the config file to add the new policy `/home/roberto/unitree_rl_mjlab/deploy/robots/g1_23dof/config/config.yaml`

### Run `unitree_mujoco`
```bash
cd ~/unitree_mujoco/simulate/build
./unitree_mujoco
```

### Run `g1_ctrl`
In another terminal run the controller
```bash
cd ~/unitree_rl_mjlab/deploy/robots/g1_23dof/build
./g1_ctrl
```
Press `7` to raise the robot

Press `8` to raise the robot

Press `<KEY>` to activate the policy

Press `9` to release the robot

### Editing PoseTracking training config

The main config is: `src/tasks/pose_tracking/pose_tracking_env_cfg.py`

The functions defined for rewards, observations, etc. are located in: `src/tasks/pose_tracking/mdp`

- For example, reward functions: `src/tasks/pose_tracking/mdp/rewards.py`

Other task environment config: `src/tasks/pose_tracking/config/g1_23dof/env_cfgs.py`

Config for PPO neural network: `src/tasks/pose_tracking/config/g1_23dof/rl_cfg.py`