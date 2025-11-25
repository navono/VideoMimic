<div align="center">
  <h1 align="center">📹 <em>VideoMimic Gym</em> </h1>
</div>

<p align="center">
  <strong>This is a repository for learning locomotion over terrain from human videos. Includes implementations of normal walking, DeepMimic, and distillation. It also provides support for depth rendering, heightmap etc.</strong> 
</p>


## 📦 Installation and Configuration

Please refer to [setup.md](/doc/setup.md) for installation and configuration steps.

## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: Use the Gym simulation environment to let the robot interact with the environment and find a policy that maximizes the designed rewards. Real-time visualization during training is not recommended to avoid reduced efficiency.
- **Play**: Use the Play command to verify the trained policy and ensure it meets expectations.
- **Sim2Sim**: Deploy the Gym-trained policy to other simulators to ensure it's not overly specific to Gym characteristics.
- **Sim2Real**: Deploy the policy to a physical robot to achieve motion control.

## 🛠️ User Guide

### 1. Training

Run the following command to start training:

```bash
python legged_gym/scripts/train.py --task=xxx
```

For multi-GPU training, use torchrun:

```bash
torchrun --nproc-per-node <num_gpus> legged_gym/scripts/train.py --multi_gpu --task=xxx
```

#### ⚙️ Parameter Description

##### Basic Parameters
- `--task`: Required parameter; values can be:
  - `g1_deepmimic`: For normal RL training
  - `g1_deepmimic_dagger`: For policy cloning/distillation
- `--headless`: Defaults to starting with a graphical interface; set to true for headless mode (higher efficiency)
- `--resume`: Resume training from a checkpoint in the logs (resumes from the checkpoint at `load_run`)
- `--experiment_name`: Name of the experiment to run/load
- `--run_name`: Name of the run to execute/load
- `--load_run`: Name of the run to load; defaults to the latest run
- `--checkpoint`: Checkpoint number to load; defaults to the latest file
- `--num_envs`: Number of environments for parallel training
- `--seed`: Random seed
- `--max_iterations`: Maximum number of training iterations
- `--sim_device`: Simulation computation device; specify CPU as `--sim_device=cpu`
- `--rl_device`: Reinforcement learning computation device; specify CPU as `--rl_device=cpu`
- `--multi_gpu`: Enable multi-GPU training
- `--wandb_note`: Add notes to Weights & Biases logging (use quote `""` for strings with spaces)

We also are able to override the parameters from the python configs by setting `--env.x=y` for environment parameters or `--train.x=y` for training parameters. Some of the most important parameters are detailed below:

##### Environment Parameters (--env.*)
- `deepmimic.use_amass`: Use AMASS motion capture data (True/False)
- `deepmimic.amass_terrain_difficulty`: Difficulty level for AMASS terrain data. (AMASS data is paired with random rough terrains. 1 means no rough terrains, up to 5 which is sample between none and the hardest difficulty.)
- `deepmimic.use_human_videos`: Use human video data (True/False)
- `deepmimic.human_video_oversample_factor`: Oversampling factor for human video data. Basically if set it will create multiple human terrains. Useful if mixing a few human videos with a larger amass dataset.
- `deepmimic.amass_replay_data_path`: Path to AMASS data files. Can include a wildcard (eg. ACCAD_export_retargeted_vnp6/*.pkl)
- `deepmimic.human_video_folders`: List of folders containing human video data
- `deepmimic.init_velocities`: Initialize with velocities from reference motion (when resetting.)
- `deepmimic.randomize_start_offset`: Randomize starting position offset (when resetting, otherwise always will be initialised to the start of the motion.)
- `deepmimic.n_append`: Number of freeze frames to append to motion. Useful to force the model to be stable at the end of the motion.
- `deepmimic.link_pos_error_threshold`: Threshold for link position error. If any joint has a cartesian error from the reference above this value, the episode will be terminated.
- `deepmimic.is_csv_joint_only`: Whether to use CSV joint data only. (Used only for re-exporting Unitree LaFan data to pkl format).
- `deepmimic.cut_off_import_length`: Maximum length of imported motion (useful if importing super long motions.)
- `deepmimic.respawn_z_offset`: Vertical offset for respawning. Useful if your motions have feet intersecting the terrain and want to raise the root to prevent this.
- `deepmimic.weighting_strategy`: Weighting strategy to use for sampling the start positions within episodes. Options are "uniform" or "linear".
- `terrain.n_rows`: Number of terrain rows. Used for efficiency purposes (see below section on explaination of data loading)
- `asset.terminate_after_large_feet_contact_forces`: Whether to terminate the episode on large contact forces. Useful to restrict the robot from hitting the ground too hard.
- `asset.large_feet_contact_force_threshold`: Threshold for large contact forces
- `asset.use_alt_files`: Use alternative robot model files. This is used if you want to randomise the robot geometry slightly on different GPUs (eg. we have been experimenting with using sphere collision geometry.)
- `rewards.scales.xx`: Weight for xx rewards (see [g1_deepmimic_config.py](/legged_gym/envs/g1/g1_deepmimic_config.py) for possible values)
- `rewards.only_positive_rewards`: Use only positive rewards. Set to `True` for normal non-deepmimic envs, but recommended to set to `False` even though it crashes performance at the beginning of training, otherwise it will ignore penalties.
- `rewards.joint_pos_tracking_k`: Coefficient for joint position tracking. Basically reward for tracking joint position is exp(- <sum of joint position errors> * k) -- so higher values of k mean it only gets a reward for tracking closer to the reference. However if k is too high it may learn to ignore the reward. 
- `rewards.joint_vel_tracking_k`: Coefficient for joint velocity tracking. as above
- `rewards.link_pos_tracking_k`: Coefficient for link position tracking. as above.
- `rewards.collision`: Weight for collision penalties.
- `rewards.feet_contact_matching`: Weight for feet contact matching
- `normalization.clip_actions`: Maximum allowed action values. Recommended value: ~10 for G1.
- `normalization.clip_observations`: Maximum allowed observation values. Recommended value: ~100 for G1
- `control.beta`: Parameter controlling how much EMA to apply on action outputs (lower values = more averaging, 1.0=no ema)
- `domain_rand.randomize_base_mass`: Whether to randomize robot base mass
- `domain_rand.push_robots`: Whether to apply random pushes to robots
- `domain_rand.max_push_vel_xy`: Maximum push velocity in xy plane
- `domain_rand.max_push_vel_interval`: Maximum interval between pushes
- `domain_rand.torque_rfi_rand`: Whether to randomize torque RFI
- `domain_rand.p_gain_rand`: Whether to randomize P gains
- `domain_rand.p_gain_rand_scale`: Scale for P gain randomization
- `domain_rand.d_gain_rand`: Whether to randomize D gains
- `domain_rand.d_gain_rand_scale`: Scale for D gain randomization
- `domain_rand.control_delays`: Whether to add control delays
- `domain_rand.control_delay_min`: Minimum control delay
- `domain_rand.control_delay_max`: Maximum control delay

##### Training Parameters (--train.*)
- `policy.re_init_std`: Reinitialize policy with noise
- `policy.init_noise_std`: Standard deviation for policy initialization noise
- `algorithm.learning_rate`: Learning rate for training
- `algorithm.bc_loss_coef`: Coefficient for behavior cloning loss (for dagger)
- `algorithm.policy_to_clone`: Path to policy to clone (for dagger)
- `algorithm.bounds_loss_coef`: Coefficient for bounds loss. This basically prevents the policy meanactions from going outside the range specified by `clip_actions` (see above). Recommended value ~0.0005.
- `algorithm.entropy_coef`: Coefficient for entropy regularization. Higher values will support the policy std to encourage explorating continuing later in the episode. 
- `algorithm.schedule`: Learning rate schedule type. 'fixed' for fixed LR or `adaptive` for kl divergence based one.
- `algorithm.desired_kl`: Target KL divergence
- `runner.save_interval`: Interval between model saves

### Explanation of data loading

Currently, we have 2 types of data:
* VideoMimic data with terrains [here](../data/videomimic_captures)
* Other motion capture data without terrains [here](../data/unitree_lafan)

The loading is done in [replay_data.py](legged_gym/tensor_utils/replay_data.py). This class takes a list of pickle files. We then sample from it using the member methods. The motion clips are ingested as pkl files. The data is located in [data](../data) directory.

IsaacGym (and other simulators) generally like to batch use of terrains between different environments by having them share a terrain meshes. This makes things efficient, however it's annoying when we want different terrains for env. The workaround we have implemented is to concat the meshes for different terrains to one and have a global env_offsets variable (see [robot_deepmimic.py](/legged_gym/envs/base/robot_deepmimic.py)) which is added to the starting position of clips to align them with the terrain.

Another gotcha we found is that if robots overlap in the simulator, Isaac Gym registers collisions between them (though does not apply them -- the physics is correct but somehow becomes super slow). This is problematic if you have many robots doing the same motion on the same terrain at once as it blows up memory usage. Hence the `n_rows` variable, which will create multiple rows. This will effectively expand the number of terrains and reduce the number of overlapping robots.

The concatenation of the terrain meshes is done by [DeepMimicTerrain](/legged_gym/utils/deepmimic_terrain.py). This then computes the offsets given the clip index. 

We have two kinds of motion clips loading supported in the repository. They are picked up in [G1 Deepmimic class](/legged_gym/envs/g1/g1_deepmimic.py). The first one is normal amass motion clips. Specify the folder for this kinds of motions with `amass_replay_data_path` and enable/disable with use_amass. We pair these with random terrains. The second is human video data. Because this requires terrain information, we pick these up as files in folders with both pkl and mesh information. Toggle this with use_human_videos flag (see above docs of arguments), and can specify the list of human videos with human_video_folders=[ /list of paths of videos within the retargeted data folder/ ].



**Default Training Result Directory**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

#### Example Commands

1. Multi-GPU RL Training:
```bash
torchrun --nproc-per-node 2 legged_gym/scripts/train.py --multi_gpu --task=g1_deepmimic \
  --headless --wandb_note "new_ft_old_terrains" \
  --env.deepmimic.use_amass=False \
  --load_run 20250225_132031_g1_deepmimic --resume \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.human_video_oversample_factor=10 \
  --env.terrain.n_rows=6 \
  --train.policy.re_init_std=True \
  --train.policy.init_noise_std=0.5 \
  --train.algorithm.learning_rate=2e-5 \
  --env.deepmimic.n_append=50 \
  --env.deepmimic.link_pos_error_threshold=0.5 \
  --env.deepmimic.init_velocities=True \
  --env.deepmimic.randomize_start_offset=True \
  --env.asset.terminate_after_large_feet_contact_forces=False \
  --env.asset.use_alt_files=False
```

2. Policy Cloning (DAgger):
```bash
torchrun --nproc_per_node 2 legged_gym/scripts/train.py --task=g1_deepmimic_dagger \
  --multi_gpu --headless --wandb_note "distill" \
  --env.deepmimic.use_amass=False \
  --env.terrain.n_rows=10 \
  --env.deepmimic.amass_terrain_difficulty=1 \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.init_velocities=True \
  --env.deepmimic.randomize_start_offset=True \
  --env.rewards.scales.feet_orientation=0.0 \
  --env.control.beta=1.0 \
  --train.runner.save_interval=50 \
  --train.algorithm.policy_to_clone_jitted=False \
  --train.algorithm.policy_to_clone=logs/g1_deepmimic/20250317_152046_g1_deepmimic \
  --train.algorithm.bc_loss_coef=1.0 \
  --train.algorithm.learning_rate=1e-4 \
  --env.deepmimic.n_append=50 \
  --env.asset.terminate_after_large_feet_contact_forces=False \
  --num_envs 2048
```


Training dance (assuming you cloned retargeted_data as specified in the [setup](./doc/setup_en.md)):

```bash
torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
  --multi_gpu \
  --task=g1_deepmimic_mocap \
  --headless \
  --env.terrain.n_rows=4096 \
  --env.deepmimic.amass_replay_data_path=lafan_replay_data/env_11_dance1_subject2.pkl \
  --env.deepmimic.cut_off_import_length=1600
```

(if you don't have multiple GPUs remove the `multi_gpu` arg and just do `python legged_gym/scripts/train.py` instead.)

Checkpoint will be saved in `logs/g1_deepmimic/TAG` where the tag depends on the date and time. If you have WandB configured you should see a run with this tag also appear in there.

---

### 2. Play

To visualize the training results in Gym, run the following command:

```bash
python legged_gym/scripts/play.py --task=xxx
```

#### Basic Play Parameters
- `--num_envs`: Number of environments to visualize (default: 1)
- `--load_run`: Name of the run to load; defaults to the latest run
- `--checkpoint`: Checkpoint number to load; defaults to the latest file
- `--headless`: Run without GUI (useful for recording)

#### Visualization Options

##### 1. Standard Isaac Gym Visualization
The default visualization uses Isaac Gym's built-in viewer. This provides basic visualization capabilities but may be less interactive.

##### 2. Viser Visualization (Recommended)
Viser provides an enhanced visualization experience with more interactive features. You can also use it over the network. To use Viser:

```bash
python legged_gym/scripts/play.py --task=xxx --env.viser.enable=True
```

Viser-specific parameters:
- `env.viser.enable`: Enable Viser visualization
- `env.control.decimation`: Control update rate (higher values = slower visualization)
- `env.control.beta`: Smoothing factor for actions (lower values = smoother motion)

#### Example Commands

1. Basic visualization with latest model:
```bash
python legged_gym/scripts/play.py --task=g1_deepmimic --num_envs 1
```

2. Viser visualization with specific model and DeepMimic settings (eg. replaying the dance):
```bash
python legged_gym/scripts/play.py \\
  --task=g1_deepmimic_mocap \
  --env.viser.enable=True \
  --load_run TAG \
  --num_envs 1 \
  --env.deepmimic.amass_replay_data_path=lafan_replay_data/env_11_dance1_subject2.pkl \
  --headless
```

#### 💾 Export Network

Can export the network easily from the Viser UI.

---

### 3. Sim2Real (Physical Deployment)

Code currently unreleased, but we used [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) (for Python initial testing), [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) (for real deployment on Jetson), and [Humanoid Elevation Mapping](https://github.com/smoggy-P/elevation_mapping_humanoid) packages.

---

## 🎉 Acknowledgments

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [legged_gym](https://github.com/leggedrobotics/legged_gym): The foundation for training and running codes.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithm implementation.
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Hardware communication interface for physical deployment.
- [Unitree rl gym](https://github.com/unitreerobotics/unitree_rl_gym): Gym for unitree robots.

---