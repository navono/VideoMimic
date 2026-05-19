# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoMimic (CoRL 2025 Best Student Paper) is a system for visual imitation learning for humanoid robot control. It takes single-camera RGB videos, reconstructs 3D environments and human motion, retargets motion to the Unitree G1 humanoid, and trains RL policies in simulation for real-world deployment.

## Repository Structure

Three independent pipelines, each with its own dependencies and conda environments:

- **`real2sim/`** — Vision pipeline: video → 3D scene + robot motion. Uses conda envs `vm1rs` and `vm1recon`.
- **`simulation/`** — RL training in Isaac Gym. Uses conda env `rlgpu`.
- **`sim2real/`** — C++ deployment on Jetson/PC with ROS 1 Noetic and TorchScript.

## Real-to-Sim Pipeline (`real2sim/`)

### Setup

Install third-party dependencies (SAM2, ViTPose, MegaSAM, etc.) and download model checkpoints. See `real2sim/docs/setup.md` for full instructions.

### Running the Pipeline

```bash
conda activate vm1rs

# Extract frames from video
python utilities/extract_frames_from_video.py --video-path video.MOV \
    --output-dir ./demo_data/input_images/<video_name>/cam01

# Run full pipeline
./process_video.sh <video_name> <start_frame> <end_frame> <subsample_factor> g1 <height>
# height: -1 = auto-detect, 0 = use G1 shape, or specify manually (e.g. 1.8)
```

### Pipeline Stages

1. **Preprocessing** (`stage0_preprocessing/`) — SAM2 segmentation, ViTPose 2D poses, VIMO 3D mesh, BSTRO contact detection
2. **Reconstruction** (`stage1_reconstruction/`) — MegaSAM environment reconstruction (uses `vm1recon` env)
3. **Optimization** (`stage2_optimization/`) — MegaHunter human-scene alignment, SMPL shape fitting
4. **Postprocessing** (`stage3_postprocessing/`) — Gravity calibration (GeoCalib), pointcloud→mesh (NKSR)
5. **Retargeting** (`stage4_retargeting/`) — PyRoKi collision/contact-aware human-to-robot motion retargeting

All data flows through `demo_data/` subdirectories (input_images, input_masks, input_2d_poses, input_3d_meshes, input_contacts, output_smpl_and_points, output_calib_mesh).

## Simulation Pipeline (`simulation/`)

### Setup

Requires Isaac Gym (Nvidia GPU). Conda env: `rlgpu`, Python 3.11, PyTorch 2.3.1 with CUDA 12.1.

```bash
conda activate rlgpu
cd simulation/videomimic_rl && pip install -e . && cd ..
cd simulation/videomimic_gym && pip install -e . && cd ..
```

### Training (4 stages, sequential)

```bash
# Stage 1: MoCap Pre-training
bash videomimic_gym/legged_gym/scripts/train_stage_1_mcpt.sh

# Stage 2: Terrain RL
bash videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh

# Stage 3: Distillation (requires stage 2 run name)
LOAD_RUN=<stage2_run_name> bash videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh ${LOAD_RUN}

# Stage 4: RL Finetuning (requires stage 3 run name)
LOAD_RUN=<stage3_run_name> bash videomimic_gym/legged_gym/scripts/train_stage_4_rl_finetune.sh ${LOAD_RUN}
```

Adjust `--nproc-per-node` in scripts to match available GPUs. Multi-node training is natively supported.

### Inference

```bash
bash videomimic_gym/legged_gym/scripts/play_terrain_policy.sh
bash videomimic_gym/legged_gym/scripts/play_flat_policy.sh
bash videomimic_gym/legged_gym/scripts/play_mcpt_policy.sh
```

Inference opens a viser UI at `localhost:8080` for visualization.

### Key Architecture

- `videomimic_gym/legged_gym/envs/g1/` — G1 robot environment configs and task definitions (DeepMimic-style)
- `videomimic_gym/legged_gym/scripts/train.py` — Entry point (uses `torchrun`)
- `videomimic_rl/rsl_rl/` — PPO-based RL algorithms (fork of rsl_rl)
- Training tasks: `g1_deepmimic`, `g1_deepmimic_proj_heightfield`, `g1_deepmimic_root_heightfield_no_history_dagger`
- Checkpoints stored in `simulation/data/checkpoints/`
- Motion data configs in `videomimic_gym/legged_gym/resources/data_config/`

## Sim-to-Real (`sim2real/`)

C++ deployment for Unitree G1. Builds with CMake against ROS Noetic, LibTorch, and CUDA.

```bash
mkdir build && cd build
export CMAKE_PREFIX_PATH=/path/to/torch:/path/to/ros_ws/devel:/opt/ros/noetic
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_ARCHITECTURES="87"
make
```

Main inference binary: `videomimic_real/videomimic_inference_real.cpp`. Requires elevation mapping from the external `elevation_mapping_humanoid` package.

## Key Conda Environments

| Environment | Pipeline | Purpose |
|---|---|---|
| `vm1rs` | real2sim | Preprocessing, optimization, retargeting |
| `vm1recon` | real2sim | MegaSAM reconstruction (stage 1) |
| `rlgpu` | simulation | Isaac Gym training and inference |
