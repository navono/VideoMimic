# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoMimic (CoRL 2025 Best Student Paper) is a framework for visual imitation learning on humanoid robots. It processes single-camera videos to reconstruct 3D environments and human motion, retargets motion to humanoid robots, trains policies via a 4-stage RL pipeline, and deploys to real Unitree G1 robots.

The pipeline flows: **real2sim** (video → 3D reconstruction + retargeting) → **simulation** (4-stage RL training) → **sim2real** (C++ deployment on robot).

## Environments

Three separate conda environments due to dependency conflicts:

- **vm1rs** (Python 3.12, CUDA 12.4, PyTorch 2.5.1) — real2sim preprocessing, optimization, retargeting
- **vm1recon** (Python 3.10, CUDA 11.8) — real2sim reconstruction (MegaSam, NKSR, GeoCalib)
- **rlgpu** (Python 3.11, CUDA 12.8, PyTorch 2.7.0) — simulation training/inference with IsaacLab

## Common Commands

### Real-to-Sim Pipeline (from `real2sim/`)

```bash
# Setup both environments
make setup

# Full pipeline: extract frames + all stages
make run VIDEO=video.mp4 NAME=my_video

# Run individual stages
make stage0 NAME=my_video    # Preprocessing (SAM2, ViTPose, VIMO, BSTRO)
make stage1 NAME=my_video    # MegaSam reconstruction
make stage2 NAME=my_video    # MegaHunter optimization
make stage3 NAME=my_video    # Postprocessing (GeoCalib, NKSR meshing)
make stage4 NAME=my_video    # Robot motion retargeting

# Override defaults: START=0 END=auto STRIDE=1 ROBOT=g1 HEIGHT=-1
# HEIGHT=0 uses G1 body shape; HEIGHT=1.8 fits SMPL to 1.8m

# Visualization (viser at localhost:8080)
make vis-env NAME=my_video
make vis-megahunter NAME=my_video
make vis-results NAME=my_video
```

### Simulation (from `simulation/`)

```bash
# Install simulation packages
cd videomimic_rl && pip install -e . && cd ..
cd videomimic_gym && pip install -e . && cd ..

# Download data
cd data && bash download_videomimic_data.sh

# Inference (viser UI at localhost:8080)
make play-terrain    # Terrain policy
make play-flat       # Distilled flat policy
make play-mcpt       # MCPT policy with reference joints

# Training
make train-s1                        # Stage 1: MoCap pre-training
make train-s2                        # Stage 2: Terrain RL tracking
make train-s3 LOAD_RUN=stage2_run    # Stage 3: Distillation
make train-s4 LOAD_RUN=stage3_run    # Stage 4: RL finetuning
# Override: NGPU=2 NUM_ENVS=4096
```

### Sim-to-Real (from `sim2real/`)

C++ build with ROS 1 Noetic + TorchScript + CUDA. See `sim2real/README.md`.

## Architecture

### Real-to-Sim (`real2sim/`)

Four sequential stages, each with its own Python module:

1. **`stage0_preprocessing/`** — SAM2 segmentation, ViTPose 2D poses, VIMO 3D meshes, BSTRO contact detection
2. **`stage1_reconstruction/`** — MegaSam monocular 3D reconstruction (produces `.h5` files)
3. **`stage2_optimization/`** — MegaHunter human-scene alignment, SMPL shape optimization
4. **`stage3_postprocessing/`** — GeoCalib gravity calibration, NKSR pointcloud-to-mesh
5. **`stage4_retargeting/`** — PyRoKi robot motion retargeting

Data flows through `demo_data/<NAME>/` with subdirectories per stage (cam01, masks, poses2d, meshes3d, contacts, megasam, megahunter, calib_mesh). Third-party code lives in `third_party/` with individual setup instructions.

### Simulation (`simulation/`)

**Environment hierarchy** (IsaacLab DirectRLEnv base):
```
DirectRLEnv → LeggedRobotEnv (legged_robot.py)
            → RobotDeepMimicEnv (robot_deepmimic.py)
            → G1DeepMimic (g1/g1_deepmimic.py)
```

- `videomimic_gym/legged_gym/envs/base/` — Base environments and configs
- `videomimic_gym/legged_gym/envs/g1/` — G1 robot-specific implementations
- `videomimic_gym/legged_gym/scripts/` — Training (`train.py`) and inference (`play.py`) entry points
- `videomimic_gym/resources/` — Robot URDF/MJCF assets, data config YAMLs
- `videomimic_rl/rsl_rl/` — PPO implementation, actor-critic networks, rollout storage

**4-stage training pipeline:**
1. MoCap pre-training (AMASS data, `--task=g1_deepmimic`)
2. Scene-conditioned tracking over terrain (`--task=g1_deepmimic_proj_heightfield`)
3. Distillation — removes MoCar reference dependency (`--task=g1_deepmimic_root_heightfield_no_history_dagger`)
4. RL finetuning with rewards

**Config system:** Dataclass-based with `configclass` decorator. Override any config via CLI flags: `--env.deepmimic.use_amass=True`, `--train.algorithm.learning_rate=2e-5`. Task names select the environment class and config.

### Sim-to-Real (`sim2real/`)

Single C++ file (`videomimic_inference_real.cpp`) with ROS 1 integration. Uses TorchScript-exported checkpoints. Requires elevation_mapping_humanoid for terrain awareness.

## Key Conventions

- All data for a video lives under `real2sim/demo_data/<NAME>/`
- Simulation configs are overridden via dot-path CLI args (Hydra-style), not config files
- Motion sources specified via YAML files in `resources/data_config/`
- RL checkpoints stored in `videomimic_gym/logs/` organized by run name
- Isaac Sim must be initialized before importing gym environments (lazy loading pattern)
- The project uses `numpy<2` throughout due to dependency constraints
