SHELL := /bin/bash

# --------------- Configurable variables ---------------
NUM_GPUS     ?= 2
NUM_ENVS     ?= 4096
LOAD_RUN     ?=
MOTION_CFG   ?= resources/data_config/human_motion_list_124_motions.yaml
HEADLESS     ?=
CUDA_DEVICES ?= 0,1

# --------------- Conda helpers ---------------
CONDA_EVAL := eval "$$(conda shell.bash hook)"

# Isaac Gym needs libpython shared library on LD_LIBRARY_PATH
# RTX 4090 fix: include conda lib for libpython, WSL lib for CUDA driver (warp),
# and set CUDA_HOME to conda env to avoid linking to system CUDA 12.8
LD_LIB := export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$CONDA_PREFIX/lib && export CUDA_HOME=$$CONDA_PREFIX &&

# RTX 4090 fix: CUDA init order is now handled in gymdeps.py
# (PyTorch CUDA is initialized BEFORE PhysX loads, preventing runtime corruption)
ISAAC_PRELOAD :=

# Skip wandb login requirement
WANDB := export WANDB_MODE=disabled &&

# GPU selection
CUDA := export CUDA_VISIBLE_DEVICES=$(CUDA_DEVICES) &&

# --------------- Paths ---------------
SCRIPTS_DIR := videomimic_gym/legged_gym/scripts

# --------------- Targets ---------------

.PHONY: train-stage1 train-stage2 train-stage3 train-stage4 \
        play-terrain play-flat play-mcpt play-sitting-standing play-teacher play-teacher-sitting-standing \
        tensorboard-ss-distill

# ---------- Training (4 stages, sequential) ----------

# Stage 1: MoCap Pre-training
train-stage1:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/train_stage_1_mcpt.sh

# Stage 2: Terrain RL
# Usage:
#   H20 (4 cards):   make train-stage2 NUM_GPUS=4
#   Current machine:  make train-stage2 NUM_GPUS=1 NUM_ENVS=2048
#   Custom data:      make train-stage2 NUM_GPUS=4 MOTION_CFG=resources/data_config/human_motion_list_123_motions.yaml
train-stage2:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(LD_LIB) $(WANDB) \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NUM_GPUS) legged_gym/scripts/train.py \
	--task=g1_deepmimic_proj_heightfield --headless \
	--env.terrain.n_rows=1 --num_envs=$(NUM_ENVS) \
	--run_name g1_deepmimic_terrain --wandb_note "videomimic_stage_2" \
	--env.deepmimic.human_motion_source=$(MOTION_CFG) \
	--load_run 20250410_063030_g1_deepmimic_mcpt --resume \
	--train.policy.re_init_std=True --train.policy.init_noise_std=0.5 \
	--train.algorithm.learning_rate=2e-5 --train.algorithm.schedule=fixed \
	--env.deepmimic.amass_terrain_difficulty=1 --env.deepmimic.upsample_data=True \
	--env.deepmimic.use_human_videos=True --env.deepmimic.use_amass=False \
	--env.deepmimic.link_pos_error_threshold=0.5 --train.runner.save_interval=500 \
	--env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.randomize_terrain_offset=False \
	--env.terrain.cast_mesh_to_heightfield=False --env.deepmimic.truncate_rollout_length=500 \
	--train.runner.load_model_strict=False \
	--env.rewards.scales.termination=-2000 --env.rewards.scales.alive=200.0 \
	--env.rewards.scales.ankle_action=-3.0 --env.rewards.scales.action_rate=-3.0

# Stage 3: Distillation (requires LOAD_RUN from stage 2)
train-stage3:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(LD_LIB) $(WANDB) \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NUM_GPUS) legged_gym/scripts/train.py \
	--task=g1_deepmimic_root_heightfield_no_history_dagger --headless \
	--env.terrain.n_rows=1 --num_envs=$(NUM_ENVS) \
	--run_name g1_deepmimic_distill --wandb_note "videomimic_stage_3" \
	--env.deepmimic.human_motion_source=$(MOTION_CFG) \
	--train.algorithm.learning_rate=1e-3 --train.algorithm.schedule=fixed \
	--env.deepmimic.upsample_data=True --env.deepmimic.use_human_videos=True \
	--env.deepmimic.use_amass=False \
	--env.deepmimic.link_pos_error_threshold=0.3 --train.runner.save_interval=500 \
	--env.deepmimic.respawn_z_offset=0.1 --env.terrain.cast_mesh_to_heightfield=False \
	--env.deepmimic.truncate_rollout_length=500 --train.runner.load_model_strict=False \
	--train.algorithm.policy_to_clone=$(LOAD_RUN)

# Stage 4: RL Finetuning (requires LOAD_RUN from stage 3)
train-stage4:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(LD_LIB) $(WANDB) \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NUM_GPUS) legged_gym/scripts/train.py \
	--task=g1_deepmimic_root_heightfield_no_history_dagger --headless \
	--env.terrain.n_rows=1 --num_envs=$(NUM_ENVS) \
	--run_name g1_deepmimic_finetune --wandb_note "stage_4_rl_finetune" \
	--env.deepmimic.human_motion_source=$(MOTION_CFG) \
	--train.algorithm.learning_rate=3e-5 --train.algorithm.bc_loss_coef=0.0 \
	--train.algorithm.schedule=fixed \
	--env.deepmimic.amass_terrain_difficulty=1 --env.deepmimic.upsample_data=True \
	--env.deepmimic.use_human_videos=True --env.deepmimic.use_amass=True \
	--env.deepmimic.amass_replay_data_path=lafan_walk_and_dance/*.pkl \
	--env.deepmimic.link_pos_error_threshold=0.5 --train.runner.save_interval=500 \
	--env.noise.add_noise=True --env.deepmimic.respawn_z_offset=0.1 \
	--train.runner.load_model_strict=False \
	--resume --load_run $(LOAD_RUN) \
	--train.policy.re_init_std=True --train.policy.init_noise_std=0.5

# ---------- Inference ----------

play-terrain:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/play_terrain_policy.sh

play-flat:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/play_flat_policy.sh

play-mcpt:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/play_mcpt_policy.sh

# make play-sitting-standing LOAD_RUN=20250414_170842_g1_deepmimic NUM_EPISODES=5

# Play with sitting_standing video data
# Usage: make play-sitting-standing LOAD_RUN=<distill_run_name> NUM_EPISODES=20
SS_PLAY_RUN ?= 20260603_094747_g1_deepmimic_ss_distill
# GPU PhysX currently fails silently on this RTX 4090/Isaac Gym runtime; use CPU
# PhysX for playback unless explicitly overridden.
SS_PLAY_SIM_DEVICE ?= cpu
SS_PLAY_PIPELINE ?= cpu
play-sitting-standing:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(CUDA) $(LD_LIB) $(WANDB) \
	FILTER_PHYSX_LOG=$(CURDIR)/videomimic_gym/logs/play-sitting-standing.console.log \
	$(FILTER_PHYSX) python $(SCRIPTS_DIR)/play.py --task=g1_deepmimic_root_heightfield_no_history_dagger \
	--headless \
	--sim_device=$(SS_PLAY_SIM_DEVICE) --pipeline=$(SS_PLAY_PIPELINE) \
	--load_run $(or $(LOAD_RUN),$(SS_PLAY_RUN)) --num_envs 1 \
	--env.deepmimic.use_amass=False \
	--env.deepmimic.use_human_videos=True \
	--env.deepmimic.human_motion_source=resources/data_config/sitting_standing_motion.yaml \
	--env.deepmimic.respawn_z_offset=0.1 \
	--env.deepmimic.link_pos_error_threshold=10.0 \
	--train.algorithm.bc_loss_coef=0.0 \
	--env.terrain.cast_mesh_to_heightfield=False \
	--train.runner.load_model_strict=True \
	--env.viser.enable=True \
	--run_name g1_deepmimic_sitting_standing \
	--max_episodes $(or $(NUM_EPISODES),10)

# Play teacher (Stage 2 terrain) checkpoint with AMASS data
# Uses g1_deepmimic_dagger task (different network from distill)
# Usage: make play-teacher NUM_EPISODES=5 SS_PLAY_SIM_DEVICE=cuda:0 SS_PLAY_PIPELINE=cuda
TEACHER_RUN ?= 20250414_170842_g1_deepmimic
SS_PLAY_SIM_DEVICE ?= cuda:0
SS_PLAY_PIPELINE ?= cuda
play-teacher:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(CUDA) $(LD_LIB) $(WANDB) \
	FILTER_PHYSX_LOG=$(CURDIR)/videomimic_gym/logs/play-teacher.console.log \
	$(FILTER_PHYSX) python $(SCRIPTS_DIR)/play.py --task=g1_deepmimic_dagger \
	--headless \
	--sim_device=$(SS_PLAY_SIM_DEVICE) \
	--pipeline=$(SS_PLAY_PIPELINE) \
	--load_run $(or $(LOAD_RUN),$(TEACHER_RUN)) \
	--num_envs 1 \
	--env.deepmimic.use_amass=True \
	--env.deepmimic.amass_terrain_difficulty=1 \
	--env.deepmimic.amass_replay_data_path="lafan_walk_and_dance/*.pkl" \
	--env.deepmimic.use_human_videos=False \
	--env.deepmimic.respawn_z_offset=0.1 \
	--env.deepmimic.link_pos_error_threshold=10.0 \
	--train.algorithm.bc_loss_coef=0.0 \
	--env.viser.enable=True \
	--max_episodes $(or $(NUM_EPISODES),10)

# Play teacher checkpoint with sitting-standing motion data
# Usage: make play-teacher-sitting-standing NUM_EPISODES=5
#        make play-teacher-sitting-standing LOAD_RUN=20250410_063030_g1_deepmimic_mcpt
play-teacher-sitting-standing:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(CUDA) $(LD_LIB) $(WANDB) \
	FILTER_PHYSX_LOG=$(CURDIR)/videomimic_gym/logs/play-teacher-sitting-standing.console.log \
	$(FILTER_PHYSX) python $(SCRIPTS_DIR)/play.py --task=g1_deepmimic_dagger \
	--headless \
	--sim_device=$(SS_PLAY_SIM_DEVICE) \
	--pipeline=$(SS_PLAY_PIPELINE) \
	--load_run $(or $(LOAD_RUN),$(TEACHER_RUN)) \
	--num_envs 1 \
	--env.deepmimic.use_amass=False \
	--env.deepmimic.use_human_videos=True \
	--env.deepmimic.human_motion_source=resources/data_config/sitting_standing_motion.yaml \
	--env.deepmimic.respawn_z_offset=0.5 \
	--env.deepmimic.init_velocities=False \
	--env.deepmimic.link_pos_error_threshold=10.0 \
	--train.algorithm.bc_loss_coef=0.0 \
	--env.terrain.cast_mesh_to_heightfield=False \
	--env.viser.enable=True \
	--max_episodes $(or $(NUM_EPISODES),5)

# ---------- Monitoring ----------

# Launch TensorBoard for the latest distillation run
# Usage: make tensorboard-ss-distill
#        make tensorboard-ss-distill LOG_DIR=videomimic_gym/logs/g1_deepmimic/<run_name>
#   判断收敛的信号
#   1. Loss/bc 不再下降 → 学生已经充分学到老师的策略
#   2. Episode/rew_joint_pos_tracking 趋于稳定高位 → 关节追踪效果好
#   3. success/overall 趋于稳定 → 成功率不再提升
#
#   ┌─────────────────────┬─────────────────────────────────────────────┬──────────────────────────────┐
#   │        指标         │                    含义                     │            怎么看            │
#   ├─────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
#   │ Loss/bc ⭐          │ Behavior cloning loss（学生模仿老师的差距） │ 最关键。持续下降→稳定 = 收敛 │
#   ├─────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
#   │ Loss/surrogate      │ PPO 策略优化 loss                            │ 辅助参考，下降趋势即可       │
#   ├─────────────────────┼─────────────────────────────────────────────┼──────────────────────────────┤
#   ┌───────────────────────────────────┬──────────────────────┬──────────────────────────────────────┐
#   │               指标                │         含义         │                怎么看                │
#   ├───────────────────────────────────┼──────────────────────┼──────────────────────────────────────┤
#   │ Episode/rew_joint_pos_tracking ⭐ │ 关节位置追踪奖励     │ 最关键。上升→稳定 = 动作越来越像参考 │
#   ├───────────────────────────────────┼──────────────────────┼──────────────────────────────────────┤
#   │ Episode/rew_link_pos_tracking ⭐  │ 身体部位位置追踪奖励 │ 上升→稳定 = 整体姿态越来越像         │
#   ├───────────────────────────────────┼──────────────────────┼──────────────────────────────────────┤
#   │ Episode/rew_torso_pos_tracking ⭐ │ 躯干位置追踪奖励     │ 上升→稳定 = 躯干运动越来越像         │
#   ├───────────────────────────────────┼──────────────────────┼──────────────────────────────────────┤
#   │ Episode/rew_termination           │ 终止惩罚             │ 越接近 0 越好（少摔倒）              │
#   ├───────────────────────────────────┼──────────────────────┼──────────────────────────────────────┤
#   ┌───────────────────┬────────────────────────┐
#   │       指标        │          含义          │
#   ├───────────────────┼────────────────────────┤
#   │ Train/mean_reward │ 总平均奖励（综合指标） │
#   ├───────────────────┼────────────────────────┤
#   │ success/overall   │ 整体成功率             │
#   ├───────────────────┼────────────────────────┤
#   │ success/clip_...  │ 单个动作片段的成功率   │
#   └───────────────────┴────────────────────────┘
SS_LOG_DIR ?= videomimic_gym/logs/g1_deepmimic
tensorboard-ss-distill:
	$(CONDA_EVAL) && conda activate rlgpu && \
	tensorboard --logdir $(SS_LOG_DIR) --bind_all


# ---------- Sitting/Standing Training ----------
# Uses pre-trained Stage 2 checkpoint as teacher for distillation, then fine-tunes.
# No need to run base Stage 1-2; pre-trained checkpoints are provided.

# Filter PhysX C++ noise from stderr (harmless warnings on RTX 4090)
FILTER_PHYSX := $(CURDIR)/scripts/filter_physx.sh

# Stage 3 (Distillation): clone terrain teacher into proprioceptive student
# Usage: make train-sitting-standing-distill
SS_TEACHER := 20250414_170842_g1_deepmimic
SS_TEACHER_DIR := $(CURDIR)/data/checkpoints/$(SS_TEACHER)
train-sitting-standing-distill:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(CUDA) $(LD_LIB) $(WANDB) \
	cd videomimic_gym && \
	FILTER_PHYSX_LOG=$(CURDIR)/videomimic_gym/logs/train-sitting-standing-distill.console.log \
	$(FILTER_PHYSX) torchrun --nproc-per-node $(NUM_GPUS) legged_gym/scripts/train.py \
	--task=g1_deepmimic_root_heightfield_no_history_dagger --headless \
	--multi_gpu \
	--env.terrain.n_rows=1 --num_envs=$(NUM_ENVS) \
	--train.runner.max_iterations=5000 \
	--run_name g1_deepmimic_ss_distill \
	--wandb_note "sitting_standing_distill" \
	--env.deepmimic.human_motion_source=resources/data_config/sitting_standing_motion.yaml \
	--train.algorithm.learning_rate=1e-3 \
	--train.algorithm.schedule=fixed \
	--env.deepmimic.upsample_data=True \
	--env.deepmimic.use_human_videos=True \
	--env.deepmimic.use_amass=False \
	--env.deepmimic.link_pos_error_threshold=0.3 \
	--train.runner.save_interval=500 \
	--env.deepmimic.respawn_z_offset=0.1 \
	--env.terrain.cast_mesh_to_heightfield=False \
	--env.deepmimic.truncate_rollout_length=500 \
	--train.runner.load_model_strict=False \
	--train.algorithm.policy_to_clone=$(SS_TEACHER_DIR)

# Stage 4 (RL Finetuning): fine-tune the distilled student
# Usage: make train-sitting-standing-finetune LOAD_RUN=<stage3_run_name>
train-sitting-standing-finetune:
	$(CONDA_EVAL) && conda activate rlgpu && $(ISAAC_PRELOAD) $(CUDA) $(LD_LIB) $(WANDB) \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NUM_GPUS) legged_gym/scripts/train.py \
	--task=g1_deepmimic_root_heightfield_no_history_dagger --headless \
	--env.terrain.n_rows=1 --num_envs=$(NUM_ENVS) \
	--run_name g1_deepmimic_ss_finetune \
	--wandb_note "sitting_standing_finetune" \
	--env.deepmimic.human_motion_source=resources/data_config/sitting_standing_motion.yaml \
	--train.algorithm.learning_rate=3e-5 \
	--train.algorithm.bc_loss_coef=0.0 \
	--train.algorithm.schedule=fixed \
	--env.deepmimic.upsample_data=True \
	--env.deepmimic.use_human_videos=True \
	--env.deepmimic.use_amass=False \
	--env.deepmimic.amass_replay_data_path=lafan_walk_and_dance/*.pkl \
	--env.deepmimic.link_pos_error_threshold=0.5 \
	--train.runner.save_interval=500 \
	--env.noise.add_noise=True \
	--env.deepmimic.respawn_z_offset=0.1 \
	--env.rewards.scales.termination=-2000 \
	--env.rewards.scales.alive=200.0 \
	--train.runner.load_model_strict=False \
	--resume --load_run $(LOAD_RUN) \
	--train.policy.re_init_std=True \
	--train.policy.init_noise_std=0.5
