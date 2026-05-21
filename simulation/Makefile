SHELL := /bin/bash

# --------------- Configurable variables ---------------
NUM_GPUS     ?= 2
NUM_ENVS     ?= 4096
LOAD_RUN     ?=
MOTION_CFG   ?= resources/data_config/human_motion_list_124_motions.yaml
HEADLESS     ?=

# --------------- Conda helpers ---------------
CONDA_EVAL := eval "$$(conda shell.bash hook)"

# Isaac Gym needs libpython shared library on LD_LIBRARY_PATH
LD_LIB := export LD_LIBRARY_PATH=$$CONDA_PREFIX/lib:$$LD_LIBRARY_PATH &&

# Skip wandb login requirement
WANDB := export WANDB_MODE=disabled &&

# --------------- Paths ---------------
SCRIPTS_DIR := videomimic_gym/legged_gym/scripts

# --------------- Targets ---------------

.PHONY: train-stage1 train-stage2 train-stage3 train-stage4 \
        play-terrain play-flat play-mcpt play-sitting-standing

# ---------- Training (4 stages, sequential) ----------

# Stage 1: MoCap Pre-training
train-stage1:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/train_stage_1_mcpt.sh

# Stage 2: Terrain RL
# Usage:
#   H20 (4 cards):   make train-stage2 NUM_GPUS=4
#   Current machine:  make train-stage2 NUM_GPUS=1 NUM_ENVS=2048
#   Custom data:      make train-stage2 NUM_GPUS=4 MOTION_CFG=resources/data_config/human_motion_list_123_motions.yaml
train-stage2:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
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
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
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
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
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
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/play_terrain_policy.sh

play-flat:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/play_flat_policy.sh

play-mcpt:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/play_mcpt_policy.sh

# Play with sitting_standing video data
# Usage: make play-sitting-standing NUM_EPISODES=20
play-sitting-standing:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	python $(SCRIPTS_DIR)/play.py --task=g1_deepmimic_root_heightfield_no_history_dagger \
	--load_run 20250502_124756_g1_deepmimic_distill --num_envs 1 \
	--env.deepmimic.use_amass=False --env.deepmimic.use_human_videos=True \
	--env.deepmimic.human_motion_source=resources/data_config/sitting_standing_motion.yaml \
	--env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.link_pos_error_threshold=10.0 \
	--train.algorithm.bc_loss_coef=0.0 --env.terrain.cast_mesh_to_heightfield=False \
	--train.runner.load_model_strict=True --env.viser.enable=True \
	--run_name g1_deepmimic_sitting_standing \
	--max_episodes $(or $(NUM_EPISODES),10)
