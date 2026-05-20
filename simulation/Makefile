SHELL := /bin/bash

# --------------- Configurable variables ---------------
NUM_GPUS     ?= 1
LOAD_RUN     ?=
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
train-stage2:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	bash $(SCRIPTS_DIR)/train_stage_2_terrain_rl.sh

# Stage 3: Distillation (requires LOAD_RUN from stage 2)
train-stage3:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	LOAD_RUN=$(LOAD_RUN) bash $(SCRIPTS_DIR)/train_stage_3_distillation.sh $(LOAD_RUN)

# Stage 4: RL Finetuning (requires LOAD_RUN from stage 3)
train-stage4:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	LOAD_RUN=$(LOAD_RUN) bash $(SCRIPTS_DIR)/train_stage_4_rl_finetune.sh $(LOAD_RUN)

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
play-sitting-standing:
	$(CONDA_EVAL) && conda activate rlgpu_ig && $(LD_LIB) $(WANDB) \
	python $(SCRIPTS_DIR)/play.py --task=g1_deepmimic_root_heightfield_no_history_dagger \
	--load_run 20250502_124756_g1_deepmimic_distill --num_envs 1 \
	--env.deepmimic.use_amass=False --env.deepmimic.use_human_videos=True \
	--env.deepmimic.human_motion_source=resources/data_config/sitting_standing_motion.yaml \
	--env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.link_pos_error_threshold=10.0 \
	--train.algorithm.bc_loss_coef=0.0 --env.terrain.cast_mesh_to_heightfield=False \
	--train.runner.load_model_strict=True --env.viser.enable=True \
	--run_name g1_deepmimic_sitting_standing
