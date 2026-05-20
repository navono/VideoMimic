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
        play-terrain play-flat play-mcpt

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
