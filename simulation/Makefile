# VideoMimic Simulation Makefile
# Usage:
#   make play-terrain              # Inference: terrain policy (4 motions)
#   make play-flat                 # Inference: flat policy (MoCap)
#   make play-mcpt                 # Inference: MCPT policy (reference joints)
#   make play-jit-terrain          # Inference: sim2real JIT terrain/heightmap policy
#   make play-jit-flat             # Inference: sim2real JIT flat/distilled policy
#   make play-terrain-physics      # Inference: terrain policy with kinematic replay disabled
#   make play-jit-terrain-physics  # Inference: JIT terrain policy with kinematic replay disabled
#   make play-terrain-zero         # Diagnostic: terrain env with zero actions
#   make train-s1                  # Training: Stage 1 MoCap pre-training
#   make train-s2                  # Training: Stage 2 terrain RL
#   make train-s3 LOAD_RUN=xxx     # Training: Stage 3 distillation
#   make train-s4 LOAD_RUN=xxx     # Training: Stage 4 RL finetuning

SHELL := /bin/bash

# --- Configurable variables ---
NGPU       ?= 1
NUM_ENVS   ?= 2048
JIT_DIR    ?= ../../simulation_data/checkpoints_jit

# --- Conda helpers ---
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)" && conda activate rlgpu

# ============================================================
# Inference / Play
# ============================================================

.PHONY: play-terrain play-flat play-mcpt play-jit-terrain play-jit-flat play-terrain-physics play-jit-terrain-physics play-terrain-zero

## Play terrain policy (4 motions with viser visualization)
play-terrain:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --env.deepmimic.use_amass=False \
	    --load_run 20250502_124756_g1_deepmimic \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=True \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.n_append=0 \
	    --env.deepmimic.human_motion_source=resources/data_config/4_motion.yaml \
	    --train.runner.load_model_strict=True \
	    --env.viser.enable=True

## Play flat/distilled policy (MoCap data, no reference)
play-flat:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_dagger \
	    --env.deepmimic.use_amass=True \
	    --env.deepmimic.amass_terrain_difficulty=1 \
	    --env.deepmimic.amass_replay_data_path="lafan_single_walk/*.pkl" \
	    --load_run 20250414_170842_g1_deepmimic \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=False \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.viser.enable=True

## Play MCPT policy (reference joints + root direction)
play-mcpt:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --env.deepmimic.use_amass=False \
	    --load_run 20250502_124756_g1_deepmimic \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=True \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.n_append=0 \
	    --env.deepmimic.human_motion_source=resources/data_config/1_motion.yaml \
	    --train.runner.load_model_strict=True \
	    --env.viser.enable=True

## Play sim2real JIT terrain/heightmap policy directly
play-jit-terrain:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --jit_policy_path $(JIT_DIR)/20250502_124756_g1_deepmimic_dict.pt \
	    --env.deepmimic.use_amass=False \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=True \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.n_append=0 \
	    --env.deepmimic.human_motion_source=resources/data_config/4_motion.yaml \
	    --env.viser.enable=True

## Play sim2real JIT flat/distilled policy directly
play-jit-flat:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_dagger \
	    --jit_policy_path $(JIT_DIR)/20250414_170842_g1_deepmimic_dict.pt \
	    --env.deepmimic.use_amass=True \
	    --env.deepmimic.amass_terrain_difficulty=1 \
	    --env.deepmimic.amass_replay_data_path="lafan_single_walk/*.pkl" \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=False \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.viser.enable=True

## Play terrain policy with kinematic replay disabled for policy/physics validation
play-terrain-physics:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --env.deepmimic.use_amass=False \
	    --load_run 20250502_124756_g1_deepmimic \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=True \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.n_append=0 \
	    --env.deepmimic.human_motion_source=resources/data_config/4_motion.yaml \
	    --train.runner.load_model_strict=True \
	    --env.viser.enable=True \
	    --env.deepmimic.viz_replay_sync_robot=False

## Play sim2real JIT terrain policy with kinematic replay disabled
play-jit-terrain-physics:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --jit_policy_path $(JIT_DIR)/20250502_124756_g1_deepmimic_dict.pt \
	    --env.deepmimic.use_amass=False \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=True \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.n_append=0 \
	    --env.deepmimic.human_motion_source=resources/data_config/4_motion.yaml \
	    --env.viser.enable=True \
	    --env.deepmimic.viz_replay_sync_robot=False

## Diagnostic: terrain env with kinematic replay disabled and zero actions
play-terrain-zero:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	python legged_gym/scripts/play.py \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --zero_policy \
	    --env.deepmimic.use_amass=False \
	    --num_envs 1 \
	    --env.deepmimic.use_human_videos=True \
	    --headless \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.link_pos_error_threshold=10.0 \
	    --train.algorithm.bc_loss_coef=0.0 \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.n_append=0 \
	    --env.deepmimic.human_motion_source=resources/data_config/4_motion.yaml \
	    --env.viser.enable=True \
	    --env.deepmimic.viz_replay_sync_robot=False

# ============================================================
# Training
# ============================================================

.PHONY: train-s1 train-s2 train-s3 train-s4

## Stage 1: MoCap Pre-training
train-s1:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NGPU) legged_gym/scripts/train.py \
	    --env.deepmimic.use_amass=True --multi_gpu --task=g1_deepmimic --headless \
	    --env.terrain.n_rows=16 \
	    --num_envs=$(NUM_ENVS) \
	    --wandb_note "videomimic_stage_1" \
	    --env.deepmimic.truncate_rollout_length=500 \
	    --env.noise.add_noise=True \
	    --env.deepmimic.link_pos_error_threshold=0.5 \
	    --env.rewards.scales.action_rate=-25.0 \
	    --env.deepmimic.amass_terrain_difficulty=2 \
	    --env.domain_rand.p_gain_rand=True \
	    --env.domain_rand.d_gain_rand=True \
	    --env.domain_rand.push_robots=True \
	    --env.domain_rand.control_delays=True \
	    --env.domain_rand.control_delay_min=0 \
	    --env.domain_rand.control_delay_max=5 \
	    --env.noise.offset_scales.gravity=0.02 \
	    --env.noise.offset_scales.dof_pos=0.005 \
	    --env.deepmimic.randomize_terrain_offset=True \
	    --env.asset.use_alt_files=True \
	    --env.noise.init_noise_scales.root_xy=0.1 \
	    --env.noise.init_noise_scales.root_z=0.02 \
	    --env.domain_rand.randomize_base_mass=True \
	    --env.asset.terminate_after_large_feet_contact_forces=True \
	    --env.noise.init_noise_scales.dof_pos=0.01

## Stage 2: Terrain RL tracking
train-s2:
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NGPU) legged_gym/scripts/train.py \
	    --task=g1_deepmimic_proj_heightfield \
	    --multi_gpu --headless \
	    --env.terrain.n_rows=1 \
	    --num_envs=$(NUM_ENVS) \
	    --wandb_note "videomimic_stage_2" \
	    --env.deepmimic.human_motion_source=resources/data_config/human_motion_list_123_motions.yaml \
	    --resume --train.policy.re_init_std=True --train.policy.init_noise_std=0.5 \
	    --train.algorithm.learning_rate=2e-5 --train.algorithm.schedule=fixed \
	    --headless --env.deepmimic.amass_terrain_difficulty=1 \
	    --env.deepmimic.upsample_data=True --env.deepmimic.use_human_videos=True \
	    --train.algorithm.learning_rate=2e-5 \
	    --env.deepmimic.link_pos_error_threshold=0.5 \
	    --train.runner.save_interval=500 \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.deepmimic.randomize_terrain_offset=False \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.truncate_rollout_length=500 \
	    --train.runner.load_model_strict=False \
	    --env.deepmimic.use_amass=False \
	    --env.rewards.scales.termination=-2000 --env.rewards.scales.alive=200.0 \
	    --env.rewards.scales.ankle_action=-3.0 --env.rewards.scales.action_rate=-3.0

## Stage 3: Distillation (requires LOAD_RUN from stage 2)
train-s3:
	@if [ -z "$(LOAD_RUN)" ]; then echo "Usage: make train-s3 LOAD_RUN=stage2_run_name"; exit 1; fi
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NGPU) legged_gym/scripts/train.py \
	    --multi_gpu --headless \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --env.terrain.n_rows=1 \
	    --num_envs=$(NUM_ENVS) \
	    --wandb_note "videomimic_stage_3" \
	    --env.deepmimic.human_motion_source=resources/data_config/human_motion_list_123_motions.yaml \
	    --train.algorithm.learning_rate=1e-3 --train.algorithm.schedule=fixed \
	    --env.deepmimic.upsample_data=True --env.deepmimic.use_human_videos=True \
	    --env.deepmimic.link_pos_error_threshold=0.3 \
	    --train.runner.save_interval=500 \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --env.terrain.cast_mesh_to_heightfield=False \
	    --env.deepmimic.truncate_rollout_length=500 \
	    --train.runner.load_model_strict=False \
	    --env.deepmimic.use_amass=False \
	    --train.algorithm.policy_to_clone=$(LOAD_RUN)

## Stage 4: RL Finetuning (requires LOAD_RUN from stage 3)
train-s4:
	@if [ -z "$(LOAD_RUN)" ]; then echo "Usage: make train-s4 LOAD_RUN=stage3_run_name"; exit 1; fi
	@$(CONDA_ACTIVATE) && \
	cd videomimic_gym && \
	torchrun --nproc-per-node $(NGPU) legged_gym/scripts/train.py \
	    --multi_gpu --headless \
	    --task=g1_deepmimic_root_heightfield_no_history_dagger \
	    --env.terrain.n_rows=1 \
	    --num_envs=$(NUM_ENVS) \
	    --wandb_note "stage_4_rl_finetune" \
	    --env.deepmimic.human_motion_source=resources/data_config/human_motion_list_123_motions.yaml \
	    --train.algorithm.learning_rate=3e-5 --train.algorithm.bc_loss_coef=0.0 \
	    --train.algorithm.schedule=fixed \
	    --headless --env.deepmimic.amass_terrain_difficulty=1 \
	    --env.deepmimic.upsample_data=True --env.deepmimic.use_human_videos=True \
	    --env.deepmimic.link_pos_error_threshold=0.5 \
	    --train.runner.save_interval=500 \
	    --env.noise.add_noise=True \
	    --env.deepmimic.respawn_z_offset=0.1 \
	    --train.runner.load_model_strict=False \
	    --env.deepmimic.use_amass=True \
	    --env.deepmimic.amass_replay_data_path=lafan_walk/*.pkl \
	    --resume --load_run $(LOAD_RUN) \
	    --train.policy.re_init_std=True --train.policy.init_noise_std=0.5 \
	    --env.rewards.scales.termination=-2000 --env.rewards.scales.alive=200.0

# ============================================================
# Help
# ============================================================

.PHONY: help

help:
	@echo "VideoMimic Simulation Makefile"
	@echo ""
	@echo "Inference:"
	@echo "  make play-terrain         Play terrain policy (4 motions)"
	@echo "  make play-flat            Play flat/distilled policy"
	@echo "  make play-mcpt            Play MCPT policy (single motion)"
	@echo "  make play-jit-terrain     Play sim2real JIT terrain/heightmap policy"
	@echo "  make play-jit-flat        Play sim2real JIT flat/distilled policy"
	@echo "  make play-terrain-physics Play terrain policy with kinematic replay disabled"
	@echo "  make play-jit-terrain-physics Play JIT terrain policy with kinematic replay disabled"
	@echo "  make play-terrain-zero    Diagnostic terrain env with zero actions"
	@echo ""
	@echo "Training (single GPU by default):"
	@echo "  make train-s1             Stage 1: MoCap pre-training"
	@echo "  make train-s2             Stage 2: Terrain RL"
	@echo "  make train-s3 LOAD_RUN=x  Stage 3: Distillation"
	@echo "  make train-s4 LOAD_RUN=x  Stage 4: RL finetuning"
	@echo ""
	@echo "  Optional: NGPU=2 NUM_ENVS=4096"
