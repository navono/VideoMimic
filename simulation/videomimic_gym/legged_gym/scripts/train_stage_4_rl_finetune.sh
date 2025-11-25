#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

LOAD_RUN=$1 # start from stage 3 run name
LOAD_RUN=20250502_124756_g1_deepmimic


torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
--multi_gpu \
--task=g1_deepmimic_root_heightfield_no_history_dagger --headless --env.terrain.n_rows=1 --num_envs=4096 \
--wandb_note "stage_4_rl_finetune" --env.deepmimic.human_motion_source=resources/data_config/human_motion_list_123_motions.yaml --train.algorithm.learning_rate=3e-5  --train.algorithm.bc_loss_coef=0.0 --train.algorithm.schedule=fixed legged_gym/scripts/train.py --headless --env.deepmimic.amass_terrain_difficulty=1 --env.deepmimic.upsample_data=True --env.deepmimic.use_human_videos=True --env.deepmimic.link_pos_error_threshold=0.5 --train.runner.save_interval=500 --env.noise.add_noise=True --env.deepmimic.respawn_z_offset=0.1 --train.runner.load_model_strict=False \
--env.deepmimic.use_amass=True --env.deepmimic.amass_replay_data_path=lafan_walk/*.pkl \
--resume --load_run ${LOAD_RUN} --train.policy.re_init_std=True --train.policy.init_noise_std=0.5 \
--env.rewards.scales.termination=-2000 --env.rewards.scales.alive=200.0