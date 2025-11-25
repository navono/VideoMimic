#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

LOAD_RUN=20250410_063030_g1_deepmimic # start from MPT

torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
--task=g1_deepmimic_proj_heightfield \
--multi_gpu \
--headless \
--env.terrain.n_rows=1 \
--num_envs=4096 \
--wandb_note "videomimic_stage_2" \
--env.deepmimic.human_motion_source=resources/data_config/human_motion_list_123_motions.yaml \
--load_run ${LOAD_RUN} --resume --train.policy.re_init_std=True --train.policy.init_noise_std=0.5 --train.algorithm.learning_rate=2e-5 --train.algorithm.schedule=fixed \
legged_gym/scripts/train.py  --headless --env.deepmimic.amass_terrain_difficulty=1 --env.deepmimic.upsample_data=True --env.deepmimic.use_human_videos=True --train.algorithm.learning_rate=2e-5 --env.deepmimic.link_pos_error_threshold=0.5 --train.runner.save_interval=500 --env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.randomize_terrain_offset=False --env.terrain.cast_mesh_to_heightfield=False --env.deepmimic.truncate_rollout_length=500 --train.runner.load_model_strict=False  \
--env.deepmimic.use_amass=False \
--env.rewards.scales.termination=-2000 --env.rewards.scales.alive=200.0 \
--env.rewards.scales.ankle_action=-3.0 --env.rewards.scales.action_rate=-3.0