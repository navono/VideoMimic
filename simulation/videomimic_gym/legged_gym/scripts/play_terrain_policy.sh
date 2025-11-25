#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/../.."

python legged_gym/scripts/play.py --task=g1_deepmimic_root_heightfield_no_history_dagger --env.deepmimic.use_amass=False --load_run 20250502_124756_g1_deepmimic --num_envs 1 --env.deepmimic.use_human_videos=True --headless  --env.deepmimic.respawn_z_offset=0.1 --env.deepmimic.link_pos_error_threshold=10.0 --train.algorithm.bc_loss_coef=0.0 --env.terrain.cast_mesh_to_heightfield=False --env.deepmimic.n_append=0 --env.deepmimic.human_motion_source=resources/data_config/4_motion.yaml --train.algorithm.bc_loss_coef=0.0 --train.runner.load_model_strict=True  --env.viser.enable=True