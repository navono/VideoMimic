# VideoMimic Simulation

## Setup

See [setup guide](setup.md) for how to install `videomimic_gym` and `videomimic_rl`.

### Download data

We can download the checkpoints and the video datasets with

```
cd data
bash download_videomimic_data.sh
```

### Running inference

In all the below inference scripts, after running the command a viser UI will be available at localhost:8080 to see the robot playing. If you are on a remote server, you can either forward this through SSH tunnels, VS Code's ports interface, or persistent cloudflare tunnels.

Ensure you have your `videomimic` conda env activated, and run:

*For terrain policy*

```bash
bash videomimic_gym/legged_gym/scripts/play_terrain_policy.sh
```

(note that this loads one motion, you can also pass in `resources/data_config/human_motion_list_123_motions.yaml` to run all of them, just note that viser can take longer to load all the meshes).


*For distilled flat policy (for comparison) -- takes root direction but not the reference.*

```bash 
bash videomimic_gym/legged_gym/scripts/play_flat_policy.sh
```


*MCPT (phase 1) policy inference -- takes reference joints and root direction.* 
```bash
bash videomimic_gym/legged_gym/scripts/play_mcpt_policy.sh
```

Note that all of the above checkpoints have been tested on a real unitree G1.

### Running training 

Again, ensure you have the videomimic conda environment activated for all the below.

#### MoCap Pre-training

```bash
bash videomimic_gym/legged_gym/scripts/train_stage_1_mcpt.sh
```

#### Stage 2 (tracking over terrain)

```bash
bash videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh
```

#### Stage 3 (Distillation)

```bash
LOAD_RUN=stage_2_run_name # replace with output of stage 2
bash videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh ${LOAD_RUN}
```

#### Stage 4 (RL Finetuning)

```bash
LOAD_RUN=stage_3_run_name # replace with output of stage 3
bashv ideomimic_gym/legged_gym/scripts/train_stage_4_rl_finetune.sh ${LOAD_RUN}
```


#### Other notes

Adjust nproc-per-node to be however many GPUs you have available. We also natively support multi-node training if needed.

Note that not all the above stages have been tested end2end pre-release. Results may also vary depending on the data. Please let me (Arthur) know about any specific issues you run into, happy to help.

There is more details on changing parameters of train scripts in [the corresponding readme for videomimic_gym](videomimic_gym/README.md).