# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import os

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)

def get_wandb_path(load_run: str, multi_gpu: bool = False, multi_gpu_rank: int = 0) -> str:
    """
    Get the path to a wandb checkpoint.
    Saves in a temp folder in /tmp
    
    Args:
        load_run: Either a wandb run ID or a name prefixed with 'wandb_'
    
    Returns:
        Path to the downloaded checkpoint file
    """
    import wandb
    api = wandb.Api()
    run_id = None
    
    # Handle run ID
    # if all(c.isalnum() or c == '-' for c in load_run):
        # run_id = load_run
    if load_run.startswith('wandb_id_'):
        run_id = load_run[9:]
    # Handle run name
    elif load_run.startswith('wandb_'):
        run_name = load_run[6:]  # Remove 'wandb_' prefix
        runs = list(api.runs("rsl_rl", filters={"display_name": run_name}))
        if not runs:
            raise ValueError(f"No wandb run found with name: {run_name}")
        if len(runs) > 1:
            print(f"Warning: Multiple runs found with name {run_name}, using the most recent one")
        run_id = runs[0].id
    else:
        return None
    
    print(f"Downloading checkpoint from wandb run ID: {run_id}")
    
    # Get the run and its checkpoints
    run = api.run(f"rsl_rl/{run_id}")
    files = run.files()
    checkpoint_files = [f for f in files if 'model_' in f.name and f.name.endswith('.pt')]
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found for wandb run {run_id}")
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('model_')[1].split('.')[0]))
    
    # Create target directory and download
    if multi_gpu:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name, f'rank_{multi_gpu_rank}')
    else:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, latest_checkpoint.name)
    print(f"Downloading checkpoint to {target_path}")
    latest_checkpoint.download(root=target_dir, replace=True)
    print("Download complete!")
    
    return target_path

def get_checkpoint_path(log_dir_or_checkpoint_path, checkpoint=-1, multi_gpu=False, multi_gpu_rank=0):
    """Get the latest checkpoint from the log directory"""

    wandb_path = get_wandb_path(log_dir_or_checkpoint_path, multi_gpu=multi_gpu, multi_gpu_rank=multi_gpu_rank)
    if wandb_path is not None:
        return wandb_path

    if not os.path.isdir(log_dir_or_checkpoint_path):
        return log_dir_or_checkpoint_path

    if checkpoint==-1:
        models = [file for file in os.listdir(log_dir_or_checkpoint_path) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    return os.path.join(log_dir_or_checkpoint_path, model)