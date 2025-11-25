import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.utils.isaacgym_utils import parse_arguments_modified

from legged_gym.utils.configclass import configclass
from dataclasses import dataclass


from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import wandb
from typing import Optional, Dict, Any

def class_to_dict(obj) -> dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

# def get_load_path(root: str, load_run: str = '', checkpoint: int = -1) -> str:
#     """
#     Get the path to load a checkpoint from.
    
#     Args:
#         root: Root directory containing checkpoints
#         load_run: Name of the run to load from. If it's a wandb run ID, will download from wandb
#         checkpoint: Checkpoint number to load. -1 for latest checkpoint
    
#     Returns:
#         Path to the checkpoint file
#     """
#     # Check if this is a wandb run ID (they are typically alphanumeric)
#     if load_run and all(c.isalnum() or c == '-' for c in load_run):
#         print(f"Downloading checkpoint from wandb run ID: {load_run}")
        
#         # Initialize wandb API
#         api = wandb.Api()
        
#         try:
#             # Get the specific run using its ID
#             run = api.run(f"rsl_rl/{load_run}")
            
#             # Get all checkpoint files
#             files = run.files()
#             checkpoint_files = [f for f in files if 'model_' in f.name and f.name.endswith('.pt')]
            
#             if not checkpoint_files:
#                 raise ValueError(f"No checkpoints found for wandb run {load_run}")
            
#             # Get the latest checkpoint
#             latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('model_')[1].split('.')[0]))
            
#             # Create target directory using run name
#             target_dir = os.path.join(root, run.name)
#             os.makedirs(target_dir, exist_ok=True)
            
#             # Download the checkpoint
#             target_path = os.path.join(target_dir, latest_checkpoint.name)
#             print(f"Downloading checkpoint to {target_path}")
#             latest_checkpoint.download(root=target_dir, replace=True)
#             print("Download complete!")
            
#             return target_path
#         except Exception as e:
#             print(f"Error downloading from wandb: {e}")
#             raise
    
#     # Handle local checkpoints as before
#     if load_run == '':
#         load_run = os.listdir(root)[-1]
#     checkpoint_root = os.path.join(root, load_run)
#     if checkpoint == -1:
#         models = [file for file in os.listdir(checkpoint_root) if 'model_' in file]
#         models.sort(key=lambda m: '{0:0>15}'.format(m))
#         model = models[-1]
#         checkpoint = int(model.split('_')[-1].split('.')[0])
#     return os.path.join(checkpoint_root, f'model_{checkpoint}.pt')

def get_wandb_path(root: str, load_run: str, multi_gpu: bool = False, multi_gpu_rank: int = 0) -> str:
    """
    Get the path to a wandb checkpoint.
    
    Args:
        root: Root directory to save the checkpoint
        load_run: Either a wandb run ID or a name prefixed with 'wandb_'
    
    Returns:
        Path to the downloaded checkpoint file
    """
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
        target_dir = os.path.join(root, run.name, f'rank_{multi_gpu_rank}')
    else:
        target_dir = os.path.join(root, run.name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, latest_checkpoint.name)
    print(f"Downloading checkpoint to {target_path}")
    latest_checkpoint.download(root=target_dir, replace=True)
    print("Download complete!")
    
    return target_path

def get_load_path(root: str, load_run: str = '', checkpoint: int = -1, multi_gpu: bool = False, multi_gpu_rank: int = 0) -> str:
    """
    Get the path to load a checkpoint from.
    
    Args:
        root: Root directory containing checkpoints
        load_run: Name of the run to load from. Can be:
                 - A wandb run ID (alphanumeric with hyphens)
                 - A run name prefixed with 'wandb_'
                 - A local run name
        checkpoint: Checkpoint number to load. -1 for latest checkpoint
    
    Returns:
        Path to the checkpoint file
    """
    # Try to get wandb path first
    wandb_path = get_wandb_path(root, load_run, multi_gpu=multi_gpu, multi_gpu_rank=multi_gpu_rank)
    if wandb_path is not None:
        return wandb_path
    
    # Handle local checkpoints
    if load_run == '':
        load_run = os.listdir(root)[-1]
    checkpoint_root = os.path.join(root, load_run)
    if checkpoint == -1:
        models = [file for file in os.listdir(checkpoint_root) if 'model_' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = int(model.split('_')[-1].split('.')[0])
    return os.path.join(checkpoint_root, f'model_{checkpoint}.pt')

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.multi_gpu: # only call this in env config one since it gets called first and we don't want to increment the seed twice
            # extract rank that torch.distributed provides
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            global_rank = int(os.getenv("RANK", "0"))
            print(f"Horovod global rank {global_rank} local rank: {local_rank}")
            args.sim_device = f'cuda:{local_rank}'
            args.rl_device = f'cuda:{local_rank}'
            if args.seed:
                # need a different seed for each env so that scaling works properly :)
                args.seed += global_rank
            
            if args.multi_gpu and hasattr(env_cfg.asset, 'alt_files') and env_cfg.asset.use_alt_files:
                all_files = [env_cfg.asset.file] + env_cfg.asset.alt_files
                file_idx = global_rank % len(all_files)
                env_cfg.asset.file = all_files[file_idx]
                print(f'swapping asset file for multi-gpu rank {global_rank} to {env_cfg.asset.file}')
            if args.multi_gpu and hasattr(env_cfg.terrain, 'alternate_cast_to_heightfield') and env_cfg.terrain.alternate_cast_to_heightfield:
                if global_rank % 2 == 1:
                    env_cfg.terrain.cast_mesh_to_heightfield = True
                print(f'setting cast_mesh_to_heightfield to {env_cfg.terrain.cast_mesh_to_heightfield} for multi-gpu rank {global_rank}')

    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def parse_unknown_args(unknown):
    env_overrides = {}
    train_overrides = {}
    
    for arg in unknown:
        if not arg.startswith('--'):
            continue
        arg = arg[2:] # Remove leading --
        key, value = arg.split('=')
        
        if key.startswith('env.'):
            # Remove env. prefix and store in env_overrides
            env_key = key[4:]
            try:
                env_overrides[env_key] = float(value) # Try converting to float
            except ValueError:
                env_overrides[env_key] = value # Keep as string if not a number
                
        elif key.startswith('train.'):
            # Remove train. prefix and store in train_overrides  
            train_key = key[6:]
            try:
                train_overrides[train_key] = float(value)
            except ValueError:
                train_overrides[train_key] = value

    return env_overrides, train_overrides

def update_cfg_from_overrides(cfg, overrides: dict):
    """
    Updates the config with the overrides.
    Args:
        cfg (LeggedRobotCfg): The config to update.
        overrides (dict): The overrides to apply. Of the form (eg.) {'deepmimic.foo': 1.0, 'deepmimic.bar': 2.0}
    """
    def _update_recursive(curr_cfg, key_parts, value):
        if len(key_parts) == 1:
            if hasattr(curr_cfg, key_parts[0]):
                curr_value = getattr(curr_cfg, key_parts[0])
                # Cast value to the same type as the current value
                try:
                    if isinstance(curr_value, bool):
                        if isinstance(value, str):
                            typed_value = value.lower() == 'true'
                        else:
                            typed_value = bool(value)
                    else:
                        typed_value = type(curr_value)(value)
                    setattr(curr_cfg, key_parts[0], typed_value)
                    return True, ""
                except (ValueError, TypeError):
                    return False, f"Could not convert {value} to type {type(curr_value)} for {key_parts[0]}"
            return False, f"Key {key_parts[0]} not found in config"
        
        if hasattr(curr_cfg, key_parts[0]):
            next_cfg = getattr(curr_cfg, key_parts[0])
            return _update_recursive(next_cfg, key_parts[1:], value)
        return False, f"More than one equals sign in key {key_parts}"

    for key, value in overrides.items():
        key_parts = key.split('.')
        success, reason = _update_recursive(cfg, key_parts, value)
        if not success:
            raise ValueError(f"Key {key} to override not able to be updated in config: {reason}")

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--no_use_wandb", "action": "store_true", "default": True, "help": "Disable using wandb for logging."},
        {"name": "--wandb_note", "type": str, "help": "Note to add to wandb run."},
        {"name": "--multi_gpu", "action": "store_true", "default": False, "help": "Whether to enable multi-gpu training", },


    ]
    # parse arguments
    args, unknown = parse_arguments_modified(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args, unknown