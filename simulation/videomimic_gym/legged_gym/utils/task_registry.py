import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, update_cfg_from_overrides
from .configclass import configclass



class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    @property
    def registry(self):
        return self.task_classes

    def register(self, name: str, task_class: VecEnv, env_cfg: configclass, train_cfg: configclass):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name, env_overrides=None, train_overrides=None) -> Tuple[configclass, configclass]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        if env_overrides is not None:
            update_cfg_from_overrides(env_cfg, env_overrides)
        if train_overrides is not None:
            update_cfg_from_overrides(train_cfg, train_overrides)
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None, env_overrides=None) -> Tuple[VecEnv, configclass]:
        """Creates an environment using IsaacLab's DirectRLEnv interface.

        Args:
            name: Name of a registered env.
            args: Command line arguments. If None get_args() will be called.
            env_cfg: Environment config to override the registered config.
            env_overrides: Dict of overrides for the env config.

        Returns:
            env: The created environment
            env_cfg: the corresponding config file
        """
        if args is None:
            args = get_args()
        # Trigger lazy registration if env classes haven't been imported yet
        if name not in self.task_classes:
            from legged_gym.envs import _ensure_registered
            _ensure_registered()
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        if env_overrides is not None:
            update_cfg_from_overrides(env_cfg, env_overrides)
            print(env_overrides)
        print(env_cfg)
        set_seed(env_cfg.seed)

        # Determine render mode from args
        render_mode = None
        if args.headless:
            render_mode = None  # DirectRLEnv handles headless via enable_rendering=False in cfg
        else:
            render_mode = "rgb_array"

        # Create env using IsaacLab-style constructor
        env = task_class(cfg=env_cfg, render_mode=render_mode)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, train_overrides=None, log_root="default") -> Tuple[OnPolicyRunner, configclass]:
        """Creates the training algorithm runner.

        Args:
            env: The environment to train.
            name: Name of a registered env.
            args: Command line arguments.
            train_cfg: Training config file.
            train_overrides: Dict of overrides for the train config.
            log_root: Logging directory root.

        Returns:
            runner: The created OnPolicyRunner
            train_cfg: the corresponding config file
        """
        if args is None:
            args = get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        train_cfg.runner.use_wandb = not args.no_use_wandb
        train_cfg.runner.wandb_note = args.wandb_note
        if train_overrides is not None:
            update_cfg_from_overrides(train_cfg, train_overrides)

        total_run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        if train_cfg.runner.run_name:
            total_run_name += '_' + train_cfg.runner.run_name
        else:
            total_run_name += '_' + train_cfg.runner.experiment_name
        train_cfg.runner.run_name = total_run_name

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, total_run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, total_run_name)

        if hasattr(train_cfg.algorithm, 'use_multi_teacher') and train_cfg.algorithm.use_multi_teacher:
            train_cfg.algorithm.policy_to_clone = env.teacher_checkpoints

        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device, multi_gpu=args.multi_gpu)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint, multi_gpu=args.multi_gpu, multi_gpu_rank=os.getenv("RANK", "0"))
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path, load_optimizer=False)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
