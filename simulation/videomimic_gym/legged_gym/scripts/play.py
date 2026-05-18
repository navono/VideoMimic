import os
import sys
import torch
import numpy as np

# Parse args first — must happen before SimulationApp so --headless is available
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import parse_unknown_args, get_load_path, class_to_dict

args, unknown = get_args()

# Bootstrap Isaac Sim Kit runtime before any isaaclab imports
from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless})

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from rsl_rl.utils.jit import export_policy_as_jit


class PlayManager:
    def __init__(self, args, unknown):
        self.args = args
        self.unknown = unknown
        self.env_overrides, self.train_overrides = parse_unknown_args(unknown)
        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=args.task)
        self.env_cfg.viser.enabled = True
        self.curr_task = args.task

        # Adjust environment configuration based on task
        self._configure_env(args.task)

        # Create the environment
        self.env, _ = task_registry.make_env(
            name=args.task, args=args, env_cfg=self.env_cfg, env_overrides=self.env_overrides
        )

        # Set some additional config flags
        args.use_wandb = False
        self.train_cfg.runner.resume = True

        # ---- Joint-order debug ----
        try:
            lab_names = list(self.env.dof_names)
        except Exception:
            lab_names = []
        cfg_keys = list(self.env_cfg.init_state.default_joint_angles.keys())
        print("[joint-order] IsaacLab articulation order ({}):".format(len(lab_names)))
        for i, n in enumerate(lab_names):
            print(f"  lab[{i:2d}] = {n}")
        print("[joint-order] cfg.init_state.default_joint_angles keys ({}):".format(len(cfg_keys)))
        for i, n in enumerate(cfg_keys):
            print(f"  cfg[{i:2d}] = {n}")
        # ---- end debug ----

        # Initialize PPO runner and policy
        self.ppo_runner, self.train_cfg = task_registry.make_alg_runner(
            env=self.env, name=args.task, args=args, train_cfg=self.train_cfg, train_overrides=self.train_overrides
        )
        self.policy = self.ppo_runner.get_inference_policy(device=self.env.device)

        # Simulation step counter
        self.t = 0

        # Set up callbacks if visualization is enabled
        if hasattr(self.env, 'viser_viz'):
            self.log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.train_cfg.runner.experiment_name)
            self.env.viser_viz.setup_checkpoint_selection(self.log_root, self.on_checkpoint_selected)
            self.env.viser_viz.export_button.on_click(self.on_export_clicked)

    def _configure_env(self, task):
        """Configure environment parameters based on the task."""
        if 'deepmimic' in task:
            self.env_cfg.deepmimic.viz_replay = True

            if self.env_cfg.deepmimic.viz_replay_sync_robot:
                self.env_cfg.control.stiffness = {k: 0.0 for k in self.env_cfg.control.stiffness}
                self.env_cfg.control.damping = {k: 0.0 for k in self.env_cfg.control.damping}

            self.env_cfg.terrain.num_rows = 5
            self.env_cfg.terrain.num_cols = 5
            self.env_cfg.terrain.curriculum = False

            self.env_cfg.deepmimic.randomize_start_offset = False
            self.env_cfg.deepmimic.init_velocities = True

        self.env_cfg.terrain.n_rows = 1
        self.env_cfg.noise.add_noise = False
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.domain_rand.push_robots = False
        self.env_cfg.env.test = True

    def on_checkpoint_selected(self, checkpoint_dir):
        """Callback for reloading checkpoints when selected from viser."""
        print(f"Loading checkpoint from directory: {checkpoint_dir}")
        load_path = get_load_path(self.log_root, load_run=checkpoint_dir)
        print(f"Loading model from: {load_path}")

        try:
            self.ppo_runner.load(load_path)
        except Exception as e:
            print(f'Loading failed, recreating runner assuming we need to switch task: {e}')
            checkpoint_dict = torch.load(load_path)
            if 'policy_cfg' in checkpoint_dict:
                self.train_cfg.policy = checkpoint_dict['policy_cfg']
                train_cfg_dict = class_to_dict(self.train_cfg)
            else:
                from legged_gym.envs import G1DeepMimicCfgPPO, G1DeepMimicCfgDagger
                if self.curr_task == 'g1_deepmimic_dagger':
                    self.train_cfg.policy = G1DeepMimicCfgPPO().policy
                    train_cfg_dict = class_to_dict(self.train_cfg)
                    self.curr_task = 'g1_deepmimic'
                elif self.curr_task == 'g1_deepmimic':
                    self.train_cfg.policy = G1DeepMimicCfgDagger().policy
                    train_cfg_dict = class_to_dict(self.train_cfg)
                    self.curr_task = 'g1_deepmimic_dagger'
                else:
                    raise ValueError(f'Unknown task: {self.curr_task}')

                from rsl_rl.runners import OnPolicyRunner
                self.ppo_runner = OnPolicyRunner(
                    self.env, train_cfg_dict, None, device=self.args.rl_device, multi_gpu=self.args.multi_gpu
                )
                self.ppo_runner.load(load_path)

        self.policy = self.ppo_runner.get_inference_policy(device=self.env.device)

    def on_export_clicked(self, _):
        """Callback for exporting the policy as a jit module when requested from viser."""
        export_path = self.env.viser_viz.export_path.value
        try:
            export_policy_as_jit(self.ppo_runner.alg.actor_critic, export_path)
            print(f'Successfully exported policy to: {export_path}')
        except Exception as e:
            print(f'Error exporting policy: {e}')
            raise e

    def step_simulation(self):
        """Perform one simulation step."""
        obs = self.env.get_observations()
        actions = self.policy({k: v.detach() for k, v in obs.items()}, monitor_activations=False)
        obs, rews, dones, infos = self.env.step(actions.detach())

        self.t += 1
        if self.t % 50 == 0:
            crv = getattr(self.env, 'current_reward_value', {}) or {}
            def _g(k):
                v = crv.get(k)
                return float(v[0]) if v is not None else float('nan')
            print(
                f"[diag] step={self.t:>5d} "
                f"rew={float(rews[0]):+.3f} "
                f"joint_track={_g('joint_pos_tracking'):+.3f} "
                f"link_track={_g('link_pos_tracking'):+.3f} "
                f"torso_ori={_g('torso_orientation_tracking'):+.3f} "
                f"action_abs_max={float(actions.abs().max()):.3f} "
                f"done={int(dones[0])}",
                flush=True,
            )

    def run(self):
        """Main simulation loop."""
        while True:
            self.step_simulation()


if __name__ == '__main__':
    play_manager = PlayManager(args, unknown)
    play_manager.run()
