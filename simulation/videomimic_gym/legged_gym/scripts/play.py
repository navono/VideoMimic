import os
import isaacgym
import torch
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import parse_unknown_args, get_load_path, class_to_dict
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

        # Initialize PPO runner and policy
        self.ppo_runner, self.train_cfg = task_registry.make_alg_runner(
            env=self.env, name=args.task, args=args, train_cfg=self.train_cfg, train_overrides=self.train_overrides
        )
        self.policy = self.ppo_runner.get_inference_policy(device=self.env.device)

        # Success rate tracking
        self.total_episodes = 0
        self.success_count = 0
        self.max_episodes = getattr(args, 'max_episodes', None)
        self.episode_rewards = []
        self.episode_link_errors = []
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
        # Reset environment so the new policy starts from a clean state
        self.env.reset()

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
        """Perform one simulation step for both IsaacGym."""
        obs = self.env.get_observations()

        actions = self.policy({k: v.detach() for k, v in obs.items()}, monitor_activations=False)

        # Step the IsaacGym environment
        obs, rews, dones, infos = self.env.step(actions.detach())

        # Accumulate per-step metrics for running episode
        self.episode_rewards.append(rews[0].item())
        if hasattr(self.env, 'link_pos_error'):
            self.episode_link_errors.append(torch.norm(self.env.link_pos_error[0], dim=-1).mean().item())

        # Log success/failure when episode ends
        if dones[0]:
            self.total_episodes += 1
            total_reward = sum(self.episode_rewards)
            avg_link_error = sum(self.episode_link_errors) / max(len(self.episode_link_errors), 1)

            # Success = average tracking error below threshold
            # Walking/stairs: ~0.05-0.1m, sitting/standing: ~0.3-0.4m
            success_threshold = 0.4
            is_success = avg_link_error < success_threshold
            if is_success:
                self.success_count += 1
            rate = self.success_count / self.total_episodes * 100
            status = "SUCCESS" if is_success else "FAILED "
            print(f"[Episode {self.total_episodes}] {status} | "
                  f"total_reward: {total_reward:.2f} | "
                  f"avg_link_error: {avg_link_error:.4f}m | "
                  f"success rate: {self.success_count}/{self.total_episodes} ({rate:.1f}%)")

            self.episode_rewards = []
            self.episode_link_errors = []

            if self.max_episodes and self.total_episodes >= self.max_episodes:
                rate = self.success_count / self.total_episodes * 100
                print(f"\n=== Done. {self.success_count}/{self.total_episodes} success ({rate:.1f}%) ===")
                return False

        return True

    def run(self):
        """Main simulation loop."""
        while self.step_simulation():
            pass


if __name__ == '__main__':
    args, unknown = get_args()

    play_manager = PlayManager(args, unknown)
    play_manager.run()
