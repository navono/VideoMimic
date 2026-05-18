"""RobotDeepMimicEnv — IsaacLab-based DeepMimic robot environment.

Migrated from the IsaacGym-based RobotDeepMimic(LeggedRobot).
Inherits from LeggedRobotEnv (DirectRLEnv) and replaces all IsaacGym API calls.
"""

import torch
import glob

from abc import ABC, abstractmethod
from typing import Tuple, List

from legged_gym.envs.base.legged_robot import LeggedRobotEnv
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.tensor_utils.replay_data import ReplayDataLoader
from legged_gym.tensor_utils.torch_jit_utils import (
    quat_mul, quat_conjugate, calc_heading_quat_inv, quat_rotate, calc_heading,
    normalize_angle, get_roll, get_pitch, get_yaw,
)
from legged_gym.utils.deepmimic_terrain import DeepMimicTerrain


# ---- Quaternion convention helpers ----
# torch_jit_utils uses xyzw convention.
# IsaacLab uses wxyz convention.
# Replay data stores xyzw.
# These helpers convert between the two conventions.

def _wxyz_to_xyzw(q):
    """Convert quaternion from wxyz (IsaacLab) to xyzw (torch_jit_utils)."""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)

def _xyzw_to_wxyz(q):
    """Convert quaternion from xyzw (torch_jit_utils) to wxyz (IsaacLab)."""
    return torch.cat([q[..., 3:4], q[..., :3]], dim=-1)


class RobotDeepMimicEnv(LeggedRobotEnv, ABC):
    """DeepMimic robot environment using IsaacLab's DirectRLEnv.

    Adds replay data loading, tracking rewards, contact matching,
    and DeepMimic-specific observation/reward functions.
    """

    @abstractmethod
    def get_replay_terrain_path(self, cfg: LeggedRobotCfg) -> Tuple[List[str], List[str], List]:
        pass

    def __init__(self, cfg: LeggedRobotCfg, render_mode: str | None = None, **kwargs):
        # Load replay data paths
        replay_data_path, terrain_paths, data_fps_override = self.get_replay_terrain_path(cfg)

        self.terrain_paths = terrain_paths

        # Setup terrain
        assert cfg.terrain.terrain_class == 'DeepMimicTerrain'
        self.terrain = DeepMimicTerrain(cfg.terrain, cfg.env.num_envs, terrain_paths)

        # Store flag for post-init setup
        self._deepmimic_needs_init = True

        super().__init__(cfg, render_mode, **kwargs)

        # Setup replay data loader (needs dof_names from _init_buffers)
        dof_names_in_file = [name.split('_joint')[0] for name in self.dof_names]

        if hasattr(cfg, 'deepmimic') and hasattr(cfg.deepmimic, 'tracked_body_names'):
            self.tracked_body_names = cfg.deepmimic.tracked_body_names
        else:
            self.tracked_body_names = self.body_names

        self.tracked_body_sim_names = self._resolve_body_names(self.tracked_body_names)
        self.tracked_body_indices = [self.body_names.index(name) for name in self.tracked_body_sim_names]

        if hasattr(cfg, 'deepmimic') and hasattr(cfg.deepmimic, 'extra_link_names'):
            self.extra_link_names = cfg.deepmimic.extra_link_names
            self.extra_link_sim_names = self._resolve_body_names(self.extra_link_names)
            self.torso_index = self.body_names.index(self.extra_link_sim_names[0])
            self.extra_link_torso_index = self.extra_link_names.index('torso_link')
        else:
            self.extra_link_names = None

        self.replay_data_loader = ReplayDataLoader(
            replay_data_path,
            self.num_envs, self.device, self.dt,
            dof_names=dof_names_in_file,
            motor_names=dof_names_in_file,
            link_names=self.tracked_body_names,
            contact_names=cfg.deepmimic.contact_names,
            data_quat_format='xyzw',
            adjust_root_pos=False,
            start_offset=0,
            height_direct_offset=cfg.deepmimic.height_direct_offset,
            randomize_start_offset=cfg.deepmimic.randomize_start_offset,
            n_prepend=cfg.deepmimic.n_prepend,
            n_append=cfg.deepmimic.n_append,
            extra_link_names=cfg.deepmimic.extra_link_names if hasattr(cfg.deepmimic, 'extra_link_names') else None,
            is_csv_joint_only=cfg.deepmimic.is_csv_joint_only,
            default_joint_order_type=cfg.deepmimic.default_joint_order_type,
            cut_off_import_length=cfg.deepmimic.cut_off_import_length,
            default_data_fps=cfg.deepmimic.default_data_fps if cfg.deepmimic.default_data_fps != -1 else 1 / self.dt,
            data_fps_override=data_fps_override,
            upsample_data=cfg.deepmimic.upsample_data,
            weighting_strategy=cfg.deepmimic.weighting_strategy,
            inorder_envs=cfg.env.export_trajectory,
            clip_weighting_strategy=cfg.deepmimic.clip_weighting_strategy,
            min_weight_factor=cfg.deepmimic.min_success_rate_weight_factor,
            max_weight_factor=cfg.deepmimic.max_success_rate_weight_factor,
        )
        self.ep_lengths = self.replay_data_loader.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.update_replay_data()

        self.camera_set = False
        self.env_offsets = self.terrain.get_terrain_offset(self.replay_data_loader.episode_indices)

        # Initialize visualization mode from config
        self.viz_replay_sync_robot = cfg.deepmimic.viz_replay_sync_robot if hasattr(cfg.deepmimic, 'viz_replay_sync_robot') else False

        if self.use_viser_viz:
            available_episodes = self.get_available_episodes()
            self.viser_viz.setup_clip_selection(available_episodes)

    def _init_buffers(self):
        super()._init_buffers()
        self.last_contact_state = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_target_contact = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.feet_air_max_height = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        # Per-env max episode length (in steps). DirectRLEnv.max_episode_length is a
        # read-only scalar property, so we use a separate attribute for the per-env tensor.
        self.ep_max_episode_length = torch.full((self.num_envs,), self.max_episode_length, dtype=torch.long, device=self.device)

    def get_terrain_paths(self):
        return self.terrain_paths

    # ---- Env/world frame conversion ----

    @property
    def env_root_pos(self):
        return self.root_states[:, 0:3] - self.env_offsets

    @property
    def env_rigid_body_pos(self):
        return self.rigid_body_pos - self.env_offsets.unsqueeze(1)

    def env_frame_to_world_frame(self, points, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if points.shape[1] == 3:
            return points + self.env_offsets[env_ids]
        else:
            return points + self.env_offsets[env_ids].unsqueeze(1)

    def world_frame_to_env_frame(self, points, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if points.shape[1] == 3:
            return points - self.env_offsets[env_ids]
        else:
            return points - self.env_offsets[env_ids].unsqueeze(1)

    # ---- Replay visualization ----

    def viz_replay_data(self, points=None, set_robot_pos=False):
        state = self.replay_data_loader.get_next_data()

        reset = self.episode_length_buf > self.ep_lengths
        self.ep_lengths = self.replay_data_loader.reset(reset)
        self.episode_length_buf[reset] = 0

        if not self.camera_set:
            base_pos = state.root_pos[0, 0, :]
            camera_offset = torch.tensor([1.0, 0.0, 1.0], device=self.device)
            # Camera setup handled by viewer in DirectRLEnv
            self.camera_set = True

        env_id = 0
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.int32)

        # current positions for link targets
        points = self.env_frame_to_world_frame(state.link_pos[env_id, 0, :], env_ids)

        i_rb = self.tracked_body_indices
        # current positions of replay link targets
        points_2 = self.rigid_body_pos[env_id, i_rb]

        # target link velocities
        points_3 = self.env_frame_to_world_frame(state.link_pos[env_id, 0, :] + state.link_vels[env_id, 0, :,], env_ids)
        # link velocities — use body_lin_vel_w instead of rigid_body_states view
        root_vels = self.rigid_body_vel[env_id, i_rb]
        points_4 = points + root_vels

        # Update Viser visualization for env 0
        if env_id == 0:
            target_points = points.cpu().numpy()
            current_points = points_2.cpu().numpy()
            velocity_points = points_4.cpu().numpy()
            if hasattr(self, 'viser_viz'):
                self.viser_viz.update_keypoints(target_points, current_points, velocity_points)

        if set_robot_pos:
            env_ids = torch.arange(self.num_envs, device=self.device)

            # Set root pose — convert replay quat from xyzw to wxyz for IsaacLab
            root_pos = self.env_frame_to_world_frame(state.root_pos[env_ids, 0, :], env_ids)
            root_quat_xyzw = state.root_quat[env_ids, 0, :]
            root_quat_wxyz = torch.cat([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], dim=-1)
            root_pose = torch.cat([root_pos, root_quat_wxyz], dim=-1)
            self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)

            # Zero root velocity
            root_vel = torch.zeros(self.num_envs, 6, device=self.device)
            self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

            # Set joint states
            dof_pos = state.dofs[env_ids, 0, :]
            dof_vel = torch.zeros_like(dof_pos)
            joint_ids = None if self._joint_order_is_identity else self._lab_to_train_perm.tolist()
            self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids, joint_ids=joint_ids)

    def set_visualization_episode(self, episode_idx: int, start_offset: int = 0):
        print(f'Setting episode {episode_idx} with start offset {start_offset}')
        self.selected_episode_idx = episode_idx
        self.selected_start_offset = start_offset
        self._reset_idx(torch.tensor([0], device=self.device), already_reset_replay_data=True)

    # ---- Reset ----

    def _reset_idx(self, env_ids, already_reset_replay_data=False):
        if len(env_ids) == 0:
            return

        # For visualization environment (env 0), if we have a selected episode, keep cycling it
        if 0 in env_ids and hasattr(self, 'selected_episode_idx'):
            other_env_ids = env_ids[env_ids != 0]
            if len(other_env_ids) > 0:
                env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                env_mask[other_env_ids] = True
                if not already_reset_replay_data:
                    reset_episode_length = self.replay_data_loader.reset(env_mask)
                    if self.cfg.deepmimic.truncate_rollout_length > 0:
                        self.ep_max_episode_length = torch.min(
                            self.cfg.deepmimic.truncate_rollout_length * torch.ones_like(reset_episode_length),
                            reset_episode_length
                        )
                    else:
                        self.ep_max_episode_length = reset_episode_length

            # For env 0, explicitly set it back to the selected episode
            self.replay_data_loader.set_env_data(0, self.selected_episode_idx, self.selected_start_offset)
        else:
            # Normal reset behavior for other environments
            env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            env_mask[env_ids] = True
            if not already_reset_replay_data:
                if self.cfg.deepmimic.truncate_rollout_length > 0:
                    reset_episode_length = self.replay_data_loader.reset(env_mask)
                    self.ep_max_episode_length = torch.min(
                        self.cfg.deepmimic.truncate_rollout_length * torch.ones_like(reset_episode_length),
                        reset_episode_length
                    )
                else:
                    self.ep_max_episode_length = self.replay_data_loader.reset(env_mask)

        self.reset_start_state = self.replay_data_loader.get_current_data()

        # Terrain offsets
        if len(env_ids) > 0:
            self.env_offsets[env_ids] = self.terrain.get_terrain_offset(self.reset_start_state.clip_index[env_ids])
            if self.cfg.deepmimic.randomize_terrain_offset:
                self.env_offsets[env_ids, 0:2] += torch.randn_like(self.env_offsets[env_ids, 0:2]) * self.cfg.deepmimic.randomize_terrain_offset_range

        if not self.camera_set:
            base_pos = self.reset_start_state.root_pos[0, :]
            self.camera_set = True

        # Call parent reset (sets default root/joint states)
        super()._reset_idx(env_ids)

        # Override with replay data states
        if len(env_ids) > 0:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

            # Ensure actions match trajectory to prevent large jerks
            self.actions[env_ids] = (self.dof_pos[env_ids] - self.default_dof_pos) / self.cfg.control.action_scale
            self.last_actions[env_ids] = self.actions[env_ids]
            self.last_last_actions[env_ids] = self.last_actions[env_ids]
            self.last_dof_vel[env_ids] = self.dof_vel[env_ids]

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.reset_start_state.dofs[env_ids]

        if hasattr(self.cfg.deepmimic, 'init_default_frac') and self.cfg.deepmimic.init_default_frac > 0:
            reset_default_frac = self.cfg.deepmimic.init_default_frac
            reset_default_mask = torch.rand(env_ids.shape, device=self.device) < reset_default_frac
            reset_default_env_ids = env_ids[reset_default_mask]
            self.dof_pos[reset_default_env_ids] = self.default_dof_pos
        else:
            reset_default_env_ids = env_ids

        if self.cfg.deepmimic.init_velocities:
            self.dof_vel[env_ids] = self.reset_start_state.motor_vels[env_ids]
            self.dof_vel[reset_default_env_ids] = 0.
        else:
            self.dof_vel[env_ids] = 0.

        self.dof_pos[env_ids] = self.dof_pos[env_ids] + self.cfg.noise.init_noise_scales.dof_pos * (torch.randn_like(self.dof_pos[env_ids]))
        self.dof_vel[env_ids] = self.dof_vel[env_ids] + self.cfg.noise.init_noise_scales.dof_vel * (torch.randn_like(self.dof_pos[env_ids]))

        # Write to simulation via IsaacLab API. ``self.dof_pos`` is held in
        # training (URDF) order; map to IsaacLab joint indices via the perm.
        joint_ids = None if self._joint_order_is_identity else self._lab_to_train_perm.tolist()
        self.robot.write_joint_state_to_sim(
            self.dof_pos[env_ids], self.dof_vel[env_ids], env_ids=env_ids, joint_ids=joint_ids,
        )

    def _reset_root_states(self, env_ids):
        root_pos = self.env_frame_to_world_frame(self.reset_start_state.root_pos[env_ids], env_ids)
        root_pos[:, 0:2] += self.cfg.noise.init_noise_scales.root_xy * (torch.randn_like(root_pos[:, 0:2]))
        root_pos[:, 2:3] += self.cfg.noise.init_noise_scales.root_z * (torch.randn_like(root_pos[:, 2:3]))
        root_pos[:, 2] += self.cfg.deepmimic.respawn_z_offset

        # Replay data stores quaternion in xyzw; IsaacLab uses wxyz
        root_quat_xyzw = self.reset_start_state.root_quat[env_ids]
        random_quat = torch.randn_like(root_quat_xyzw)
        root_quat_xyzw = root_quat_xyzw + random_quat * self.cfg.noise.init_noise_scales.root_quat
        # Normalize
        root_quat_xyzw = root_quat_xyzw / torch.norm(root_quat_xyzw, dim=-1, keepdim=True)
        # Convert xyzw -> wxyz
        root_quat_wxyz = torch.cat([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], dim=-1)

        root_pose = torch.cat([root_pos, root_quat_wxyz], dim=-1)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)

        # Set root velocities
        if self.cfg.deepmimic.init_velocities:
            root_lin_vel = self.reset_start_state.root_vel[env_ids]
            root_ang_vel = self.reset_start_state.root_ang_vel[env_ids]
        else:
            root_lin_vel = torch.zeros(len(env_ids), 3, device=self.device)
            root_ang_vel = torch.zeros(len(env_ids), 3, device=self.device)

        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    # ---- Replay data update ----

    def update_replay_data(self):
        state = self.replay_data_loader.get_current_data()
        self.target_root_pos = state.root_pos
        self.target_root_quat = state.root_quat
        self.target_root_vel = state.root_vel
        self.target_root_ang_vel = state.root_ang_vel
        self.target_dofs = state.dofs
        self.target_motors = state.motors
        self.target_motor_vels = state.motor_vels
        self.target_link_pos = state.link_pos
        self.target_link_quat = state.link_quat
        self.target_link_vel = state.link_vels
        self.target_contacts = state.contacts
        self.target_extra_link_pos = state.extra_link_pos
        self.target_extra_link_quat = state.extra_link_quat
        self.target_extra_link_vel = state.extra_link_vels
        self.target_extra_link_ang_vel = state.extra_link_ang_vels

    def _post_step_update(self):
        freeze_env_prob = self.cfg.noise.playback_noise_scales.freeze_env_prob
        unfreeze_env_prob = self.cfg.noise.playback_noise_scales.unfreeze_env_prob

        if freeze_env_prob > 0:
            freeze_mask = torch.rand(self.num_envs, device=self.device) < freeze_env_prob
            unfreeze_mask = torch.rand(self.num_envs, device=self.device) < unfreeze_env_prob

            if not hasattr(self, 'frozen_env_mask'):
                self.frozen_env_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

            self.frozen_env_mask[freeze_mask] = True
            self.frozen_env_mask[unfreeze_mask] = False

            self.replay_data_loader.increment_indices(env_ids=(~self.frozen_env_mask).nonzero(as_tuple=False).flatten())
            self.episode_length_buf[self.frozen_env_mask] -= 1
        else:
            self.replay_data_loader.increment_indices()

        self.update_replay_data()

        self.link_pos_error = self._compute_link_pos_error()
        self.link_vel_error = self._compute_link_vel_error()

        if self.cfg.deepmimic.viz_replay:
            self.viz_replay_data(set_robot_pos=self.viz_replay_sync_robot)
            if self.viz_replay_sync_robot:
                return

        super()._post_step_update()

    def get_current_replay_state(self, env_ids):
        state = self.replay_data_loader.get_current_data()
        for attr in state.__dict__:
            if isinstance(getattr(state, attr), torch.Tensor):
                setattr(state, attr, getattr(state, attr)[env_ids])
        return state

    def get_next_replay_state(self, env_ids, K=1, collapse_next_dim=True):
        state = self.replay_data_loader.get_next_data(K=K)
        for attr in state.__dict__:
            if isinstance(getattr(state, attr), torch.Tensor):
                to_set = getattr(state, attr)
                if K == 1 and collapse_next_dim and len(to_set.shape) > 2:
                    to_set = to_set.squeeze(1)
                setattr(state, attr, to_set[env_ids])
        return state

    # ---- Termination ----

    def _get_dones(self):
        # Run post-step update first (replay data, link errors, etc.)
        self._post_step_update()

        died = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1
        )

        link_pos_error = torch.norm(self.link_pos_error, dim=-1)
        link_pos_error_threshold = self.cfg.deepmimic.link_pos_error_threshold
        died |= torch.any(link_pos_error > link_pos_error_threshold, dim=1) & (self.episode_length_buf >= 2)

        self._compute_invalid_changes()

        if self.cfg.asset.terminate_after_large_feet_contact_forces:
            died |= torch.any(
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.asset.large_feet_contact_force_threshold,
                dim=1
            )

        time_out = self.episode_length_buf >= self.ep_max_episode_length

        return died, time_out

    def _compute_invalid_changes(self):
        current_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        target_contact = self.target_contacts

        contact_changed = current_contact != self.last_contact_state
        target_changed = target_contact != self.last_target_contact
        current_is_wrong = current_contact != target_contact

        invalid_changes = contact_changed & ~target_changed & current_is_wrong
        invalid_changes = (invalid_changes) & (self.episode_length_buf > 10).unsqueeze(1)

        self.invalid_changes = invalid_changes
        self.last_contact_state = current_contact
        self.last_target_contact = target_contact

    def _compute_link_pos_error(self):
        return self.env_rigid_body_pos[:, self.tracked_body_indices] - self.target_link_pos

    def _compute_link_vel_error(self):
        return self.rigid_body_vel[:, self.tracked_body_indices] - self.target_link_vel

    # ---- Rewards ----

    def _reward_joint_pos_tracking(self):
        motor_pos_error = self.dof_pos - self.target_motors
        k = self.cfg.rewards.joint_pos_tracking_k
        if not getattr(self, '_dbg_joint_track_printed', False):
            self._dbg_joint_track_printed = True
            with torch.no_grad():
                print("[dbg-joint-track] dof_names (train order):", list(self.dof_names))
                print("[dbg-joint-track] dof_pos[0]      :", self.dof_pos[0].cpu().tolist())
                print("[dbg-joint-track] target_motors[0]:", self.target_motors[0].cpu().tolist())
                print("[dbg-joint-track] per-joint error :", (self.dof_pos[0] - self.target_motors[0]).cpu().tolist())
                print("[dbg-joint-track] sum-sq error    :", float((motor_pos_error[0] ** 2).sum()))
        return torch.exp(-torch.pow(motor_pos_error, 2).sum(dim=-1) * k)

    def _reward_joint_vel_tracking(self):
        motor_vel_error = self.dof_vel - self.target_motor_vels
        k = self.cfg.rewards.joint_vel_tracking_k
        return torch.exp(-torch.pow(motor_vel_error, 2).sum(dim=-1) * k)

    def _reward_root_pos_tracking(self):
        root_pos_error = self.env_root_pos - self.target_root_pos
        k = self.cfg.rewards.root_pos_tracking_k
        return torch.exp(-torch.pow(root_pos_error, 2).sum(dim=-1) * k)

    def _reward_root_orientation_tracking(self):
        # IsaacLab: root_states[:, 3:7] is wxyz — convert to xyzw for torch_jit_utils
        root_quat_xyzw = _wxyz_to_xyzw(self.root_states[:, 3:7])
        target_quat_xyzw = self.target_root_quat  # already xyzw from replay data

        quat_diff = quat_mul(root_quat_xyzw, quat_conjugate(target_quat_xyzw))
        root_orientation_error = 2. * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
        k = self.cfg.rewards.root_orientation_tracking_k
        return torch.exp(-root_orientation_error * k)

    def _reward_torso_pos_tracking(self):
        torso_pos_error = self.env_rigid_body_pos[:, self.torso_index] - self.target_extra_link_pos[:, self.extra_link_torso_index]
        k = self.cfg.rewards.torso_pos_tracking_k
        return torch.exp(-torch.pow(torso_pos_error, 2).sum(dim=-1) * k)

    def _reward_torso_orientation_tracking(self):
        # Convert IsaacLab wxyz to xyzw for torch_jit_utils
        torso_quat_xyzw = _wxyz_to_xyzw(self.rigid_body_quat[:, self.torso_index])
        target_quat_xyzw = self.target_extra_link_quat[:, self.extra_link_torso_index]

        quat_diff = quat_mul(torso_quat_xyzw, quat_conjugate(target_quat_xyzw))
        torso_orientation_error = 2. * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
        k = self.cfg.rewards.torso_orientation_tracking_k
        return torch.exp(-torso_orientation_error * k)

    def _reward_link_pos_tracking(self):
        k = self.cfg.rewards.link_pos_tracking_k
        assert len(self.tracked_body_names) == 13
        return torch.exp(-torch.pow(self.link_pos_error, 2).sum(dim=1).sum(dim=1) * k)

    def _reward_feet_max_height_for_this_air(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)

        self.last_contacts = contact
        self.last_contacts_filt = contact_filt

        self.feet_air_max_height = torch.max(self.feet_air_max_height, self.env_rigid_body_pos[:, self.feet_indices, 2])

        desired_feet_max_height_for_this_air = 0.2
        rew_feet_max_height = torch.sum(
            (torch.clamp_min(desired_feet_max_height_for_this_air - self.feet_air_max_height, 0)) * from_air_to_contact,
            dim=1
        )
        self.feet_air_max_height *= ~contact_filt
        return rew_feet_max_height

    def _reward_link_vel_tracking(self):
        k = self.cfg.rewards.link_vel_tracking_k
        return torch.exp(-torch.pow(self.link_vel_error, 2).sum(dim=1).sum(dim=1) * k)

    def _reward_root_vel_tracking(self):
        root_vel_error = self.root_states[:, 7:10] - self.target_root_vel
        k = self.cfg.rewards.root_vel_tracking_k
        return torch.exp(-torch.norm(root_vel_error, dim=-1) * k)

    def _reward_root_ang_vel_tracking(self):
        root_ang_vel_error = self.root_states[:, 10:13] - self.target_root_ang_vel
        k = self.cfg.rewards.root_ang_vel_tracking_k
        return torch.exp(-torch.norm(root_ang_vel_error, dim=-1) * k)

    def _reward_feet_contact_matching(self):
        if self.cfg.deepmimic.contact_names is None:
            raise ValueError('Contact names are not set')
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        desired_contact = self.target_contacts
        return torch.sum((contact == desired_contact).float(), dim=1)

    def _reward_contact_smoothness(self):
        return self.invalid_changes.float().sum(dim=1)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
        fly = (~contacts[:, 0]) & (~contacts[:, 1])
        should_fly = (self.target_contacts[:, 0]).bool() & (self.target_contacts[:, 1]).bool()
        fly_penalty = (fly & ~should_fly).float()
        return fly_penalty

    def _reward_feet_swing_height(self):
        swing_height_target = 0.08
        if self.cfg.deepmimic.contact_names is None:
            contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
            pos_error = torch.square(self.feet_pos[:, :, 2] - swing_height_target) * (1 - contact.float())
            pos_error += swing_height_target * contact.float()
        else:
            pos_error = torch.square(self.feet_pos[:, :, 2] - swing_height_target) * (1 - self.target_contacts.float())
            pos_error += swing_height_target * self.target_contacts.float()
        return torch.sum(pos_error, dim=(1))

    def _reward_ankle_action(self):
        ankle_indices = [
            self.dof_names.index('right_ankle_pitch_joint'),
            self.dof_names.index('left_ankle_pitch_joint'),
            self.dof_names.index('right_ankle_roll_joint'),
            self.dof_names.index('left_ankle_roll_joint')
        ]
        ankle_action = self.actions[:, ankle_indices]
        return torch.pow(ankle_action, 2).sum(dim=-1)

    def _reward_feet_orientation(self):
        # Convert IsaacLab wxyz to xyzw for torch_jit_utils
        feet_quat_xyzw = _wxyz_to_xyzw(self.rigid_body_quat[:, self.feet_indices])
        body_quat_xyzw = _wxyz_to_xyzw(self.root_states[:, 3:7])
        assert len(self.feet_indices) == 2

        feet_orientation_error = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for i in range(2):
            foot_heading = calc_heading(feet_quat_xyzw[:, i])
            body_heading = calc_heading(body_quat_xyzw)
            heading_error = torch.abs(foot_heading - body_heading)
            feet_orientation_error += heading_error

        k = 6.0
        return torch.exp(-feet_orientation_error * k)

    # ---- Observations ----

    def _obs_torso(self):
        return torch.cat((
            self.get_sensor_data('root_height').squeeze(2),
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

    def _obs_torso_real(self):
        obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity + self.gravity_rand_seed * self.cfg.noise.offset_scales.gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos + self.dof_pos_rand_seed * self.cfg.noise.offset_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        if not hasattr(self, 'obs_torso_real_noise_scale'):
            self.obs_torso_real_noise_scale = self._get_noise_scale_vec_torso_real(obs)

        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.obs_torso_real_noise_scale
        return obs

    def _obs_phase(self):
        phase = self.replay_data_loader.get_episode_phase()
        if hasattr(self, 'phase_offset'):
            phase = phase + self.phase_offset / self.ep_max_episode_length
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        return torch.cat((sin_phase, cos_phase), dim=-1)

    def _obs_torso_xy_rel(self):
        torso_pos = self.env_rigid_body_pos[:, self.torso_index]
        torso_pos_rel = self.target_extra_link_pos[:, self.extra_link_torso_index] - torso_pos

        # Convert IsaacLab wxyz to xyzw for torch_jit_utils
        torso_quat_xyzw = _wxyz_to_xyzw(self.rigid_body_quat[:, self.torso_index])
        heading = calc_heading_quat_inv(torso_quat_xyzw)
        obs_torso_pos_rel = quat_rotate(heading, torso_pos_rel)

        rel_xy = obs_torso_pos_rel[:, :2]

        if self.cfg.deepmimic.zero_torso_xy:
            rel_xy[:] = 0.0

        rel_xy += self.cfg.noise.noise_scales.rel_xy * torch.randn_like(rel_xy)

        if self.cfg.domain_rand.randomize_odom_update_frequency:
            if hasattr(self, 'last_rel_xy'):
                last_rel_xy = self.last_rel_xy
            else:
                last_rel_xy = rel_xy

            update = torch.remainder(self.episode_length_buf + self.odom_update_frequency_offset, self.odom_update_frequency) == 0
            update = update & (self.episode_length_buf > 0)
            rel_xy[~update] = last_rel_xy[~update]

        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            manual_offset_target = torch.zeros((self.num_envs, 2), device=self.device)
            lin_vel_scale = 0.5
            if self.viser_viz.move_forward.value:
                manual_offset_target[:, 0] = lin_vel_scale
            if self.viser_viz.move_back.value:
                manual_offset_target[:, 0] = -lin_vel_scale
            if self.viser_viz.move_left.value:
                manual_offset_target[:, 1] = lin_vel_scale
            if self.viser_viz.move_right.value:
                manual_offset_target[:, 1] = -lin_vel_scale
            return manual_offset_target

        return rel_xy.view(-1, 2)

    def _obs_torso_yaw_rel(self):
        # Convert IsaacLab wxyz to xyzw for torch_jit_utils
        torso_quat_xyzw = _wxyz_to_xyzw(self.rigid_body_quat[:, self.torso_index])
        target_quat_xyzw = self.target_extra_link_quat[:, self.extra_link_torso_index]  # already xyzw

        target_heading = calc_heading(target_quat_xyzw)
        torso_heading = calc_heading(torso_quat_xyzw)
        heading_error = target_heading - torso_heading
        heading_error = normalize_angle(heading_error)

        if self.cfg.deepmimic.zero_torso_yaw:
            heading_error[:] = 0.0

        heading_error += self.cfg.noise.noise_scales.rel_yaw * torch.randn_like(heading_error)

        if self.cfg.domain_rand.randomize_odom_update_frequency:
            if hasattr(self, 'last_torso_yaw_rel'):
                last_torso_yaw_rel = self.last_torso_yaw_rel
            else:
                last_torso_yaw_rel = heading_error

            update = torch.remainder(self.episode_length_buf + self.odom_update_frequency_offset, self.odom_update_frequency) == 0
            update = update & (self.episode_length_buf > 0)
            heading_error[~update] = last_torso_yaw_rel[~update]

        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            manual_ang_vel = torch.zeros((self.num_envs, 1), device=self.device)
            ang_vel_scale = 0.3
            if self.viser_viz.rotate_left.value:
                manual_ang_vel[:, 0] = ang_vel_scale
            if self.viser_viz.rotate_right.value:
                manual_ang_vel[:, 0] = -ang_vel_scale
            return manual_ang_vel

        return heading_error.view(-1, 1)

    def _obs_upper_body_joint_targets(self):
        upper_body_dof_names = self.cfg.asset.upper_body_dof_names
        upper_body_dof_indices = [self.dof_names.index(name) for name in upper_body_dof_names]
        upper_body_joint_targets = (
            self.target_motors[:, upper_body_dof_indices] - self.default_dof_pos[:, upper_body_dof_indices]
        ) * self.obs_scales.dof_pos
        return upper_body_joint_targets

    def _obs_torso_xy(self):
        torso_pos = self.env_rigid_body_pos[:, self.torso_index]
        return torso_pos[:, :2]

    def _obs_torso_yaw(self):
        torso_quat_xyzw = _wxyz_to_xyzw(self.rigid_body_quat[:, self.torso_index])
        return calc_heading(torso_quat_xyzw)

    def _obs_target_joints(self):
        return (self.target_motors - self.default_dof_pos) * self.obs_scales.dof_pos

    def _obs_target_root_roll(self):
        # Replay quat is already xyzw — torch_jit_utils expects xyzw
        return normalize_angle(get_roll(self.target_root_quat)).unsqueeze(1)

    def _obs_target_root_pitch(self):
        return normalize_angle(get_pitch(self.target_root_quat)).unsqueeze(1)

    def _obs_target_root_yaw(self):
        return normalize_angle(get_yaw(self.target_root_quat)).unsqueeze(1)

    def _obs_deepmimic(self):
        K = self.cfg.deepmimic.num_next_obs

        state = self.replay_data_loader.get_next_data(K=K)
        target_root_pos = state.root_pos
        target_root_quat = state.root_quat  # xyzw from replay data
        target_dofs = state.dofs
        target_motors = state.motors
        target_motor_vels = state.motor_vels
        target_link_pos = state.link_pos
        target_link_quat = state.link_quat
        target_link_vel = state.link_vels
        target_contacts = state.contacts

        # Convert IsaacLab wxyz to xyzw for torch_jit_utils
        torso_rot_xyzw = _wxyz_to_xyzw(self.root_states[:, 3:7])
        heading = calc_heading_quat_inv(torso_rot_xyzw)

        obs_root_pos = target_root_pos - self.env_root_pos.unsqueeze(1)
        obs_root_pos = quat_rotate(
            heading.unsqueeze(1).repeat(1, K, 1).view(self.num_envs * K, 4),
            obs_root_pos.view(self.num_envs * K, 3)
        ).view(self.num_envs, K, 3)

        # Both target_root_quat and torso_rot are now in xyzw
        obs_root_quat = quat_mul(target_root_quat, quat_conjugate(torso_rot_xyzw.unsqueeze(1).repeat(1, K, 1)))

        obs_joints = target_motors - self.dof_pos.unsqueeze(1)

        current_tracked_link_pos = self.env_rigid_body_pos[:, self.tracked_body_indices]
        target_link_pos_global = target_link_pos
        obs_link_pos = (target_link_pos_global - current_tracked_link_pos.unsqueeze(1))

        obs_tracked_link_heights = self.get_sensor_data('link_heights')

        num_tracked_links = self.cfg.deepmimic.num_tracked_links

        heading_expand = heading.unsqueeze(1).repeat(1, K * num_tracked_links, 1).view(self.num_envs * K * num_tracked_links, 4)
        obs_link_pos = quat_rotate(
            heading_expand,
            obs_link_pos.view(self.num_envs * K * num_tracked_links, 3)
        ).view(self.num_envs, K, num_tracked_links, 3)

        heading_expand_vels = heading.unsqueeze(1).repeat(1, num_tracked_links, 1).view(self.num_envs * num_tracked_links, 4)
        link_vels_global = self.rigid_body_vel[:, self.tracked_body_indices]
        obs_link_vel = quat_rotate(
            heading_expand_vels,
            link_vels_global.view(self.num_envs * num_tracked_links, 3)
        ).view(self.num_envs, num_tracked_links, 3)

        obs = torch.cat((
            obs_tracked_link_heights.view(self.num_envs, -1),
            obs_root_quat.view(self.num_envs, -1),
            obs_root_pos.view(self.num_envs, -1),
            obs_joints.view(self.num_envs, -1),
            obs_link_pos.view(self.num_envs, -1),
            obs_link_vel.view(self.num_envs, -1),
        ), dim=-1)

        if self.cfg.deepmimic.contact_names is not None:
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.
            contact_changed = contact != self.last_contact_state
            target_changed = self.target_contacts != self.last_target_contact
            current_is_wrong = contact != self.target_contacts
            invalid_changes = contact_changed & ~target_changed & current_is_wrong

            obs = torch.cat((
                obs,
                contact.float().view(self.num_envs, -1),
                target_contacts.float().view(self.num_envs, -1),
                self.last_contact_state.float().view(self.num_envs, -1),
                invalid_changes.float().view(self.num_envs, -1),
            ), dim=-1)

        return obs

    def _obs_deepmimic_lin_ang_vel(self):
        K = 1
        state = self.replay_data_loader.get_next_data(K=K)
        target_root_vel = state.root_vel
        target_root_ang_vel = state.root_ang_vel

        # Convert IsaacLab wxyz to xyzw for torch_jit_utils
        torso_rot_xyzw = _wxyz_to_xyzw(self.root_states[:, 3:7])
        heading = calc_heading_quat_inv(torso_rot_xyzw)

        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            manual_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
            manual_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
            lin_vel_scale = 0.5
            ang_vel_scale = 1.0

            if self.viser_viz.move_forward.value:
                manual_lin_vel[:, 0] = lin_vel_scale
            if self.viser_viz.move_back.value:
                manual_lin_vel[:, 0] = -lin_vel_scale
            if self.viser_viz.move_left.value:
                manual_lin_vel[:, 1] = lin_vel_scale
            if self.viser_viz.move_right.value:
                manual_lin_vel[:, 1] = -lin_vel_scale
            if self.viser_viz.move_up.value:
                manual_lin_vel[:, 2] = lin_vel_scale
            if self.viser_viz.move_down.value:
                manual_lin_vel[:, 2] = -lin_vel_scale
            if self.viser_viz.rotate_left.value:
                manual_ang_vel[:, 2] = ang_vel_scale
            if self.viser_viz.rotate_right.value:
                manual_ang_vel[:, 2] = -ang_vel_scale

            obs_lin_vel = manual_lin_vel
            obs_ang_vel = manual_ang_vel
            obs = torch.cat((
                obs_lin_vel.view(self.num_envs, -1),
                obs_ang_vel.view(self.num_envs, -1),
            ), dim=-1)
        else:
            obs_lin_vel = quat_rotate(heading, target_root_vel.squeeze(1))
            obs_ang_vel = quat_rotate(heading, target_root_ang_vel.squeeze(1))

            obs = torch.cat((
                obs_lin_vel.view(self.num_envs, -1),
                obs_ang_vel.view(self.num_envs, -1),
            ), dim=-1)

            if not hasattr(self, 'obs_lin_ang_vel_noise_scale'):
                self.obs_lin_ang_vel_noise_scale = self._get_noise_scale_vec_lin_ang_vel(obs)

            if self.add_noise:
                obs += (2 * torch.rand_like(obs) - 1) * self.obs_lin_ang_vel_noise_scale

        return obs

    def _manual_obs_teacher(self):
        return torch.cat((
            self.obs_dict['history_torso_real'].view(self.num_envs, -1),
            self.obs_dict['history_torso_xy_rel'].view(self.num_envs, -1),
            self.obs_dict['history_torso_yaw_rel'].view(self.num_envs, -1),
            self.obs_dict['target_joints'].view(self.num_envs, -1),
            self.obs_dict['target_root_roll'].view(self.num_envs, -1),
            self.obs_dict['target_root_pitch'].view(self.num_envs, -1),
        ), dim=-1)

    def _get_noise_scale_vec_torso_real(self, obs):
        noise_vec = torch.zeros_like(obs[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:6 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6 + self.num_actions:6 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6 + 2 * self.num_actions:6 + 3 * self.num_actions] = 0.
        return noise_vec

    def _get_noise_scale_vec_lin_ang_vel(self, obs):
        noise_vec = torch.zeros_like(obs[0])
        noise_vec[:3] = self.cfg.noise.noise_scales.lin_vel * self.cfg.noise.noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = self.cfg.noise.noise_scales.ang_vel * self.cfg.noise.noise_level * self.obs_scales.ang_vel
        return noise_vec

    # ---- Control ----

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type in ['P', 'V', 'T', 'DIRECT']:
            return super()._compute_torques(actions)
        elif control_type == 'DEEPMIMIC_DELTA':
            torques = (actions_scaled + self.target_motors - self.dof_pos) * self.p_gains - self.d_gains * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ---- Domain randomization ----

    def _init_randomisation_buffers(self):
        super()._init_randomisation_buffers()
        self.dof_pos_rand_seed = torch.randn_like(self.dof_pos)
        self.gravity_rand_seed = torch.randn_like(self.projected_gravity)
        if self.cfg.domain_rand.randomize_odom_update_frequency:
            self.odom_update_frequency = torch.randint(
                self.cfg.domain_rand.odom_update_steps_min,
                self.cfg.domain_rand.odom_update_steps_max,
                (self.num_envs,), device=self.device
            )
            self.odom_update_frequency_offset = torch.randint(
                0, self.cfg.domain_rand.odom_update_steps_max,
                (self.num_envs,), device=self.device
            )

    def _resample_episodic_randomisations(self, env_ids):
        super()._resample_episodic_randomisations(env_ids)
        self.dof_pos_rand_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.gravity_rand_seed[env_ids] = torch.randn_like(self.gravity_rand_seed[env_ids])
        if self.cfg.domain_rand.randomize_odom_update_frequency:
            self.odom_update_frequency[env_ids] = torch.randint(
                self.cfg.domain_rand.odom_update_steps_min,
                self.cfg.domain_rand.odom_update_steps_max,
                (len(env_ids),), device=self.device
            )
            self.odom_update_frequency_offset[env_ids] = torch.randint(
                0, self.cfg.domain_rand.odom_update_steps_max,
                (len(env_ids),), device=self.device
            )

    def _push_robots(self):
        if self.use_viser_viz and not self.viser_viz.enable_push_robots.value:
            return

        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return

        max_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        max_vel_z = 0.1
        if self.use_viser_viz:
            max_vel_xy *= self.viser_viz.push_force_scale.value
            max_vel_z = self.viser_viz.push_force_z_scale.value

        # Apply random pushes by writing root velocity directly
        root_lin_vel = self.robot.data.root_lin_vel_w.clone()
        root_ang_vel = self.robot.data.root_ang_vel_w.clone()

        root_lin_vel[:, 0:2] += torch_rand_float(-max_vel_xy, max_vel_xy, (self.num_envs, 2), device=self.device)
        root_lin_vel[:, 2:3] += torch_rand_float(-max_vel_z, max_vel_z, (self.num_envs, 1), device=self.device)

        root_vel_6d = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_velocity_to_sim(root_vel_6d, env_ids=push_env_ids)
