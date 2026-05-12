"""G1RobotEnv — IsaacLab-based G1 legged robot environment.

Migrated from the IsaacGym-based G1Robot(LeggedRobot).
Inherits from LeggedRobotEnv (DirectRLEnv) and replaces all IsaacGym API calls.
"""

import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobotEnv


class G1RobotEnv(LeggedRobotEnv):
    """G1-specific legged robot environment using IsaacLab.

    Adds foot tracking, gait phase computation, and G1-specific rewards.
    """

    def _init_buffers(self):
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        super()._init_buffers()
        self._init_foot()

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        # IsaacLab: body states accessed directly from Articulation data
        self.feet_pos = self.rigid_body_pos[:, self.feet_indices, :3]
        self.feet_vel = self.rigid_body_vel[:, self.feet_indices, :3]

    def update_feet_state(self):
        # No need to call gym.refresh — IsaacLab auto-updates on scene.update()
        self.feet_pos = self.rigid_body_pos[:, self.feet_indices, :3]
        self.feet_vel = self.rigid_body_vel[:, self.feet_indices, :3]

    def _post_step_update(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        return super()._post_step_update()

    # ---- Observations ----

    def _obs_torso(self):
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        obs = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)
        return obs

    def _obs_torso_privileged(self):
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        return torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_dict['torso'])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.  # sin/cos phase
        return noise_vec

    # ---- Rewards ----

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        new_penalty = torch.norm(contact_feet_vel[:, :, :3], dim=2).sum(dim=1)
        return new_penalty

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)
