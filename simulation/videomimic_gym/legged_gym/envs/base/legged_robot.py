"""LeggedRobotEnv — IsaacLab DirectRLEnv-based legged robot environment.

Migrated from the IsaacGym-based LeggedRobot + BaseTask classes.
Inherits from isaaclab.envs.DirectRLEnv and replaces all IsaacGym API calls
with IsaacLab's Articulation/Scene/SimulationContext APIs.
"""

import math
import os
import time

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Dict, Sequence

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg, UrdfFileCfg, UsdFileCfg
from isaaclab.utils import configclass

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.history import HistoryHandler
from .legged_robot_config import LeggedRobotCfg


class LeggedRobotEnv(DirectRLEnv):
    """Legged robot environment using IsaacLab's DirectRLEnv.

    Replaces the old BaseTask + LeggedRobot (IsaacGym) hierarchy.

    Accepts a LeggedRobotCfg (custom config) and internally builds a
    DirectRLEnvCfg for the DirectRLEnv base class. This allows child
    configs (G1RoughCfg, etc.) to remain unchanged.
    """

    cfg: LeggedRobotCfg

    def __init__(self, cfg: LeggedRobotCfg, render_mode: str | None = None, **kwargs):
        # Store the VideoMimic config — used throughout for domain-specific settings
        self._robot_cfg = cfg
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        # Initialize viser flag
        self.use_viser_viz = hasattr(cfg, 'viser') and cfg.viser.enable

        # Initialize trajectory export
        self.export_trajectory = cfg.env.export_trajectory
        if self.export_trajectory:
            self.export_dir = cfg.env.export_dir
            os.makedirs(self.export_dir, exist_ok=True)

        # Parse config early (computes dt, max_episode_length, etc.)
        self._parse_cfg(cfg)

        # Validate sensor configuration vs requested observations
        self._validate_sensor_observations()

        # Build a DirectRLEnvCfg from LeggedRobotCfg fields
        direct_cfg = self._build_direct_rl_env_cfg(cfg)

        # Call DirectRLEnv.__init__ — this creates sim, scene, and calls _setup_scene
        super().__init__(direct_cfg, render_mode, **kwargs)

        # Override self.cfg to point to the original LeggedRobotCfg
        # so that all downstream code (reward/obs functions, child classes) work unchanged
        self.cfg = cfg

        # Initialize buffers after scene is set up
        self._init_buffers()
        self._prepare_reward_function()
        self._prepare_observation_function()

        # Initialize viser if enabled
        if self.use_viser_viz:
            from legged_gym.utils.viser_visualizer import LeggedRobotViser
            import viser
            from viser.extras import ViserUrdf
            from robot_descriptions.loaders.yourdfpy import load_robot_description
            self.viser_viz = LeggedRobotViser(
                urdf_path=cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
                dt=cfg.control.decimation * direct_cfg.sim.dt
            )
            self.viser_viz.init_isaaclab_robot(self)

        if self.export_trajectory:
            self.trajectory_data = [{
                'joint_names': [],
                'joints': [],
                'root_quat': [],
                'root_pos': [],
                'link_names': [],
                'link_pos': [],
                'link_quat': [],
                'contacts': {},
                'trajectory_name': None,
                'joint_targets': [],
                'obs': {}
            } for _ in range(cfg.env.num_envs)]
            self.env_episode_done = torch.zeros(cfg.env.num_envs, dtype=torch.bool, device=self.device)
            self.num_envs_completed = 0
            self.stepped_before_export = False

        self.init_done = True
        self.history_handler = None

    @staticmethod
    def _build_direct_rl_env_cfg(cfg: LeggedRobotCfg) -> DirectRLEnvCfg:
        """Build a DirectRLEnvCfg from a LeggedRobotCfg.

        This bridges VideoMimic's custom config hierarchy to IsaacLab's required
        DirectRLEnvCfg, so child configs (G1RoughCfg, etc.) remain unchanged.
        """

        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            num_envs = cfg.env.num_envs
            env_spacing = cfg.env.env_spacing

        @configclass
        class _RLEnvCfg(DirectRLEnvCfg):
            sim: SimulationCfg = SimulationCfg(
                dt=cfg.sim.dt,
                substeps=cfg.sim.substeps,
                gravity=cfg.sim.gravity,
            )
            scene: InteractiveSceneCfg = _SceneCfg()
            decimation: int = cfg.control.decimation
            episode_length_s: float = cfg.env.episode_length_s
            observation_space = cfg.env.num_actions  # placeholder; refined after _init_buffers
            action_space = cfg.env.num_actions
            is_finite_horizon = False
            ui_window_class_type = None

        return _RLEnvCfg()

    # ---- DirectRLEnv required methods ----

    def _setup_scene(self):
        """Set up the simulation scene: robot, terrain, contact sensor, lights."""
        from isaaclab.assets import ArticulationCfg
        from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        # Create robot articulation
        asset_path = self._robot_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        robot_articulation_cfg = self._build_articulation_cfg(asset_path)
        self.robot = Articulation(robot_articulation_cfg)

        # Add terrain
        if self._robot_cfg.terrain.mesh_type == 'trimesh':
            assert hasattr(self, 'terrain'), "Terrain object must be set before _setup_scene"
            self._create_trimesh()
        else:
            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(),
            )

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Add robot to scene
        self.scene.articulations["robot"] = self.robot

        # Add contact sensor for all robot bodies
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3,
            update_period=0.0,
        )
        self._contact_sensor = ContactSensor(contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Add light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Initialize sensors
        self.sensors = {}
        self._init_sensors()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions from the policy."""
        actions = actions + self.actions_offset_seed * self.cfg.noise.offset_scales.action

        self.actions_pre_clip = actions

        clip_actions = self.cfg.normalization.clip_actions
        actions_clipped = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        beta = self.cfg.control.beta
        self.actions = beta * actions_clipped + (1 - beta) * self.actions

        if self.cfg.domain_rand.action_delays:
            self.action_queue.add('actions', self.actions)
            self.actions = self.action_queue.query_at_history(self.action_delay_idx, 'actions')

        # Export trajectory data if enabled
        if self.export_trajectory and not torch.all(self.env_episode_done):
            if not self.stepped_before_export:
                self.stepped_before_export = True
            else:
                self._export_trajectory_step()

    def _apply_action(self) -> None:
        """Apply actions to the robot at each physics time-step."""
        if not self.cfg.control.control_type == 'POS':
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.robot.set_joint_effort_target(self.torques)
        else:
            self.dof_pos_targets = self._compute_dof_pos_targets(self.actions).view(self.dof_pos_targets.shape)
            self.robot.set_joint_position_target(self.dof_pos_targets)

    def _get_observations(self) -> dict:
        """Compute and return observations."""
        for i, obs_fn in enumerate(self.observation_functions):
            self.obs_dict[self.observation_names[i]] = obs_fn()

        # Initialize history handler if not already initialized
        if self.history_handler is None and hasattr(self.cfg.env, 'obs_history'):
            from legged_gym.utils.history import HistoryHandler
            obs_dims = {k: v.shape[1:] for k, v in self.obs_dict.items()}
            self.history_handler = HistoryHandler(self.num_envs, self.cfg.env.obs_history, obs_dims, self.device)

        # Update history if handler exists
        if self.history_handler is not None:
            for key in self.cfg.env.obs_history.keys():
                self.history_handler.add(key, self.obs_dict[key])
            for key in self.cfg.env.obs_history.keys():
                self.obs_dict[f'history_{key}'] = self.history_handler.query(key)

        self.obs_dict['teacher'] = self._manual_obs_teacher()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        for key in self.obs_dict.keys():
            if self.obs_dict[key].is_floating_point():
                self.obs_dict[key] = torch.clip(self.obs_dict[key], -clip_obs, clip_obs)

        # Return primary observation for policy
        # Concatenate all obs in cfg.env.obs list for the policy
        obs_list = []
        for name in self.cfg.env.obs:
            if name in self.obs_dict:
                obs_list.append(self.obs_dict[name])
        if obs_list:
            policy_obs = torch.cat(obs_list, dim=-1)
        else:
            policy_obs = self.obs_dict.get('torso', torch.zeros(self.num_envs, 1, device=self.device))

        observations = {"policy": policy_obs}

        # Add privileged observations if available
        if 'torso_privileged' in self.obs_dict:
            observations["critic"] = self.obs_dict['torso_privileged']

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return rewards."""
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.current_reward_value[name] = rew

        label = "total_pre_clip" if self.cfg.rewards.only_positive_rewards else "total"
        self.current_reward_value[label] = self.rew_buf.clone()

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            self.current_reward_value["total"] = self.rew_buf

        # Add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.current_reward_value["termination"] = rew

        return self.rew_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.

        Called by DirectRLEnv.step() before _get_rewards and _get_observations.
        We run the post-step update here so that all three methods have fresh data.
        """
        self._post_step_update()

        died = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1
        )
        died |= torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset specific environments."""
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # Reset robot joint states
        joint_pos = self.default_dof_pos[env_ids] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        joint_vel = torch.zeros(len(env_ids), self.num_dof, device=self.device)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset root states
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        if self.custom_origins:
            default_root_state[:, :3] += self.env_origins[env_ids]
            default_root_state[:, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            default_root_state[:, :3] += self.env_origins[env_ids]

        # Add random base velocities
        default_root_state[:, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        # Resample commands
        self._resample_commands(env_ids)
        self._resample_episodic_randomisations(env_ids)

        # Reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.

        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['Episode/rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["Episode/max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.history_handler is not None:
            self.history_handler.reset(env_ids)

        for sensor_name, sensor in self.sensors.items():
            sensor.reset(env_ids)

    # ---- Initialization helpers ----

    def _build_articulation_cfg(self, asset_path: str) -> ArticulationCfg:
        """Build an IsaacLab ArticulationCfg from the VideoMimic asset config."""
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.from_files import UrdfFileCfg

        asset_cfg = self._robot_cfg.asset

        # Build actuator configs from control stiffness/damping
        actuator_configs = {}
        if self._robot_cfg.control.control_type == 'POS':
            for dof_name, stiffness in self._robot_cfg.control.stiffness.items():
                damping = self._robot_cfg.control.damping[dof_name]
                actuator_configs[dof_name] = ImplicitActuatorCfg(
                    joint_names_expr=[f"*{dof_name}*"],
                    stiffness=stiffness,
                    damping=damping,
                )
        else:
            # Effort mode — all joints get zero stiffness
            actuator_configs["all_joints"] = ImplicitActuatorCfg(
                joint_names_expr=["*"],
                effort_limit_sim=100.0,
                stiffness=0.0,
                damping=0.0,
            )

        # Use UrdfFileCfg for URDF assets, UsdFileCfg for USD assets
        if asset_path.endswith('.urdf') or asset_path.endswith('.xml'):
            spawn_cfg = UrdfFileCfg(
                asset_path=asset_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=asset_cfg.max_linear_velocity,
                    max_angular_velocity=asset_cfg.max_angular_velocity,
                    max_depenetration_velocity=100.0,
                    enable_gyroscopic_forces=True,
                    linear_damping=asset_cfg.linear_damping,
                    angular_damping=asset_cfg.angular_damping,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=asset_cfg.self_collisions == 0,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.001,
                    fix_root_link=asset_cfg.fix_base_link,
                ),
            )
        else:
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=asset_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=asset_cfg.max_linear_velocity,
                    max_angular_velocity=asset_cfg.max_angular_velocity,
                    max_depenetration_velocity=100.0,
                    enable_gyroscopic_forces=True,
                    linear_damping=asset_cfg.linear_damping,
                    angular_damping=asset_cfg.angular_damping,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=asset_cfg.self_collisions == 0,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.001,
                    fix_root_link=asset_cfg.fix_base_link,
                ),
            )

        articulation_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(self._robot_cfg.init_state.pos),
                # Config stores quaternion as xyzw (IsaacGym convention);
                # IsaacLab expects wxyz, so we reorder.
                rot=tuple(self._robot_cfg.init_state.rot[3:4] + self._robot_cfg.init_state.rot[:3]),
                joint_pos=self._robot_cfg.init_state.default_joint_angles,
            ),
            actuators=actuator_configs,
        )

        return articulation_cfg

    def _init_buffers(self):
        """Initialize torch tensors for simulation states."""
        # Access robot data directly (no gymtorch.wrap_tensor needed)
        self.root_states = self.robot.data.root_state_w
        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel
        # Contact forces from ContactSensor (shape: num_envs, num_bodies, 3)
        self.contact_forces = self._contact_sensor.data.net_forces_w

        # Rigid body states
        self.rigid_body_pos = self.robot.data.body_pos_w
        self.rigid_body_vel = self.robot.data.body_lin_vel_w
        self.rigid_body_quat = self.robot.data.body_quat_w

        # Derived quantities
        # Note: IsaacLab uses wxyz quaternion convention
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:, 0:3]
        self.base_lin_vel = self._quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = self._quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.gravity_vec[:, 2] = -1.0
        self.forward_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.forward_vec[:, 0] = 1.0
        self.projected_gravity = self._quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Body/DOF names and indices (from Articulation)
        self.body_names = list(self.robot.data.body_names)
        self.dof_names = list(self.robot.data.joint_names)
        self.num_bodies = len(self.body_names)
        self.num_dof = self.robot.num_joints
        self.num_dofs = self.num_dof

        # Build mapping from robot body names → contact sensor body indices.
        # ContactSensor body order may differ from Articulation body order,
        # so we use find_bodies() to resolve indices in the sensor's frame.
        self._contact_body_names = list(self._contact_sensor.body_names)

        # Feet indices (in contact sensor body order)
        self.feet_names = [s for s in self.body_names if self._robot_cfg.asset.foot_name in s]
        feet_sensor_ids, _ = self._contact_sensor.find_bodies(
            [f".*{n}.*" for n in self.feet_names]
        )
        self.feet_indices = torch.tensor(feet_sensor_ids, dtype=torch.long, device=self.device)

        # Penalized and termination contact indices (in contact sensor body order)
        penalized_contact_names = []
        for name in self._robot_cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self._robot_cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        if penalized_contact_names:
            penalised_ids, _ = self._contact_sensor.find_bodies(
                [f".*{n}.*" for n in penalized_contact_names]
            )
            self.penalised_contact_indices = torch.tensor(penalised_ids, dtype=torch.long, device=self.device)
        else:
            self.penalised_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)

        if termination_contact_names:
            termination_ids, _ = self._contact_sensor.find_bodies(
                [f".*{n}.*" for n in termination_contact_names]
            )
            self.termination_contact_indices = torch.tensor(termination_ids, dtype=torch.long, device=self.device)
        else:
            self.termination_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)

        # Joint limits
        self.dof_pos_limits = self.robot.data.soft_joint_pos_limits
        self.dof_vel_limits = self.robot.data.joint_vel_limits
        self.torque_limits = self.robot.data.joint_effort_limits

        self.num_actions = self.cfg.env.num_actions

        # Control buffers
        if self.cfg.control.control_type == 'POS':
            self.dof_pos_targets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        else:
            self.torques = torch.zeros(self.num_envs, self.cfg.env.num_actions, dtype=torch.float, device=self.device)

        self.p_gains = torch.zeros(self.cfg.env.num_actions, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.cfg.env.num_actions, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.env.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        # Commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device
        )

        # Contact tracking
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)

        # Default DOF positions
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles.get(name, 0.0)
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Environment origins
        self._get_env_origins()

        # Base init state
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = torch.tensor(base_init_state_list, dtype=torch.float, device=self.device)

        # Randomization buffers
        self._init_randomisation_buffers()

        # Observation dict
        self.obs_dict = {}

    def _parse_cfg(self, cfg):
        self.dt = cfg.control.decimation * cfg.sim.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.command_ranges = class_to_dict(cfg.commands.ranges)

        # max_episode_length_s and max_episode_length are read-only properties
        # on DirectRLEnv that derive from cfg.episode_length_s.

        if hasattr(cfg.domain_rand, 'push_interval_s'):
            cfg.domain_rand.push_interval = math.ceil(cfg.domain_rand.push_interval_s / self.dt)

    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(int(num_rows)), torch.arange(int(num_cols)), indexing='ij')
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    # ---- Observation methods ----

    def _obs_torso(self):
        return torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

    def _obs_torso_privileged(self):
        return torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

    def _obs_sensor(self, sensor_name):
        if sensor_name not in self.sensors:
            raise ValueError(f"Sensor '{sensor_name}' not initialized but observation requested")
        sensor = self.sensors[sensor_name]
        if sensor.depth_map.dtype == torch.uint8:
            return sensor.depth_map.float() / 255.0
        return sensor.depth_map

    def _manual_obs_teacher(self):
        return torch.zeros(self.num_envs, 0, device=self.device)

    # ---- Reward functions ----

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_accel(self):
        accel = (self.actions - 2 * self.last_actions + self.last_last_actions)
        return torch.sum(torch.square(accel), dim=1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        return self.reset_terminated.float() * (~self.reset_time_outs).float()

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # ---- Control ----

    def _compute_torques(self, actions):
        if self.cfg.domain_rand.control_delays:
            self.control_queue.add('controls', actions)
            actions = self.control_queue.query_at_history(self.control_delay_idx, 'controls')

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        p_gains = self.p_gains * (1 + self.p_gain_rand_seed * self.p_gain_rand_scale) if self.p_gain_rand else self.p_gains
        d_gains = self.d_gains * (1 + self.d_gain_rand_seed * self.d_gain_rand_scale) if self.d_gain_rand else self.d_gains

        if control_type == "P":
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.dt
        elif control_type == "T":
            torques = actions_scaled
        elif control_type == "DIRECT":
            torques = self.p_gains * (actions - self.dof_pos) - self.d_gains * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        if self.enable_torque_rfi:
            torques += self.torque_rfi_seed * self.torque_limits * self.torque_rfi_rand_scale

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_dof_pos_targets(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            return actions_scaled + self.default_dof_pos
        elif control_type == "DIRECT":
            return actions
        elif control_type == "DEEPMIMIC_DELTA":
            return actions_scaled + self.target_motors
        else:
            return None

    # ---- Callbacks ----

    def _post_step_update(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = self._quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if hasattr(self, 'viser_viz') and self.viser_viz.manual_control.value:
            self.commands[:, :3] = 0.
            if self.viser_viz.move_forward.value:
                self.commands[:, 0] = 1.0
            if self.viser_viz.move_back.value:
                self.commands[:, 0] = -1.0
            if self.viser_viz.move_left.value:
                self.commands[:, 1] = 1.0
            if self.viser_viz.move_right.value:
                self.commands[:, 1] = -1.0
            if self.viser_viz.rotate_left.value:
                self.commands[:, 2] = 1.0
            if self.viser_viz.rotate_right.value:
                self.commands[:, 2] = -1.0

        # Update sensors
        for sensor_name, sensor in self.sensors.items():
            sensor.update_buffers(episode_step=self.episode_length_buf, env_ids=...)

        # Push robots
        if self.cfg.domain_rand.push_robots or (self.use_viser_viz and self.viser_viz.enable_push_robots):
            self._push_robots()

        # Update last buffers
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _init_randomisation_buffers(self):
        self.enable_torque_rfi = self.cfg.domain_rand.torque_rfi_rand
        self.torque_rfi_rand_scale = self.cfg.domain_rand.torque_rfi_rand_scale
        self.torque_rfi_seed = torch.randn_like(self.dof_pos)
        self.p_gain_rand = self.cfg.domain_rand.p_gain_rand
        self.p_gain_rand_scale = self.cfg.domain_rand.p_gain_rand_scale
        self.d_gain_rand = self.cfg.domain_rand.d_gain_rand
        self.d_gain_rand_scale = self.cfg.domain_rand.d_gain_rand_scale
        self.p_gain_rand_seed = torch.randn_like(self.dof_pos)
        self.d_gain_rand_seed = torch.randn_like(self.dof_pos)
        self.actions_offset_seed = torch.zeros_like(self.dof_pos)

        if self.cfg.domain_rand.control_delays:
            self.control_queue = HistoryHandler(self.num_envs,
                                                {'controls': self.cfg.domain_rand.control_delay_max + 1},
                                                {'controls': (self.num_dof,)},
                                                self.device)
            self.control_delay_idx = torch.randint(self.cfg.domain_rand.control_delay_min,
                                                    self.cfg.domain_rand.control_delay_max + 1, (self.num_envs,), device=self.device)

        if self.cfg.domain_rand.action_delays:
            self.action_queue = HistoryHandler(self.num_envs,
                                                {'actions': self.cfg.domain_rand.action_delay_max + 1},
                                                {'actions': (self.num_dof,)},
                                                self.device)
            self.action_delay_idx = torch.randint(self.cfg.domain_rand.action_delay_min,
                                                    self.cfg.domain_rand.action_delay_max + 1, (self.num_envs,), device=self.device)

    def _resample_episodic_randomisations(self, env_ids):
        self.torque_rfi_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.p_gain_rand_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.d_gain_rand_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])
        self.actions_offset_seed[env_ids] = torch.randn_like(self.dof_pos[env_ids])

        if self.cfg.domain_rand.control_delays:
            self.control_delay_idx[env_ids] = torch.randint(self.cfg.domain_rand.control_delay_min,
                                                    self.cfg.domain_rand.control_delay_max + 1, (len(env_ids),), device=self.device)
            self.control_queue.reset(env_ids)

        if self.cfg.domain_rand.action_delays:
            self.action_delay_idx[env_ids] = torch.randint(self.cfg.domain_rand.action_delay_min,
                                                    self.cfg.domain_rand.action_delay_max + 1, (len(env_ids),), device=self.device)
            self.action_queue.reset(env_ids)

    def _push_robots(self):
        if self.use_viser_viz and not self.viser_viz.enable_push_robots.value:
            return

        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return

        max_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        max_vel_z = 0.5
        if self.use_viser_viz:
            max_vel_xy *= self.viser_viz.push_force_scale.value
            max_vel_z = self.viser_viz.push_force_z_scale.value

        root_velocities = self.robot.data.root_lin_vel_w.clone()
        root_velocities[:, 0:2] = torch_rand_float(-max_vel_xy, max_vel_xy, (self.num_envs, 2), device=self.device)
        root_velocities[:, 2:3] = torch_rand_float(-max_vel_z, max_vel_z, (self.num_envs, 1), device=self.device)
        # write_root_velocity_to_sim expects (lin_vel, ang_vel) as separate args or combined (6,)
        root_vel_6d = torch.cat([root_velocities, self.robot.data.root_ang_vel_w.clone()], dim=-1)
        self.robot.write_root_velocity_to_sim(root_vel_6d, env_ids=push_env_ids)

    def update_command_curriculum(self, env_ids):
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, '_reward_' + name))

        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self.reward_scales.keys()}
        self.current_reward_value = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self.reward_scales.keys()}

    def _prepare_observation_function(self):
        self.observation_functions = []
        self.observation_names = []

        for name in self.cfg.env.obs:
            if name in self.sensors:
                self.observation_functions.append(lambda sensor_name=name: self._obs_sensor(sensor_name))
                self.observation_names.append(name)
            else:
                func_name = f'_obs_{name}'
                if hasattr(self, func_name):
                    self.observation_functions.append(getattr(self, func_name))
                else:
                    if name == 'teacher':
                        self.observation_functions.append(lambda: torch.zeros(self.num_envs, 415, device=self.device))
                    else:
                        raise ValueError(f"Missing observation method {func_name} for observation {name}")
                self.observation_names.append(name)

    # ---- Sensor initialization ----

    def _init_sensors(self):
        if not hasattr(self, 'terrain'):
            assert len(self._robot_cfg.sensors.sensor_cfgs) == 0, "No terrain found, but sensors are configured"
            return

        terrain_vertices = getattr(self.terrain, 'vertices', None)
        terrain_triangles = getattr(self.terrain, 'triangles', None)

        for sensor_index, sensor_cfg in enumerate(self._robot_cfg.sensors.sensor_cfgs):
            if not sensor_cfg.enabled:
                continue

            if not sensor_cfg.name:
                same_type_count = sum(1 for i, s in enumerate(self._robot_cfg.sensors.sensor_cfgs)
                                      if i < sensor_index and s.type == sensor_cfg.type and s.enabled)
                sensor_cfg.name = f"{sensor_cfg.type}_{same_type_count}" if same_type_count > 0 else sensor_cfg.type

            if sensor_cfg.type == 'depth_camera':
                self._init_depth_camera(sensor_cfg, terrain_vertices, terrain_triangles)
            elif sensor_cfg.type == 'heightfield':
                self._init_heightfield(sensor_cfg, terrain_vertices, terrain_triangles)
            elif sensor_cfg.type == 'multi_link_height':
                self._init_multi_link_height(sensor_cfg, terrain_vertices, terrain_triangles)

    def _init_depth_camera(self, sensor_cfg, terrain_vertices, terrain_triangles):
        from legged_gym.utils.raycaster.sensors import DepthCameraSensor
        from legged_gym.utils.raycaster.sensor_cfg import DepthCameraSensorCfg

        camera_device_cfg = DepthCameraSensorCfg(
            device=self.device,
            width=sensor_cfg.width,
            height=sensor_cfg.height,
            downsample_factor=sensor_cfg.downsample_factor,
            body_name=sensor_cfg.body_name,
            max_distance=sensor_cfg.max_distance,
            only_heading=sensor_cfg.only_heading,
            intrinsic_matrix=sensor_cfg.intrinsic_matrix
        )
        self.sensors[sensor_cfg.name] = DepthCameraSensor(
            self, camera_device_cfg, terrain_vertices, terrain_triangles
        )

    def _init_heightfield(self, sensor_cfg, terrain_vertices, terrain_triangles):
        from legged_gym.utils.raycaster.sensors import HeightfieldSensor
        from legged_gym.utils.raycaster.sensor_cfg import HeightfieldSensorCfg

        sensor_args = {k: v for k, v in sensor_cfg.to_dict().items() if k in HeightfieldSensorCfg(device=None).__dict__ and k != 'type'}
        heightfield_device_cfg = HeightfieldSensorCfg(device=self.device, **sensor_args)
        self.sensors[sensor_cfg.name] = HeightfieldSensor(
            self, heightfield_device_cfg, terrain_vertices, terrain_triangles
        )

    def _init_multi_link_height(self, sensor_cfg, terrain_vertices, terrain_triangles):
        from legged_gym.utils.raycaster.sensors import MultiLinkHeightSensor
        from legged_gym.utils.raycaster.sensor_cfg import MultiLinkHeightSensorCfg

        multi_link_device_cfg = MultiLinkHeightSensorCfg(
            device=self.device,
            body_name=sensor_cfg.body_name,
            max_distance=sensor_cfg.max_distance,
            only_heading=sensor_cfg.only_heading,
            link_names=sensor_cfg.link_names,
            use_float=sensor_cfg.use_float
        )
        self.sensors[sensor_cfg.name] = MultiLinkHeightSensor(
            self, multi_link_device_cfg, terrain_vertices, terrain_triangles
        )

    def _validate_sensor_observations(self):
        if not hasattr(self._robot_cfg, 'sensors'):
            return

        sensor_names = set()
        for sensor_cfg in self._robot_cfg.sensors.sensor_cfgs:
            if sensor_cfg.enabled:
                sensor_names.add(sensor_cfg.name or sensor_cfg.type)

        for obs in self._robot_cfg.env.obs:
            if obs not in sensor_names and obs in {'depth_camera', 'heightfield'} or obs.startswith(('depth_camera_', 'heightfield_')):
                print(f"Warning: Observation '{obs}' appears to be a sensor but is not configured in sensors.sensor_cfgs")

    # ---- Terrain creation ----

    def _create_trimesh(self):
        """Add triangle mesh terrain to the simulation using IsaacLab's TerrainImporter."""
        import trimesh
        from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg

        # Get pre-computed vertices and triangles from the DeepMimicTerrain object
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles

        # Create a trimesh.Trimesh object
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Import the terrain mesh into the simulator
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=None,
            visual_material=GroundPlaneCfg().visual_material,
        )
        terrain_importer = TerrainImporter(terrain_cfg)
        terrain_importer.import_mesh("deepmimic_terrain", terrain_mesh)

        # Configure env origins from terrain offsets
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)

    # ---- Utility methods ----

    @staticmethod
    def _quat_rotate_inverse(q, v):
        """Rotate vector v by inverse of quaternion q. q is in wxyz convention."""
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

        # q * v * q_conj (inverse rotation)
        t0 = -qx * vx - qy * vy - qz * vz
        t1 = qw * vx + qy * vz - qz * vy
        t2 = qw * vy - qx * vz + qz * vx
        t3 = qw * vz + qx * vy - qy * vx

        result = torch.stack([
            -t0 * qx + t1 * qw - t2 * qz + t3 * qy,
            -t0 * qy + t2 * qw + t1 * qz - t3 * qx,
            -t0 * qz + t3 * qw - t1 * qy + t2 * qx,
        ], dim=-1)
        return result

    @staticmethod
    def _quat_apply(q, v):
        """Rotate vector v by quaternion q. q is in wxyz convention."""
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

        t0 = -qx * vx - qy * vy - qz * vz
        t1 = qw * vx + qy * vz - qz * vy
        t2 = qw * vy - qx * vz + qz * vx
        t3 = qw * vz + qx * vy - qy * vx

        result = torch.stack([
            t0 * qx + t1 * qw + t2 * qz - t3 * qy,
            t0 * qy + t2 * qw - t1 * qz + t3 * qx,
            t0 * qz + t3 * qw + t1 * qy - t2 * qx,
        ], dim=-1)
        return result

    def get_sensor(self, sensor_name):
        return self.sensors.get(sensor_name, None)

    def get_sensor_data(self, sensor_name):
        sensor = self.get_sensor(sensor_name)
        if sensor:
            return sensor.depth_map
        return None

    @property
    def env_root_pos(self):
        return self.root_states[:, 0:3]

    @property
    def env_rigid_body_pos(self):
        return self.rigid_body_pos

    # ---- Trajectory export ----

    def _export_trajectory_step(self):
        if self.episode_length_buf[0] % 100 == 0:
            print(f'Working on export...')
        for env_id in range(self.num_envs):
            if self.env_episode_done[env_id]:
                continue

            if self.reset_buf[env_id]:
                self._save_trajectory_data(env_id)
                self.env_episode_done[env_id] = True
                self.num_envs_completed += 1
                print(f"Saved trajectory for environment {env_id} ({self.num_envs_completed}/{self.num_envs} completed)")
                if self.num_envs_completed == self.num_envs:
                    print(f"\nAll {self.num_envs} environments have completed their first episode!")
                    print(f"Trajectory data saved in: {self.export_dir}/")
            else:
                if len(self.trajectory_data[env_id]['joint_names']) == 0:
                    self.trajectory_data[env_id]['joint_names'] = self.dof_names
                    self.trajectory_data[env_id]['link_names'] = self.body_names
                    self.trajectory_data[env_id]['joint_targets'] = []
                    self.trajectory_data[env_id]['actions'] = []

                    if hasattr(self, 'feet_indices'):
                        foot_names = [self.body_names[idx] for idx in self.feet_indices]
                        for foot_idx in range(len(self.feet_indices)):
                            foot_name = self.body_names[self.feet_indices[foot_idx]]
                            self.trajectory_data[env_id]['contacts'][foot_name] = []

                    if hasattr(self, 'replay_data_loader'):
                        clip_index = self.replay_data_loader.episode_indices[env_id].item()
                        trajectory_path = self.replay_data_loader.pkl_paths[clip_index]
                        trajectory_name = os.path.splitext(os.path.basename(trajectory_path))[0]
                        self.trajectory_data[env_id]['trajectory_name'] = trajectory_name

                joint_targets = self._compute_dof_pos_targets(self.actions[env_id:env_id + 1])
                if joint_targets is not None:
                    self.trajectory_data[env_id]['joint_targets'].append(joint_targets[0].cpu().numpy())

                self.trajectory_data[env_id]['joints'].append(self.dof_pos[env_id].cpu().numpy())
                self.trajectory_data[env_id]['root_quat'].append(self.base_quat[env_id].cpu().numpy())
                self.trajectory_data[env_id]['root_pos'].append(self.env_root_pos[env_id].cpu().numpy())
                self.trajectory_data[env_id]['actions'].append(self.actions[env_id].cpu().numpy())

                if len(self.trajectory_data[env_id]['obs']) == 0:
                    self.trajectory_data[env_id]['obs'] = {k: [] for k in self.obs_dict.keys()}
                for key, value in self.obs_dict.items():
                    self.trajectory_data[env_id]['obs'][key].append(value[env_id].cpu().numpy())

                link_pos = []
                link_quat = []
                for i in range(len(self.body_names)):
                    pos = self.env_rigid_body_pos[env_id, i].cpu().numpy()
                    quat = self.rigid_body_quat[env_id, i].cpu().numpy()
                    link_pos.append(pos)
                    link_quat.append(quat)
                self.trajectory_data[env_id]['link_pos'].append(link_pos)
                self.trajectory_data[env_id]['link_quat'].append(link_quat)

                if hasattr(self, 'feet_indices'):
                    for foot_idx in range(len(self.feet_indices)):
                        foot_name = self.body_names[self.feet_indices[foot_idx]]
                        contact = bool(torch.norm(self.contact_forces[env_id, self.feet_indices[foot_idx], :]) > 1.0)
                        self.trajectory_data[env_id]['contacts'][foot_name].append(contact)

    def _save_trajectory_data(self, env_id):
        import pickle

        export_data = {
            'joint_names': self.trajectory_data[env_id]['joint_names'],
            'joints': np.array(self.trajectory_data[env_id]['joints']),
            'joint_targets': np.array(self.trajectory_data[env_id]['joint_targets']) if len(self.trajectory_data[env_id]['joint_targets']) > 0 else None,
            'root_quat': np.array(self.trajectory_data[env_id]['root_quat']),
            'root_pos': np.array(self.trajectory_data[env_id]['root_pos']),
            'link_names': self.trajectory_data[env_id]['link_names'],
            'link_pos': np.array(self.trajectory_data[env_id]['link_pos']),
            'link_quat': np.array(self.trajectory_data[env_id]['link_quat']),
            'actions': np.array(self.trajectory_data[env_id]['actions']),
            'contacts': {name: np.array(contacts) for name, contacts in self.trajectory_data[env_id]['contacts'].items()},
            'stiffness': {name: float(self.p_gains[i].cpu().numpy()) for i, name in enumerate(self.dof_names)},
            'damping': {name: float(self.d_gains[i].cpu().numpy()) for i, name in enumerate(self.dof_names)},
            'obs': {key: np.array(value) for key, value in self.trajectory_data[env_id]['obs'].items()}
        }

        if self.trajectory_data[env_id]['trajectory_name'] is not None:
            export_data['trajectory_name'] = self.trajectory_data[env_id]['trajectory_name']
            filename = f"env_{env_id}_{self.trajectory_data[env_id]['trajectory_name']}.pkl"
        else:
            filename = f"env_{env_id}.pkl"

        filepath = os.path.join(self.export_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
        print(f"Trajectory data saved to {filepath}")


# ---- Utility functions ----

def torch_rand_float(lower, upper, shape, device):
    """Generate random floats between lower and upper."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def to_torch(x, device, requires_grad=False):
    """Convert to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float, device=device, requires_grad=requires_grad)
