import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import parse_unknown_args

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime


def place_trace(args, unknown):
    env_overrides, train_overrides = parse_unknown_args(unknown)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    if 'deepmimic' in args.task:
        env_cfg.terrain.n_duplicate_terrains_x = 2
        env_cfg.terrain.n_duplicate_terrains_y = 2
        env_cfg.deepmimic.viz_replay = True
        env_cfg.deepmimic.viz_replay_sync_robot = False

        if env_cfg.deepmimic.viz_replay_sync_robot:
            env_cfg.control.stiffness = {k: 0.0 for k in env_cfg.control.stiffness}
            env_cfg.control.damping = {k: 0.0 for k in env_cfg.control.damping}
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
    env_cfg.terrain.n_rows = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, env_overrides=env_overrides)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True

    args.use_wandb = False
    if not train_cfg.runner.resume:
        print(f'\n\n\nWARNING NOT RESUMING\n\n\n')
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, train_overrides=train_overrides)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Lists to store traces
    action_trace = []
    dof_pos_trace = []
    dof_vel_trace = []
    feet_pos_trace = []
    contact_trace = []
    target_contact_trace = []
    smoothness_violations = []  # Add this to store smoothness violations
    
    # Run simulation and collect traces for one episode
    step = 0
    done = False
    last_contact_state = None
    while not done:
        actions = policy(obs.detach())
            
        # Store traces
        action_trace.append(actions[0].detach().cpu().numpy())
        dof_pos_trace.append(env.dof_pos[0].detach().cpu().numpy())
        dof_vel_trace.append(env.dof_vel[0].detach().cpu().numpy())
        feet_pos_trace.append(env.feet_pos[0].detach().cpu().numpy())
        
        # Store contact states
        current_contacts = (env.contact_forces[0, env.feet_indices, 2] > 1.).cpu().numpy()
        contact_trace.append(current_contacts)
        target_contacts = env.target_contacts[0].cpu().numpy()
        target_contact_trace.append(target_contacts)
        
        # Calculate smoothness violations
        if last_contact_state is not None:
            contact_changed = current_contacts != last_contact_state
            target_changed = target_contacts != target_contact_trace[-2]  # Compare with previous target
            current_is_wrong = current_contacts != target_contacts
            invalid_changes = contact_changed & ~target_changed & current_is_wrong
            smoothness_violations.append(invalid_changes)
        else:
            smoothness_violations.append(np.zeros_like(current_contacts, dtype=bool))
        
        last_contact_state = current_contacts.copy()
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        done = dones[0].item()  # Check if first environment is done
        
        step += 1
        if step % 100 == 0:
            print(f"Step {step}")

    print(f"Episode completed in {step} steps")

    # Convert to numpy arrays
    action_trace = np.array(action_trace)
    dof_pos_trace = np.array(dof_pos_trace)
    dof_vel_trace = np.array(dof_vel_trace)
    feet_pos_trace = np.array(feet_pos_trace)
    contact_trace = np.array(contact_trace)
    target_contact_trace = np.array(target_contact_trace)
    smoothness_violations = np.array(smoothness_violations)
    
    # Calculate feet distance (distance between left and right foot)
    feet_distance = np.linalg.norm(feet_pos_trace[:, 0] - feet_pos_trace[:, 1], axis=1)
    
    # Create timestamp and directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'traces')
    run_dir = os.path.join(base_dir, f'trace_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save raw data
    np.save(os.path.join(run_dir, 'action_trace.npy'), action_trace)
    np.save(os.path.join(run_dir, 'dof_pos_trace.npy'), dof_pos_trace)
    np.save(os.path.join(run_dir, 'dof_vel_trace.npy'), dof_vel_trace)
    np.save(os.path.join(run_dir, 'feet_pos_trace.npy'), feet_pos_trace)
    np.save(os.path.join(run_dir, 'contact_trace.npy'), contact_trace)
    np.save(os.path.join(run_dir, 'target_contact_trace.npy'), target_contact_trace)
    np.save(os.path.join(run_dir, 'smoothness_violations.npy'), smoothness_violations)
    
    # Plot traces
    time_steps = np.arange(len(action_trace))
    
    # Define distinct color palette
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    
    # Group joints by body part
    leg_joints = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
    waist_joints = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
    arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
                  'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
    
    joint_groups = [
        ('Leg Joints', leg_joints),
        ('Waist Joints', waist_joints),
        ('Arm Joints', arm_joints)
    ]
    
    # Plot actions in subplots
    fig, axes = plt.subplots(len(joint_groups), 1, figsize=(15, 20))
    fig.suptitle('Action Traces by Body Part', fontsize=16)
    
    for idx, (group_name, group_joints) in enumerate(joint_groups):
        ax = axes[idx]
        color_idx = 0
        for joint_name in group_joints:
            if joint_name in env.dof_names:
                joint_idx = env.dof_names.index(joint_name)
                ax.plot(time_steps, action_trace[:, joint_idx], 
                       label=joint_name, 
                       color=colors[color_idx % len(colors)],
                       linewidth=2)
                color_idx += 1
        ax.set_title(f'{group_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'action_trace_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Calculate action rates (first derivative) and accelerations (second derivative)
    action_rates = np.diff(action_trace, axis=0)
    action_accelerations = np.diff(action_rates, axis=0)
    
    # Plot action rates by body part
    fig, axes = plt.subplots(len(joint_groups), 1, figsize=(15, 20))
    fig.suptitle('Action Rates (Action Delta per Step) by Body Part', fontsize=16)
    
    for idx, (group_name, group_joints) in enumerate(joint_groups):
        ax = axes[idx]
        color_idx = 0
        for joint_name in group_joints:
            if joint_name in env.dof_names:
                joint_idx = env.dof_names.index(joint_name)
                ax.plot(time_steps[1:], action_rates[:, joint_idx], 
                       label=joint_name, 
                       color=colors[color_idx % len(colors)],
                       linewidth=2)
                color_idx += 1
        ax.set_title(f'{group_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Rate (Δ/step)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'action_rates_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot action accelerations by body part
    fig, axes = plt.subplots(len(joint_groups), 1, figsize=(15, 20))
    fig.suptitle('Action Accelerations (Action Double Delta per Step) by Body Part', fontsize=16)
    
    for idx, (group_name, group_joints) in enumerate(joint_groups):
        ax = axes[idx]
        color_idx = 0
        for joint_name in group_joints:
            if joint_name in env.dof_names:
                joint_idx = env.dof_names.index(joint_name)
                ax.plot(time_steps[2:], action_accelerations[:, joint_idx], 
                       label=joint_name, 
                       color=colors[color_idx % len(colors)],
                       linewidth=2)
                color_idx += 1
        ax.set_title(f'{group_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Acceleration (Δ²/step²)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'action_accelerations_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot DOF positions with the same grouping
    fig, axes = plt.subplots(len(joint_groups), 1, figsize=(15, 20))
    fig.suptitle('DOF Position Traces by Body Part', fontsize=16)
    
    for idx, (group_name, group_joints) in enumerate(joint_groups):
        ax = axes[idx]
        color_idx = 0
        for joint_name in group_joints:
            if joint_name in env.dof_names:
                joint_idx = env.dof_names.index(joint_name)
                ax.plot(time_steps, dof_pos_trace[:, joint_idx], 
                       label=joint_name, 
                       color=colors[color_idx % len(colors)],
                       linewidth=2)
                color_idx += 1
        ax.set_title(f'{group_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('DOF Position')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'dof_pos_trace_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot DOF velocities with the same grouping
    fig, axes = plt.subplots(len(joint_groups), 1, figsize=(15, 20))
    fig.suptitle('DOF Velocity Traces by Body Part', fontsize=16)
    
    for idx, (group_name, group_joints) in enumerate(joint_groups):
        ax = axes[idx]
        color_idx = 0
        for joint_name in group_joints:
            if joint_name in env.dof_names:
                joint_idx = env.dof_names.index(joint_name)
                ax.plot(time_steps, dof_vel_trace[:, joint_idx], 
                       label=joint_name, 
                       color=colors[color_idx % len(colors)],
                       linewidth=2)
                color_idx += 1
        ax.set_title(f'{group_name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('DOF Velocity')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'dof_vel_trace_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot feet positions (keep this as is since it's already clear)
    plt.figure(figsize=(15, 8))
    feet_labels = ['Left foot', 'Right foot']
    for foot_idx in range(2):
        for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
            plt.plot(time_steps, feet_pos_trace[:, foot_idx, axis_idx], 
                    label=f'{feet_labels[foot_idx]} {axis}',
                    color=colors[foot_idx * 3 + axis_idx],
                    linewidth=2)
    plt.title('Feet Positions')
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'feet_pos_trace_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot feet distance
    plt.figure(figsize=(15, 8))
    plt.plot(time_steps, feet_distance, label='Inter-feet distance', color=colors[0], linewidth=2)
    plt.title('Distance Between Feet')
    plt.xlabel('Time Step')
    plt.ylabel('Distance (m)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'feet_distance_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Plot contact states vertically separated
    plt.figure(figsize=(15, 8))
    plt.title('Foot Contact States (Solid = Contact, Empty = No Contact)', fontsize=14)
    
    # Define vertical offsets for each line
    offsets = {
        'left_current': 4,
        'right_current': 3,
        'left_target': 2,
        'right_target': 1
    }
    
    # Plot current contacts
    plt.fill_between(time_steps, offsets['left_current'], 
                    offsets['left_current'] + contact_trace[:, 0],
                    label='Left foot (Current)', color='#e41a1c', alpha=0.7)
    plt.fill_between(time_steps, offsets['right_current'],
                    offsets['right_current'] + contact_trace[:, 1],
                    label='Right foot (Current)', color='#377eb8', alpha=0.7)
    
    # Plot target contacts with hatching
    for i in range(len(time_steps)):
        if target_contact_trace[i, 0]:  # Left foot target
            plt.fill_between([i-0.5, i+0.5], [offsets['left_target'], offsets['left_target']], 
                           [offsets['left_target']+1, offsets['left_target']+1],
                           color='#e41a1c', alpha=0.3, hatch='///')
        if target_contact_trace[i, 1]:  # Right foot target
            plt.fill_between([i-0.5, i+0.5], [offsets['right_target'], offsets['right_target']], 
                           [offsets['right_target']+1, offsets['right_target']+1],
                           color='#377eb8', alpha=0.3, hatch='///')
    
    # Plot smoothness violations
    for i in range(len(time_steps)):
        if smoothness_violations[i, 0]:  # Left foot violation
            plt.plot(i, offsets['left_current'] + 0.5, 'k*', markersize=10, 
                    label='Smoothness violation' if i == 0 else "")
        if smoothness_violations[i, 1]:  # Right foot violation
            plt.plot(i, offsets['right_current'] + 0.5, 'k*', markersize=10,
                    label='Smoothness violation' if i == 0 and not smoothness_violations[i, 0] else "")
    
    # Add empty bars to legend for target contacts
    plt.fill_between([], [], [], label='Left foot (Target)', 
                    color='#e41a1c', alpha=0.3, hatch='///')
    plt.fill_between([], [], [], label='Right foot (Target)', 
                    color='#377eb8', alpha=0.3, hatch='///')
    
    plt.xlabel('Time Step')
    plt.ylabel('Contact State')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set y-axis ticks and labels
    plt.yticks([1.5, 2.5, 3.5, 4.5], 
              ['Right foot\n(Target)', 'Left foot\n(Target)', 
               'Right foot\n(Current)', 'Left foot\n(Current)'])
    
    # Set x-axis ticks to show every 100th step
    tick_spacing = 100
    plt.xticks(np.arange(0, len(time_steps), tick_spacing), 
               np.arange(0, len(time_steps), tick_spacing))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'contact_states_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Traces saved in {run_dir}")


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args, unknown = get_args()
    place_trace(args, unknown) 