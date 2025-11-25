from legged_gym.utils.configclass import configclass
from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
    LeggedRobotEnvCfg,
    LeggedRobotTerrainCfg,
    LeggedRobotInitStateCfg,
    LeggedRobotControlCfg,
    LeggedRobotAssetCfg,
    LeggedRobotRewardsCfg,
    LeggedRobotPolicyCfg,
    LeggedRobotAlgorithmCfg,
    LeggedRobotRunnerCfg,
    LeggedRobotDomainRandCfg,
    LeggedRobotSimCfg,
)

@configclass
class G1RoughInitStateCfg(LeggedRobotInitStateCfg):
    pos = [0.0, 0.0, 0.85]  # x,y,z [m]
    default_joint_angles = {  # = target angles [rad] when action = 0.0
        'left_hip_yaw_joint': 0.,
        'left_hip_roll_joint': 0,
        'left_hip_pitch_joint': -0.1,
        'left_knee_joint': 0.3,
        'left_ankle_pitch_joint': -0.2,
        'left_ankle_roll_joint': 0,
        'right_hip_yaw_joint': 0.,
        'right_hip_roll_joint': 0,
        'right_hip_pitch_joint': -0.1,
        'right_knee_joint': 0.3,
        'right_ankle_pitch_joint': -0.2,
        'right_ankle_roll_joint': 0,
        'torso_joint': 0.,
        'waist_yaw_joint': 0.,
        'waist_pitch_joint': 0.,
        'waist_roll_joint': 0.,
        'left_shoulder_pitch_joint': 0.,
        'left_shoulder_roll_joint': 0.,
        'left_shoulder_yaw_joint': 0.,
        'right_shoulder_pitch_joint': 0.,
        'right_shoulder_roll_joint': 0.,
        'right_shoulder_yaw_joint': 0.,
        'left_elbow_joint': 0.,
        'left_wrist_joint': 0.,
        'right_elbow_joint': 0.,
        'right_wrist_joint': 0.,
        'left_wrist_roll_joint': 0.,
        'left_wrist_pitch_joint': 0.,
        'left_wrist_yaw_joint': 0.,
        'right_wrist_roll_joint': 0.,
        'right_wrist_pitch_joint': 0.,
        'right_wrist_yaw_joint': 0.,
    }

@configclass
class G1RoughEnvCfg(LeggedRobotEnvCfg):
    # num_actions = 29
    num_actions = 12
    num_observations = 3 + 3 + 3 + num_actions * 3 + 2
    num_privileged_obs = num_observations + 3
    obs = ['torso', 'torso_privileged']

@configclass
class G1RoughDomainRandCfg(LeggedRobotDomainRandCfg):
    randomize_friction = True
    friction_range = [0.1, 1.25]
    randomize_base_mass = True
    added_mass_range = [-1., 3.]
    push_robots = True
    push_interval_s = 5
    max_push_vel_xy = 1.5

@configclass
class G1RoughControlCfg(LeggedRobotControlCfg):
    control_type = 'P'
    stiffness = {
        'hip_yaw': 100,
        'hip_roll': 100,
        'hip_pitch': 100,
        'knee': 150,
        'ankle': 40,
        'waist': 75,
        'shoulder': 75,
        'elbow': 75,
        'wrist': 2.,
    }  # [N*m/rad]
    damping = {
        'hip_yaw': 2,
        'hip_roll': 2,
        'hip_pitch': 2,
        'knee': 4,
        'ankle': 2,
        'waist': 2,
        'shoulder': 2,
        'elbow': 2,
        'wrist': 0.1,
    }  # [N*m*s/rad]
    action_scale = 0.25
    decimation = 4

@configclass
class G1RoughSimCfg(LeggedRobotSimCfg):
    dt = 1 / (60. * 4.)  # 0.0061666..

@configclass
class G1RoughAssetCfg(LeggedRobotAssetCfg):
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
    name = "g1"
    foot_name = "ankle_roll"
    penalize_contacts_on = ["hip", "knee"]
    terminate_after_contacts_on = ["pelvis"]
    self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    flip_visual_attachments = False

@configclass
class G1RoughRewardsCfg(LeggedRobotRewardsCfg):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.78

    @configclass
    class Scales(LeggedRobotRewardsCfg.Scales):
        # tracking_lin_vel = 3.0
        # tracking_ang_vel = 1.5
        # lin_vel_z = -2.0
        # ang_vel_xy = -0.05
        # orientation = -1.0
        # base_height = -10.0

        # dof_acc = -2.5e-7 / 10.0
        # dof_vel = -1e-3 / 10.0
        # torques = -0.00001 / 10.0
        # action_rate = -0.01 / 10.0

        # feet_air_time = 0.0
        # collision = 0.0
        # dof_pos_limits = -5.0
        # alive = 5.0
        # hip_pos = -1.0
        # contact_no_vel = -0.2
        # feet_swing_height = -20.0
        # contact = 0.18
        tracking_lin_vel = 3.0
        tracking_ang_vel = 1.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -1.0
        base_height = -10.0
        dof_acc = -2.5e-7
        dof_vel = -1e-3
        feet_air_time = 0.0
        collision = 0.0
        action_rate = -0.01
        dof_pos_limits = -5.0
        alive = 0.15
        hip_pos = -1.0
        contact_no_vel = -0.2
        feet_swing_height = -20.0
        # contact = 0.18
        contact = 1.0

    scales = Scales()

@configclass
class G1RoughCfg(LeggedRobotCfg):
    env = G1RoughEnvCfg()
    init_state = G1RoughInitStateCfg()
    domain_rand = G1RoughDomainRandCfg()
    control = G1RoughControlCfg()
    sim = G1RoughSimCfg()
    asset = G1RoughAssetCfg()
    rewards = G1RoughRewardsCfg()

@configclass
class G1RoughPolicyCfg(LeggedRobotPolicyCfg):

    @configclass
    class ObsProcActor:
        torso = { 'type': 'identity' }
    
    @configclass
    class ObsProcCritic:
        torso_privileged = { 'type': 'identity' }

    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

    # init_noise_std = 0.8
    # actor_hidden_dims = [32]
    # critic_hidden_dims = [32]
    # activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    # rnn_type = 'lstm'
    # rnn_hidden_size = 64
    # rnn_num_layers = 1
    init_noise_std = 0.8
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    rnn_type = 'lstm'
    rnn_hidden_size = 512
    rnn_num_layers = 1

@configclass
class G1RoughAlgorithmCfg(LeggedRobotAlgorithmCfg):
    entropy_coef = 0.001
    learning_rate = 1.e-4
    schedule = 'fixed'

@configclass
class G1RoughRunnerCfg(LeggedRobotRunnerCfg):
    policy_class_name = "ActorCriticRecurrent"
    # policy_class_name = "ActorCritic"
    max_iterations = 10000
    run_name = ''
    experiment_name = 'g1'

@configclass
class G1RoughCfgPPO(LeggedRobotCfgPPO):
    policy = G1RoughPolicyCfg()
    algorithm = G1RoughAlgorithmCfg()
    runner = G1RoughRunnerCfg()

  
