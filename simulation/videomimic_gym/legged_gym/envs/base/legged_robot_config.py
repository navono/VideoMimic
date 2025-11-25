from legged_gym.utils.configclass import configclass
import numpy as np
import torch

@configclass
class ViserCfg:
    enable = False # off by default since we dont want it during training

@configclass
class SensorCfg:
    """Base class for all sensor configurations"""
    enabled = True
    name = ""  # Custom name - if empty, will use default name based on type
    max_delay = 0
    
@configclass
class DepthCameraCfg(SensorCfg):
    """Configuration for depth camera sensors"""
    type = "depth_camera"  # Sensor type identifier
    body_name = "d435_link"
    width = 320
    height = 240
    downsample_factor = 1
    max_distance = 1e6
    only_heading = False
    intrinsic_matrix = None

@configclass
class HeightfieldCfg(SensorCfg):
    """Configuration for heightfield sensors"""
    type = "heightfield"  # Sensor type identifier
    body_name = "pelvis"
    size = (1.0, 1.0)
    resolution = 0.1
    max_distance = 1e6
    only_heading = True
    use_float = True  # Whether to use float32 instead of uint8
    white_noise_scale = 0.0
    offset_noise_scale = 0.0
    roll_noise_scale = 0.0
    pitch_noise_scale = 0.0
    yaw_noise_scale = 0.0
    bad_distance_prob = 0.0
    update_frequency_min = 1
    update_frequency_max = 1

@configclass
class MultiLinkHeightCfg(SensorCfg):
    """Configuration for multi-link height sensors"""
    type = "multi_link_height"  # Sensor type identifier
    body_name = "pelvis"  # Reference body for the sensor
    max_distance = 1e6
    only_heading = True
    link_names = []  # List of link names to measure heights for
    use_float = True  # Whether to use float32 instead of uint8

@configclass
class LeggedRobotSensorsCfg:
    """Configuration for robot sensors.
    
    Use this class to define what sensors your robot will use.
    You can add multiple sensors of the same type with different configurations.
    
    Example:
        ```python
        @configclass
        class MyRobotCfg(LeggedRobotCfg):
            # Configure sensors
            sensors = LeggedRobotSensorsCfg(
                sensor_cfgs = [
                    DepthCameraCfg(name="front_camera", body_name="head"),
                    DepthCameraCfg(name="rear_camera", body_name="torso", width=120, height=80),
                    HeightfieldCfg(name="terrain_height", body_name="pelvis"),
                ]
            )
            
            # Include sensor observations
            env = LeggedRobotEnvCfg(
                obs = ['torso', 'front_camera', 'rear_camera', 'terrain_height'],
                ...
            )
        ```
    """
    # List of sensor configurations
    sensor_cfgs = []  # List of SensorCfg instances

@configclass
class LeggedRobotEnvCfg:
    num_envs: int = 4096
    num_actions: int = 12
    env_spacing: float = 3.  # not used with heightfields/trimeshes 
    send_timeouts: bool = True # not used with heightfields/trimeshes
    episode_length_s: float = 20  # episode length in seconds
    test: bool = False
    obs = ['torso']
    obs_history = {}
    export_trajectory: bool = False  # Whether to export trajectory data during play
    export_dir: str = 'exported_trajectories'  # Directory to save the exported trajectory data

@configclass
class LeggedRobotTerrainCfg:
    terrain_class = None
    mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
    horizontal_scale = 0.1 # [m]
    vertical_scale = 0.005 # [m]
    border_size = 25 # [m]
    curriculum = True
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.
    selected = False # select a unique terrain type and pass all arguments
    terrain_kwargs = None # Dict of arguments for selected terrain
    max_init_terrain_level = 5 # starting curriculum state
    terrain_length = 8.
    terrain_width = 8.
    num_rows= 10 # number of terrain rows (levels)
    num_cols = 20 # number of terrain cols (types)
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    # trimesh only:
    slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

@configclass
class LeggedRobotCommandsCfg:
    curriculum = False
    max_curriculum = 1.
    num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    resampling_time = 10. # time before command are changed[s]
    heading_command = True # if true: compute ang vel command from heading error
    @configclass
    class LeggedRobotCommandsRangesCfg:
        lin_vel_x = [-1.0, 1.0] # min max [m/s]
        lin_vel_y = [-1.0, 1.0]   # min max [m/s]
        ang_vel_yaw = [-1, 1]    # min max [rad/s]
        heading = [-3.14, 3.14]
    ranges = LeggedRobotCommandsRangesCfg()

@configclass
class LeggedRobotInitStateCfg:
    pos = [0.0, 0.0, 1.] # x,y,z [m]
    rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
    lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
    ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
    default_joint_angles = { # target angles when action = 0.0
        "joint_a": 0., 
        "joint_b": 0.}

@configclass
class LeggedRobotControlCfg:
    beta = 1.0 # beta for EMA. corresponds to how much of the new action is used. 1.0 is no EMA.
    control_type = 'P' # P: position, V: velocity, T: torques
    # PD Drive parameters:
    stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
    damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 4
    control_type = 'P'


@configclass
class LeggedRobotAssetCfg:
    file = ""
    name = "legged_robot"  # actor name
    foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
    penalize_contacts_on = []
    terminate_after_contacts_on = []
    disable_gravity = False
    collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
    # collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
    fix_base_link = False # fixe the base of the robot
    default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
    self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
    flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
    
    density = 0.001
    angular_damping = 0.
    linear_damping = 0.
    max_angular_velocity = 1000.
    max_linear_velocity = 1000.
    armature = 0.
    thickness = 0.01

@configclass
class LeggedRobotDomainRandCfg:
    randomize_friction = True
    friction_range = [0.5, 1.25]
    # randomize_base_mass = False
    # added_mass_range = [-1., 1.]
    push_robots = True
    push_interval_s = 15
    max_push_vel_xy = 1.


    randomize_base_mass = False #and domain_rand_general
    added_mass_range = [-3.0, 3.0]

    randomize_base_com = False #and domain_rand_general
    added_com_range = [-0.025, 0.025]

    torque_rfi_rand = True
    torque_rfi_rand_scale = 0.1
    p_gain_rand = False
    p_gain_rand_scale = 0.1
    d_gain_rand = False
    d_gain_rand_scale = 0.1

    # in control steps
    control_delays = False
    control_delay_min=0
    control_delay_max=1

    action_delays = False
    control_delay_min=0
    control_delay_max=1

    randomize_dof_friction = False
    max_dof_friction = 0.05
    dof_friction_buckets = 64

    randomize_odom_update_frequency = False
    # odom will update every _this number_ of steps
    odom_update_steps_min = 1
    odom_update_steps_max = 1

@configclass
class LeggedRobotRewardsCfg:
    @configclass
    class Scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.
        torques = -0.00001
        dof_vel = -0.
        dof_acc = -2.5e-7
        base_height = -0. 
        feet_air_time =  1.0
        collision = -1.
        feet_stumble = -0.0 
        action_rate = -0.01
        stand_still = -0.
    
    scales = Scales()

    only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 1.
    max_contact_force = 100. # forces above this value are penalized

@configclass
class LeggedRobotNormalizationCfg:
    @configclass
    class ObsScales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0
    obs_scales = ObsScales()
    clip_observations = 100.
    clip_actions = 100.

@configclass
class LeggedRobotNoiseCfg:
    add_noise = True
    noise_level = 1.0 # scales other values

    @configclass
    class NoiseScales:
        dof_pos = 0.01
        dof_vel = 1.5
        lin_vel = 0.1
        ang_vel = 0.2
        gravity = 0.05
        height_measurements = 0.1
    noise_scales = NoiseScales()

    @configclass
    class OffsetScales:
        action = 0.0

    offset_scales = OffsetScales()


@configclass
class LeggedRobotViewerCfg:
    ref_env = 0
    pos = [10, 0, 6]  # [m]
    lookat = [11., 5, 3.]  # [m]

@configclass
class LeggedRobotSimCfg:
    dt =  0.005
    substeps = 1
    gravity = [0., 0. ,-9.81]  # [m/s^2]
    # gravity = [0., 0. ,-7.5]  # [m/s^2]
    # gravity = [0., 0. ,-4.905]  # [m/s^2]
    up_axis = 1  # 0 is y, 1 is z

    @configclass
    class Physx:
        num_threads = 10
        solver_type = 1  # 0: pgs, 1: tgs
        num_position_iterations = 4
        num_velocity_iterations = 0
        contact_offset = 0.01  # [m]
        rest_offset = 0.0   # [m]
        bounce_threshold_velocity = 0.5 #0.5 [m/s]
        # max_depenetration_velocity = 1.0
        max_depenetration_velocity = 0.1
        # max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
        # max_gpu_contact_pairs = 2**24 #2**24 -> needed for 8000 envs and more
        max_gpu_contact_pairs = 2**25 #2**24 -> needed for 8000 envs and more
        default_buffer_size_multiplier = 5
        contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    physx = Physx()

@configclass
class LeggedRobotCfg:

    env = LeggedRobotEnvCfg()
    terrain = LeggedRobotTerrainCfg()
    commands = LeggedRobotCommandsCfg()
    init_state = LeggedRobotInitStateCfg()
    control = LeggedRobotControlCfg()
    asset = LeggedRobotAssetCfg()
    domain_rand = LeggedRobotDomainRandCfg()
    rewards = LeggedRobotRewardsCfg()
    normalization = LeggedRobotNormalizationCfg()
    noise = LeggedRobotNoiseCfg()
    viewer = LeggedRobotViewerCfg()
    sim = LeggedRobotSimCfg()
    sensors = LeggedRobotSensorsCfg()
    viser = ViserCfg()


@configclass
class LeggedRobotPolicyCfg:
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    re_init_std = False

    @configclass
    class ObsProcActor:
        torso = { 'type': 'identity' }
    
    @configclass
    class ObsProcCritic:
        torso = { 'type': 'identity' }
    
    obs_proc_actor = ObsProcActor()
    obs_proc_critic = ObsProcCritic()

    # only for 'ActorCriticRecurrent':
    # rnn_type = 'lstm'
    # rnn_hidden_size = 512
    # rnn_num_layers = 1

@configclass
class LeggedRobotAlgorithmCfg:
    # training params
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3 #5.e-4
    schedule = 'adaptive' # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    bounds_loss_coef = 0.0
    # used in bounds loss computation
    clip_actions_threshold = 100.0

@configclass
class LeggedRobotRunnerCfg:
    policy_class_name = 'ActorCritic'
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24 # per iteration
    max_iterations = 1500 # number of policy updates

    # logging
    save_interval = 50 # check for potential saves every this many iterations
    experiment_name = 'test'
    run_name = ''
    # load and resume
    resume = False
    load_run = -1 # -1 = last run
    checkpoint = -1 # -1 = last saved model
    resume_path = None # updated from load_run and chkpt

    # flag to load model with strict=False
    # useful for model surgery (e.g. adding a new head)
    load_model_strict = True

@configclass
class LeggedRobotCfgPPO:
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    policy = LeggedRobotPolicyCfg()
    algorithm = LeggedRobotAlgorithmCfg()
    runner = LeggedRobotRunnerCfg()