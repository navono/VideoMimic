#!/usr/bin/env python3
"""Convert VideoMimic real2sim retarget output to unitree_rl_lab mimic training NPZ format.

Usage:
    conda activate vm1rs
    python convert_to_unitree_rl_lab.py \
        --input /path/to/retarget_poses_g1.h5 \
        --output /path/to/output_motion.npz \
        [--smooth] \
        [--target_fps 50]

Reads retarget_poses_g1.h5 (VideoMimic stage4 output) and produces an NPZ file
compatible with unitree_rl_lab's MotionLoader class used in mimic training.

Output NPZ format:
    fps:               (1,)       int64    frame rate
    joint_pos:         (T, 29)    float32  joint angles in radians
    joint_vel:         (T, 29)    float32  joint velocities in rad/s
    body_pos_w:        (T, 32, 3) float32  body world positions in meters
    body_quat_w:       (T, 32, 4) float32  body world quaternions (WXYZ)
    body_lin_vel_w:    (T, 32, 3) float32  body linear velocities in m/s
    body_ang_vel_w:    (T, 32, 3) float32  body angular velocities in rad/s
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# ---------------------------------------------------------------------------
# Joint mapping: VideoMimic 23 joints -> unitree_rl_lab 29 joints
# ---------------------------------------------------------------------------
# VideoMimic joint names (23 total):
#   0-5:   left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   6-11:  right leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   12-14: waist      (yaw, roll, pitch)
#   15-18: left arm   (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   19-22: right arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#
# unitree_rl_lab 29-DOF joint order:
#   0-11:  same legs
#   12-14: same waist
#   15-18: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   19-21: left wrist  (roll, pitch, yaw) <-- VideoMimic has NO wrist joints
#   22-25: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   26-28: right wrist (roll, pitch, yaw) <-- VideoMimic has NO wrist joints

VM_TO_URL_JOINTS = np.array([
    0, 1, 2, 3, 4, 5,        # left leg   (0-5)
    6, 7, 8, 9, 10, 11,      # right leg  (6-11)
    12, 13, 14,               # waist      (12-14)
    15, 16, 17, 18,           # left arm   (15-18)
    -1, -1, -1,               # left wrist (19-21) -> zero
    19, 20, 21, 22,           # right arm  (22-25)
    -1, -1, -1,               # right wrist (26-28) -> zero
])

NUM_URL_JOINTS = 29

# ---------------------------------------------------------------------------
# Isaac Lab Articulation internal ordering (BFS traversal of kinematic tree)
# ---------------------------------------------------------------------------
# PhysX orders joints/bodies by breadth-first traversal (left/right/center at
# each depth), which differs from the SDK's depth-first grouping by body part.
# Derived empirically by matching CSV joint values against reference NPZ data.
#
# internal_to_sdk[i] = SDK index of the i-th internal joint
INTERNAL_TO_SDK_JOINT = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])

# Number of rigid bodies in the Isaac Lab articulation (root + 29 revolute-joint
# children; fixed-joint children like rubber_hand are merged into their parents).
NUM_INTERNAL_BODIES = 30

# internal_to_vm_body[i] = TARGET_BODY_NAMES index of the i-th internal body
INTERNAL_TO_VM_BODY = np.array([
    0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 24,
    5, 11, 17, 25, 6, 12, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30,
])

# ---------------------------------------------------------------------------
# Target body names: 32 rigid bodies for FK computation (VM ordering)
# ---------------------------------------------------------------------------
# These are the 39 URDF links minus 7 visual/sensor-only links that Isaac Lab
# does not include as rigid bodies: pelvis_contour_link, logo_link, head_link,
# imu_in_torso, imu_in_pelvis, d435_link, mid360_link
TARGET_BODY_NAMES = [
    "pelvis",                   # 0
    "left_hip_pitch_link",      # 1
    "left_hip_roll_link",       # 2
    "left_hip_yaw_link",        # 3
    "left_knee_link",           # 4
    "left_ankle_pitch_link",    # 5
    "left_ankle_roll_link",     # 6
    "right_hip_pitch_link",     # 7
    "right_hip_roll_link",      # 8
    "right_hip_yaw_link",       # 9
    "right_knee_link",          # 10
    "right_ankle_pitch_link",   # 11
    "right_ankle_roll_link",    # 12
    "waist_yaw_link",           # 13
    "waist_roll_link",          # 14
    "torso_link",               # 15
    "left_shoulder_pitch_link", # 16
    "left_shoulder_roll_link",  # 17
    "left_shoulder_yaw_link",   # 18
    "left_elbow_link",          # 19
    "left_wrist_roll_link",     # 20
    "left_wrist_pitch_link",    # 21
    "left_wrist_yaw_link",      # 22
    "left_rubber_hand",         # 23
    "right_shoulder_pitch_link",# 24
    "right_shoulder_roll_link", # 25
    "right_shoulder_yaw_link",  # 26
    "right_elbow_link",         # 27
    "right_wrist_roll_link",    # 28
    "right_wrist_pitch_link",   # 29
    "right_wrist_yaw_link",     # 30
    "right_rubber_hand",        # 31
]

NUM_BODIES = len(TARGET_BODY_NAMES)  # 32

# 29 actuated joint names in unitree_rl_lab order
ACTUATED_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
]

# Default URDF path for G1 29-DOF robot
_DEFAULT_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'simulation', 'videomimic_gym', 'resources',
    'robots', 'g1_description', 'g1_29dof_rev_1_0.urdf',
)


# ===========================================================================
# Utility functions (reused from convert_to_protomotions.py / convert_to_adamimic.py)
# ===========================================================================

def read_h5_fps(f: h5py.File) -> float:
    """Read fps from h5 attributes, trying common key names."""
    for key in ('fps', '/fps', 'FPS'):
        if key in f.attrs:
            return float(f.attrs[key])
    return 30.0


def remap_joints(vm_joints: np.ndarray) -> np.ndarray:
    """Remap VideoMimic 23 joints to unitree_rl_lab 29 joints.

    Inserts zero wrist joints (6 total).
    """
    T = vm_joints.shape[0]
    url_joints = np.zeros((T, NUM_URL_JOINTS), dtype=vm_joints.dtype)
    for url_idx, vm_idx in enumerate(VM_TO_URL_JOINTS):
        if vm_idx >= 0:
            url_joints[:, url_idx] = vm_joints[:, vm_idx]
    return url_joints


def resample_motion(data_dict: dict, src_fps: float, dst_fps: float) -> dict:
    """Resample all time-series arrays from src_fps to dst_fps via linear interpolation."""
    if abs(src_fps - dst_fps) < 0.01:
        return data_dict

    src_T = next(v for v in data_dict.values() if isinstance(v, np.ndarray)).shape[0]
    src_times = np.arange(src_T) / src_fps
    duration = src_times[-1]
    dst_T = int(round(duration * dst_fps)) + 1
    dst_times = np.arange(dst_T) / dst_fps
    dst_times = np.clip(dst_times, src_times[0], src_times[-1])

    resampled = {}
    for key, val in data_dict.items():
        if not isinstance(val, np.ndarray) or val.ndim < 1:
            resampled[key] = val
            continue
        shape = val.shape
        flat = val.reshape(src_T, -1)
        out = np.zeros((dst_T, flat.shape[1]), dtype=val.dtype)
        for col in range(flat.shape[1]):
            out[:, col] = np.interp(dst_times, src_times, flat[:, col])
        resampled[key] = out.reshape(dst_T, *shape[1:])

    return resampled


def resample_quaternion_slerp(quat_xyzw: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample XYZW quaternions via SLERP."""
    if abs(src_fps - dst_fps) < 0.01:
        return quat_xyzw

    T = quat_xyzw.shape[0]
    src_times = np.arange(T) / src_fps
    duration = src_times[-1]
    dst_T = int(round(duration * dst_fps)) + 1
    dst_times = np.arange(dst_T) / dst_fps
    dst_times = np.clip(dst_times, src_times[0], src_times[-1])

    rots = R.from_quat(quat_xyzw)
    slerp = Slerp(src_times, rots)
    return slerp(dst_times).as_quat()


def smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if window < 3 or signal.shape[0] < window:
        return signal
    if window % 2 == 0:
        window += 1
    from scipy.ndimage import uniform_filter1d
    original_shape = signal.shape
    flat = signal.reshape(signal.shape[0], -1)
    smoothed = np.zeros_like(flat)
    for i in range(flat.shape[1]):
        smoothed[:, i] = uniform_filter1d(flat[:, i], size=window, mode='nearest')
    return smoothed.reshape(original_shape)


def compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity via central finite differences."""
    vel = np.zeros_like(positions)
    if len(positions) > 2:
        vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
        vel[0] = (positions[1] - positions[0]) / dt
        vel[-1] = (positions[-1] - positions[-2]) / dt
    elif len(positions) == 2:
        vel[0] = vel[1] = (positions[1] - positions[0]) / dt
    return vel


def compute_so3_angular_velocity(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from a sequence of XYZW quaternions via SO3 derivative.

    Uses vectorized computation: omega = rotvec(q_{t+1} * q_{t-1}^{-1}) / (2*dt)
    Boundary frames copy nearest interior value.
    """
    T = quat_xyzw.shape[0]
    ang_vel = np.zeros((T, 3), dtype=np.float64)
    if T < 3:
        return ang_vel

    rots = R.from_quat(quat_xyzw)
    r_prev = rots[:-2]
    r_next = rots[2:]
    r_rel = r_next * r_prev.inv()
    rotvecs = r_rel.as_rotvec()  # (T-2, 3)
    ang_vel[1:-1] = rotvecs / (2.0 * dt)
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


# ===========================================================================
# URDF Forward Kinematics Solver
# ===========================================================================

def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz translation and XYZ Euler rpy."""
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from axis-angle."""
    if abs(angle) < 1e-12:
        return np.eye(3)
    return R.from_rotvec(axis * angle).as_matrix()


class URDFFKSolver:
    """Forward kinematics solver for the G1 29-DOF robot.

    Computes world poses of all 32 rigid bodies from joint angles + root state.
    """

    def __init__(self, urdf_path: str):
        self.joint_info = {}       # joint_name -> {parent, child, xyz, rpy, axis, jtype}
        self.parent_to_joints = {} # parent_link -> [joint_names]
        self.child_to_joint = {}   # child_link -> joint_name
        self.actuated_index = {}   # joint_name -> index in 29-DOF vector

        for i, name in enumerate(ACTUATED_JOINT_NAMES):
            self.actuated_index[name] = i

        self._parse_urdf(urdf_path)
        self._build_chains()

    def _parse_urdf(self, urdf_path: str):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint_el in root.iter('joint'):
            jname = joint_el.get('name')
            jtype = joint_el.get('type')
            parent_el = joint_el.find('parent')
            child_el = joint_el.find('child')
            if parent_el is None or child_el is None:
                continue
            parent_link = parent_el.get('link')
            child_link = child_el.get('link')

            origin_el = joint_el.find('origin')
            if origin_el is not None:
                xyz = np.array([float(v) for v in origin_el.get('xyz', '0 0 0').split()])
                rpy = np.array([float(v) for v in origin_el.get('rpy', '0 0 0').split()])
            else:
                xyz = np.zeros(3)
                rpy = np.zeros(3)

            axis_el = joint_el.find('axis')
            if axis_el is not None:
                axis = np.array([float(v) for v in axis_el.get('xyz').split()])
            else:
                axis = np.array([0.0, 0.0, 1.0])

            self.joint_info[jname] = {
                'parent': parent_link,
                'child': child_link,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis,
                'jtype': jtype,
            }
            self.parent_to_joints.setdefault(parent_link, []).append(jname)
            self.child_to_joint[child_link] = jname

    def _build_chains(self):
        """Pre-compute ordered chains from pelvis to each of the 32 target bodies."""
        self.body_chains = []

        for body_name in TARGET_BODY_NAMES:
            # Special case: pelvis is the root body, no chain needed
            if body_name == 'pelvis':
                self.body_chains.append({
                    'chain_joints': [],
                    'body_offset_tf': np.eye(4),
                })
                continue

            # Find the joint whose child is this body link
            body_joint_name = None
            for jname, info in self.joint_info.items():
                if info['child'] == body_name:
                    body_joint_name = jname
                    break
            if body_joint_name is None:
                raise ValueError(f"Could not find joint for body link '{body_name}'")

            body_info = self.joint_info[body_joint_name]
            parent_link = body_info['parent']
            # The offset from parent to this body (via the joint's origin)
            body_offset_tf = _make_transform(body_info['xyz'], body_info['rpy'])

            # Trace chain from parent_link back to pelvis
            chain_joints = []
            current_link = parent_link
            while current_link != 'pelvis':
                if current_link not in self.child_to_joint:
                    raise ValueError(
                        f"Link '{current_link}' has no parent joint (chain to pelvis broken)")
                jname = self.child_to_joint[current_link]
                info = self.joint_info[jname]
                origin_tf = _make_transform(info['xyz'], info['rpy'])
                is_actuated = jname in self.actuated_index
                actuated_idx = self.actuated_index.get(jname, -1)
                chain_joints.insert(0, {
                    'joint': jname,
                    'origin_tf': origin_tf,
                    'axis': info['axis'],
                    'is_actuated': is_actuated,
                    'actuated_idx': actuated_idx,
                })
                current_link = info['parent']

            self.body_chains.append({
                'chain_joints': chain_joints,
                'body_offset_tf': body_offset_tf,
            })

    def compute(
        self,
        joint_angles: np.ndarray,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
    ) -> tuple:
        """Compute FK for all 32 bodies.

        Args:
            joint_angles: (T, 29) joint angles in unitree_rl_lab order (radians).
            root_pos: (T, 3) pelvis world positions.
            root_quat: (T, 4) pelvis quaternions (XYZW convention).

        Returns:
            body_positions: (T, 32, 3) world positions.
            body_quaternions: (T, 32, 4) world quaternions (XYZW).
        """
        T_steps = joint_angles.shape[0]
        positions = np.zeros((T_steps, NUM_BODIES, 3), dtype=np.float64)
        quaternions = np.zeros((T_steps, NUM_BODIES, 4), dtype=np.float64)

        for t in range(T_steps):
            # Root transform: pelvis in world frame
            root_rot = R.from_quat(root_quat[t]).as_matrix()
            root_tf = np.eye(4)
            root_tf[:3, :3] = root_rot
            root_tf[:3, 3] = root_pos[t]

            for body_idx, chain_data in enumerate(self.body_chains):
                current_tf = root_tf.copy()

                for joint_data in chain_data['chain_joints']:
                    # Apply the joint's origin transform
                    current_tf = current_tf @ joint_data['origin_tf']

                    # Apply the joint rotation
                    if joint_data['is_actuated']:
                        angle = joint_angles[t, joint_data['actuated_idx']]
                        rot_mat = _axis_angle_to_matrix(joint_data['axis'], angle)
                        rot_tf = np.eye(4)
                        rot_tf[:3, :3] = rot_mat
                        current_tf = current_tf @ rot_tf

                # Apply body offset (from joint origin to body link)
                current_tf = current_tf @ chain_data['body_offset_tf']

                positions[t, body_idx] = current_tf[:3, 3]
                quaternions[t, body_idx] = R.from_matrix(current_tf[:3, :3]).as_quat()

        return positions, quaternions


# ===========================================================================
# Main conversion function
# ===========================================================================

def convert(
    h5_path: str,
    output_path: str,
    target_fps: float = 50.0,
    source_fps: float | None = None,
    do_smooth: bool = False,
    smooth_window: int = 5,
    urdf_path: str | None = None,
) -> str:
    """Convert VideoMimic retarget h5 to unitree_rl_lab NPZ.

    Args:
        h5_path: Path to retarget_poses_g1.h5
        output_path: Output NPZ file path
        target_fps: Target frame rate (default: 50 to match unitree_rl_lab convention)
        source_fps: Override source fps (default: read from h5)
        do_smooth: Apply smoothing to reduce noise
        smooth_window: Smoothing window size
        urdf_path: Path to G1 29-DOF URDF (default: auto-detect)

    Returns:
        output_path
    """
    if urdf_path is None:
        urdf_path = _DEFAULT_URDF_PATH
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}", file=sys.stderr)
        print("Please provide --urdf_path pointing to g1_29dof_rev_1_0.urdf", file=sys.stderr)
        sys.exit(1)

    # ---- 1. Read h5 ----
    print(f"Reading {h5_path} ...")
    with h5py.File(h5_path, 'r') as f:
        vm_root_pos = f['root_pos'][:].astype(np.float64)
        vm_root_quat = f['root_quat'][:].astype(np.float64)  # XYZW
        vm_joints = f['joints'][:].astype(np.float64)
        h5_fps = read_h5_fps(f)

    if source_fps is not None:
        src_fps = source_fps
    else:
        src_fps = h5_fps

    T = vm_root_pos.shape[0]
    duration = T / src_fps
    print(f"  {T} frames at {src_fps} fps (duration: {duration:.2f}s)")

    # ---- 2. Optional smoothing ----
    if do_smooth:
        print(f"Applying smoothing (window={smooth_window})...")
        vm_root_pos = smooth_signal(vm_root_pos, smooth_window)
        vm_joints = smooth_signal(vm_joints, smooth_window)

    # ---- 3. Remap joints (23 -> 29) ----
    url_joints = remap_joints(vm_joints)
    print(f"Joint mapping: {vm_joints.shape[1]} -> {url_joints.shape[1]}")

    # ---- 4. Resample to target fps ----
    if abs(src_fps - target_fps) > 0.01:
        print(f"Resampling {src_fps}fps -> {target_fps}fps ...")
        resample_data = {'root_pos': vm_root_pos, 'joints': url_joints}
        resampled = resample_motion(resample_data, src_fps, target_fps)
        vm_root_pos = resampled['root_pos']
        url_joints = resampled['joints']

        vm_root_quat = resample_quaternion_slerp(vm_root_quat, src_fps, target_fps)

        T = vm_root_pos.shape[0]
        src_fps = target_fps
        print(f"  After resample: {T} frames at {target_fps}fps (duration: {T/target_fps:.2f}s)")

    dt = 1.0 / src_fps

    # ---- 5. Compute FK for 32 bodies ----
    print(f"Computing FK for {NUM_BODIES} bodies using {os.path.basename(urdf_path)} ...")
    solver = URDFFKSolver(urdf_path)
    body_pos_xyzw, body_quat_xyzw = solver.compute(url_joints, vm_root_pos, vm_root_quat)
    print(f"  body_pos: {body_pos_xyzw.shape}, body_quat: {body_quat_xyzw.shape}")

    # ---- 5.5 Ground alignment ----
    # VideoMimic retarget output is in scene coordinates (root may be underground).
    # Shift root position so that the minimum ankle height is at Z=0.
    # Ankle bodies in TARGET_BODY_NAMES: 6=left_ankle_roll_link, 12=right_ankle_roll_link
    ANKLE_BODY_INDICES = [6, 12]
    ankle_z_min = body_pos_xyzw[:, ANKLE_BODY_INDICES, 2].min()
    if ankle_z_min < 0:
        z_shift = -ankle_z_min
        vm_root_pos[:, 2] += z_shift
        body_pos_xyzw[:, :, 2] += z_shift
        print(f"  Ground aligned: shifted Z by +{z_shift:.4f}m (ankle was at {ankle_z_min:.4f})")

    # ---- 6. Compute velocities ----
    print("Computing velocities ...")
    joint_vel = compute_velocity(url_joints, dt)
    body_lin_vel = compute_velocity(body_pos_xyzw, dt)

    # Angular velocity per body via SO3 derivative
    body_ang_vel = np.zeros_like(body_pos_xyzw)
    for b in range(NUM_BODIES):
        body_ang_vel[:, b, :] = compute_so3_angular_velocity(body_quat_xyzw[:, b, :], dt)

    # ---- 7. Quaternion convention: XYZW -> WXYZ ----
    body_quat_wxyz = body_quat_xyzw[:, :, [3, 0, 1, 2]]

    # ---- 7.5 Reorder to Isaac Lab internal ordering ----
    # Joints: SDK order -> PhysX BFS internal order
    url_joints = url_joints[:, INTERNAL_TO_SDK_JOINT]
    joint_vel = joint_vel[:, INTERNAL_TO_SDK_JOINT]

    # Bodies: VM 32-body order -> internal 30-body order
    # (excludes left_rubber_hand #23 and right_rubber_hand #31 which are
    #  merged via fixed joints in the simulator)
    body_pos_xyzw = body_pos_xyzw[:, INTERNAL_TO_VM_BODY, :]
    body_quat_wxyz = body_quat_wxyz[:, INTERNAL_TO_VM_BODY, :]
    body_lin_vel = body_lin_vel[:, INTERNAL_TO_VM_BODY, :]
    body_ang_vel = body_ang_vel[:, INTERNAL_TO_VM_BODY, :]
    print(f"Reordered: joints SDK->internal ({INTERNAL_TO_SDK_JOINT.shape[0]}), "
          f"bodies VM 32->internal {NUM_INTERNAL_BODIES}")

    # ---- 8. Assemble and save ----
    fps_out = np.array([int(target_fps)], dtype=np.int64)
    out_data = {
        'fps': fps_out,
        'joint_pos': url_joints.astype(np.float32),
        'joint_vel': joint_vel.astype(np.float32),
        'body_pos_w': body_pos_xyzw.astype(np.float32),
        'body_quat_w': body_quat_wxyz.astype(np.float32),
        'body_lin_vel_w': body_lin_vel.astype(np.float32),
        'body_ang_vel_w': body_ang_vel.astype(np.float32),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True) \
        if os.path.dirname(output_path) else None
    np.savez(output_path, **out_data)

    # ---- Summary ----
    print(f"\nSaved to: {output_path}")
    print(f"{'Field':<20s} {'Shape':<25s} {'dtype'}")
    print("-" * 60)
    for k, v in out_data.items():
        shape_str = str(list(v.shape)) if isinstance(v, np.ndarray) else str(v)
        dtype_str = str(v.dtype) if isinstance(v, np.ndarray) else type(v).__name__
        print(f"{k:<20s} {shape_str:<25s} {dtype_str}")

    # ---- Sanity checks ----
    print("\n--- Sanity checks ---")
    jp = out_data['joint_pos']
    bp = out_data['body_pos_w']
    bq = out_data['body_quat_w']
    blv = out_data['body_lin_vel_w']
    bav = out_data['body_ang_vel_w']

    print(f"pelvis Z range:     [{bp[0, 0, 2]:.4f}, {bp[:, 0, 2].max():.4f}] (should be ~0.76)")
    quat_norms = np.linalg.norm(bq, axis=-1)
    print(f"quat norms:         [{quat_norms.min():.6f}, {quat_norms.max():.6f}] (should be ~1.0)")
    print(f"joint_pos range:    [{jp.min():.4f}, {jp.max():.4f}]")
    print(f"zero L wrist (23,25,27): {jp[0, [23, 25, 27]].tolist()}")
    print(f"zero R wrist (24,26,28): {jp[0, [24, 26, 28]].tolist()}")
    print(f"max lin_vel:        {np.abs(blv).max():.4f} m/s")
    print(f"max ang_vel:        {np.abs(bav).max():.4f} rad/s")
    print(f"num bodies:         {bp.shape[1]} (expected {NUM_INTERNAL_BODIES})")
    print(f"fps:                {fps_out[0]}")

    # Verify pelvis position matches root_pos
    pelvis_diff = np.abs(bp[:, 0, :] - vm_root_pos).max()
    print(f"pelvis vs root_pos: max_diff = {pelvis_diff:.6e} (should be ~0)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert VideoMimic retarget h5 to unitree_rl_lab NPZ format')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to retarget_poses_g1.h5')
    parser.add_argument('--output', '-o', required=True,
                        help='Output NPZ file path')
    parser.add_argument('--target_fps', type=float, default=50.0,
                        help='Target frame rate (default: 50)')
    parser.add_argument('--source_fps', type=float, default=None,
                        help='Override source fps (default: read from h5)')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to reduce noise')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='Smoothing window size (default: 5)')
    parser.add_argument('--urdf_path', type=str, default=None,
                        help='Path to G1 29-DOF URDF (default: auto-detect)')
    args = parser.parse_args()

    convert(
        h5_path=args.input,
        output_path=args.output,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        do_smooth=args.smooth,
        smooth_window=args.smooth_window,
        urdf_path=args.urdf_path,
    )


if __name__ == '__main__':
    main()
