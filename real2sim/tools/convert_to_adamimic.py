#!/usr/bin/env python3
"""Convert VideoMimic real2sim retarget output to AdaMimic stage1 training data.

Usage:
    conda activate adamimic  # needs torch
    python convert_to_adamimic.py \
        --input /path/to/retarget_poses_g1.h5 \
        --output_dir /path/to/output_dir \
        [--task_name my_task] \
        [--smooth]

Reads retarget_poses_g1.h5 (VideoMimic stage4 output) and produces data.pt
compatible with AdaMimic's MotionLib stage1 training pipeline.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Joint mapping: VideoMimic 23 joints -> AdaMimic 27 joints
# ---------------------------------------------------------------------------
# VideoMimic joint names (23 total):
#   0-5:   left leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   6-11:  right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   12:    waist_yaw
#   13:    waist_roll   <-- AdaMimic has this as FIXED joint
#   14:    waist_pitch  <-- AdaMimic has this as FIXED joint
#   15-18: left arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   19-22: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#
# AdaMimic joint names (27 total):
#   0-11:  same legs
#   12:    waist_yaw
#   13-16: left arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   17-19: left wrist (roll, pitch, yaw) <-- VideoMimic has NO wrist joints
#   20-23: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   24-26: right wrist (roll, pitch, yaw) <-- VideoMimic has NO wrist joints

VM_TO_ADA_JOINTS = np.array([
    0, 1, 2, 3, 4, 5,       # left leg  (0-5)
    6, 7, 8, 9, 10, 11,      # right leg (6-11)
    12,                       # waist_yaw (12)
    15, 16, 17, 18,           # left arm: shoulder_pitch..elbow (13-16)
    -1, -1, -1,               # left wrist: roll, pitch, yaw (17-19) -> zero
    19, 20, 21, 22,           # right arm: shoulder_pitch..elbow (20-23)
    -1, -1, -1,               # right wrist: roll, pitch, yaw (24-26) -> zero
])

# ---------------------------------------------------------------------------
# Link mapping: VideoMimic 37 links -> AdaMimic 17 links
# ---------------------------------------------------------------------------
# VideoMimic link indices (selected from 37):
VM_TO_ADA_LINKS = np.array([
    0,    # 0:  pelvis              -> pelvis
    3,    # 1:  left_hip_roll_link  -> left_hip
    5,    # 2:  left_knee_link      -> left_knee
    6,    # 3:  left_ankle_pitch_link -> left_ankle
    9,    # 4:  right_hip_roll_link -> right_hip
    11,   # 5:  right_knee_link     -> right_knee
    12,   # 6:  right_ankle_pitch_link -> right_ankle
    19,   # 7:  head_link           -> head
    17,   # 8:  torso_link          -> torso
    21,   # 9:  left_shoulder_pitch_link -> left_collar (approximate)
    22,   # 10: left_shoulder_roll_link  -> left_shoulder
    24,   # 11: left_elbow_link     -> left_elbow
    28,   # 12: left_rubber_hand    -> left_wrist
    29,   # 13: right_shoulder_pitch_link -> right_collar (approximate)
    30,   # 14: right_shoulder_roll_link  -> right_shoulder
    32,   # 15: right_elbow_link    -> right_elbow
    36,   # 16: right_rubber_hand   -> right_wrist
])


# ---------------------------------------------------------------------------
# Forward Kinematics solver
# ---------------------------------------------------------------------------

# Default URDF path for the G1 27-DoF robot.
_DEFAULT_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', 'AdaMimic', 'legged_gym', 'resources',
    'robots', 'g1', 'urdf', 'g1_27dof.urdf',
)

# The 17 keyframe links in AdaMimic order.
KEYFRAME_LINK_NAMES = [
    "keyframe_pelvis_link",       # 0
    "keyframe_left_hip_link",     # 1
    "keyframe_left_knee_link",    # 2
    "keyframe_left_ankle_link",   # 3
    "keyframe_right_hip_link",    # 4
    "keyframe_right_knee_link",   # 5
    "keyframe_right_ankle_link",  # 6
    "keyframe_head_link",         # 7
    "keyframe_torso_link",        # 8
    "keyframe_left_collar_link",  # 9
    "keyframe_left_shoulder_link",# 10
    "keyframe_left_elbow_link",   # 11
    "keyframe_left_wrist_link",   # 12
    "keyframe_right_collar_link", # 13
    "keyframe_right_shoulder_link",#14
    "keyframe_right_elbow_link",  # 15
    "keyframe_right_wrist_link",  # 16
]

# The 27 actuated joints in AdaMimic order, mapping joint name -> index.
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
    "left_shoulder_pitch_joint",  # 13
    "left_shoulder_roll_joint",   # 14
    "left_shoulder_yaw_joint",    # 15
    "left_elbow_joint",           # 16
    "left_wrist_roll_joint",      # 17
    "left_wrist_pitch_joint",     # 18
    "left_wrist_yaw_joint",       # 19
    "right_shoulder_pitch_joint", # 20
    "right_shoulder_roll_joint",  # 21
    "right_shoulder_yaw_joint",   # 22
    "right_elbow_joint",          # 23
    "right_wrist_roll_joint",     # 24
    "right_wrist_pitch_joint",    # 25
    "right_wrist_yaw_joint",      # 26
]


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles (roll, pitch, yaw) to 3x3 rotation matrix."""
    return R.from_euler('xyz', rpy).as_matrix()


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz translation and xyz euler rpy."""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rotation_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from axis-angle (Rodrigues)."""
    if abs(angle) < 1e-12:
        return np.eye(3)
    return R.from_rotvec(axis * angle).as_matrix()


def _quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternion to 3x3 rotation matrix."""
    return R.from_quat(quat_xyzw).as_matrix()


class URDFFKSolver:
    """Forward kinematics solver built from a parsed URDF.

    Parses the URDF once at construction, then computes world poses of
    keyframe links given joint angles + root state.
    """

    def __init__(self, urdf_path: str):
        self.joint_info = {}   # joint_name -> dict(parent, child, xyz, rpy, axis, jtype)
        self.parent_to_joints = {}  # parent_link_name -> [joint_name, ...]
        self.child_to_joint = {}    # child_link_name  -> joint_name
        self.actuated_index = {}    # joint_name -> index in 27-DoF vector

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
                xyz_str = origin_el.get('xyz', '0 0 0')
                rpy_str = origin_el.get('rpy', '0 0 0')
                xyz = np.array([float(v) for v in xyz_str.split()])
                rpy = np.array([float(v) for v in rpy_str.split()])
            else:
                xyz = np.zeros(3)
                rpy = np.zeros(3)

            axis_el = joint_el.find('axis')
            if axis_el is not None:
                axis = np.array([float(v) for v in axis_el.get('xyz').split()])
            else:
                axis = np.array([0.0, 0.0, 1.0])

            info = {
                'parent': parent_link,
                'child': child_link,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis,
                'jtype': jtype,
            }
            self.joint_info[jname] = info

            if parent_link not in self.parent_to_joints:
                self.parent_to_joints[parent_link] = []
            self.parent_to_joints[parent_link].append(jname)
            self.child_to_joint[child_link] = jname

    def _build_chains(self):
        """Pre-compute the ordered chain of joints from pelvis to each keyframe link.

        For each keyframe link, the chain is a list of dicts:
            {'joint': joint_name, 'origin_tf': 4x4, 'axis': 3-vector,
             'is_actuated': bool, 'actuated_idx': int or -1}
        followed by the keyframe offset transform.
        """
        self.keyframe_chains = []  # list of 17 chains, one per KEYFRAME_LINK_NAMES

        for kf_name in KEYFRAME_LINK_NAMES:
            # Find the keyframe joint that defines this link's parent + offset
            kf_joint_name = None
            for jname, info in self.joint_info.items():
                if info['child'] == kf_name:
                    kf_joint_name = jname
                    break
            if kf_joint_name is None:
                raise ValueError(f"Could not find joint for keyframe link {kf_name}")

            kf_info = self.joint_info[kf_joint_name]
            parent_link = kf_info['parent']
            kf_offset_tf = _make_transform(kf_info['xyz'], kf_info['rpy'])

            # Trace chain from parent_link back to pelvis
            chain_joints = []
            current_link = parent_link
            while current_link != 'pelvis':
                if current_link not in self.child_to_joint:
                    raise ValueError(
                        f"Link {current_link} has no parent joint (chain to pelvis broken)")
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
                    'jtype': info['jtype'],
                })
                current_link = info['parent']

            self.keyframe_chains.append({
                'chain_joints': chain_joints,
                'kf_offset_tf': kf_offset_tf,
            })

    def compute(
        self,
        joint_angles: np.ndarray,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
    ) -> tuple:
        """Compute FK for all 17 keyframe links.

        Args:
            joint_angles: (T, 27) array of joint angles in AdaMimic order.
            root_pos: (T, 3) array of root (pelvis) positions.
            root_quat: (T, 4) array of root (pelvis) quaternions (XYZW).

        Returns:
            keyframe_positions: (T, 17, 3) world positions
            keyframe_orientations: (T, 17, 4) world quaternions (XYZW)
        """
        T = joint_angles.shape[0]
        num_kf = len(KEYFRAME_LINK_NAMES)
        positions = np.zeros((T, num_kf, 3), dtype=np.float64)
        orientations = np.zeros((T, num_kf, 4), dtype=np.float64)

        for t in range(T):
            # Root transform: pelvis in world frame
            root_rot = _quat_to_matrix(root_quat[t])
            root_tf = np.eye(4)
            root_tf[:3, :3] = root_rot
            root_tf[:3, 3] = root_pos[t]

            for kf_idx, chain_data in enumerate(self.keyframe_chains):
                current_tf = root_tf.copy()

                for joint_data in chain_data['chain_joints']:
                    # Apply the joint's fixed origin transform
                    current_tf = current_tf @ joint_data['origin_tf']

                    # Apply the joint rotation
                    if joint_data['is_actuated']:
                        angle = joint_angles[t, joint_data['actuated_idx']]
                        rot_mat = _axis_angle_to_matrix(joint_data['axis'], angle)
                        rot_tf = np.eye(4)
                        rot_tf[:3, :3] = rot_mat
                        current_tf = current_tf @ rot_tf

                # Apply keyframe offset
                current_tf = current_tf @ chain_data['kf_offset_tf']

                positions[t, kf_idx] = current_tf[:3, 3]
                orientations[t, kf_idx] = R.from_matrix(current_tf[:3, :3]).as_quat()

        return positions, orientations


def compute_keyframe_fk(
    joint_angles: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    urdf_path: str | None = None,
) -> tuple:
    """Compute world positions and orientations of 17 keyframe links via FK.

    This is the main entry point for FK computation.

    Args:
        joint_angles: (T, 27) joint angles in AdaMimic order (radians).
        root_pos: (T, 3) root (pelvis) world positions.
        root_quat: (T, 4) root (pelvis) quaternions (XYZW convention).
        urdf_path: Path to the G1 URDF. If None, uses the default path.

    Returns:
        keyframe_positions: (T, 17, 3) numpy array of world positions.
        keyframe_orientations: (T, 17, 4) numpy array of XYZW quaternions.
    """
    if urdf_path is None:
        urdf_path = _DEFAULT_URDF_PATH
    urdf_path = os.path.abspath(urdf_path)
    solver = URDFFKSolver(urdf_path)
    return solver.compute(joint_angles, root_pos, root_quat)


def remap_joints(vm_joints: np.ndarray) -> np.ndarray:
    """Remap VideoMimic 23 joints to AdaMimic 27 joints.

    Drops waist_roll/pitch, inserts zero wrist joints.
    """
    T = vm_joints.shape[0]
    ada_joints = np.zeros((T, 27), dtype=np.float32)
    for ada_idx, vm_idx in enumerate(VM_TO_ADA_JOINTS):
        if vm_idx >= 0:
            ada_joints[:, ada_idx] = vm_joints[:, vm_idx]
    return ada_joints


def remap_links(vm_link_pos: np.ndarray, vm_link_quat: np.ndarray) -> tuple:
    """Select 17 links from 37 VideoMimic links.

    Returns:
        link_pos: (T, 17, 3) world positions
        link_quat: (T, 17, 4) XYZW quaternions
    """
    link_pos = vm_link_pos[:, VM_TO_ADA_LINKS, :].copy()
    link_quat = vm_link_quat[:, VM_TO_ADA_LINKS, :].copy()
    return link_pos, link_quat


def quat_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions to XYZ Euler angles (roll, pitch, yaw).

    Args:
        quat_xyzw: (..., 4) array of XYZW quaternions

    Returns:
        rpy: (..., 3) array of roll, pitch, yaw in radians
    """
    shape = quat_xyzw.shape[:-1]
    quat_flat = quat_xyzw.reshape(-1, 4)
    rot = R.from_quat(quat_flat)  # scipy expects XYZW, which we have
    rpy = rot.as_euler('xyz', degrees=False)
    return rpy.reshape(*shape, 3)


def read_h5_fps(f: h5py.File) -> float:
    """Read fps from h5 attributes, trying common key names."""
    for key in ('fps', '/fps', 'FPS'):
        if key in f.attrs:
            return float(f.attrs[key])
    return 30.0


def resample_motion(data_dict: dict, src_fps: float, dst_fps: float) -> dict:
    """Resample all time-series arrays from src_fps to dst_fps via linear interpolation.

    Args:
        data_dict: dict of numpy arrays, first dim is time
        src_fps: source frame rate
        dst_fps: target frame rate

    Returns:
        dict with resampled arrays
    """
    if abs(src_fps - dst_fps) < 0.01:
        return data_dict

    src_T = next(v for v in data_dict.values() if isinstance(v, np.ndarray)).shape[0]
    src_times = np.arange(src_T) / src_fps
    duration = src_times[-1]
    dst_T = int(round(duration * dst_fps)) + 1
    dst_times = np.arange(dst_T) / dst_fps
    # Clamp to source range
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


def compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity via central finite differences.

    Uses forward/backward difference at boundaries.

    Args:
        positions: (T, ...) position array
        dt: time step (1/fps)

    Returns:
        velocities: same shape as positions
    """
    vel = np.zeros_like(positions)
    if len(positions) > 2:
        # Central difference for interior
        vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
        # Forward/backward for boundaries
        vel[0] = (positions[1] - positions[0]) / dt
        vel[-1] = (positions[-1] - positions[-2]) / dt
    elif len(positions) == 2:
        vel[0] = vel[1] = (positions[1] - positions[0]) / dt
    return vel


def smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving average smoothing.

    Args:
        signal: (T, ...) array
        window: smoothing window size (must be odd)

    Returns:
        smoothed signal, same shape
    """
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


def convert(
    h5_path: str,
    output_dir: str,
    task_name: str = "converted",
    do_smooth: bool = False,
    smooth_window: int = 5,
    target_fps: float = 30.0,
    source_fps: float | None = None,
    urdf_path: str | None = None,
):
    """Main conversion: h5 -> data.pt

    Computes keyframe link positions and orientations via forward kinematics
    from the URDF, joint angles, and root state, instead of remapping
    VideoMimic's link_pos/link_quat arrays.
    """
    import torch

    # ---- Read h5 ----
    with h5py.File(h5_path, 'r') as f:
        vm_root_pos = f['root_pos'][:].astype(np.float64)
        vm_root_quat = f['root_quat'][:].astype(np.float64)
        vm_joints = f['joints'][:].astype(np.float64)
        h5_fps = read_h5_fps(f)

    # Determine actual source fps
    # h5 stores fps = base_fps / subsample, where base_fps is assumed 30 by default.
    # If the original video was NOT 30fps, pass --source_fps to override.
    if source_fps is not None:
        src_fps = source_fps
    else:
        src_fps = h5_fps

    T = vm_root_pos.shape[0]
    duration = T / src_fps
    print(f"Loaded {T} frames at {src_fps} fps (duration: {duration:.2f}s) from {h5_path}")
    print(f"  h5 reported fps: {h5_fps}, target fps: {target_fps}")

    # ---- Optional smoothing ----
    if do_smooth:
        print(f"Applying smoothing (window={smooth_window})...")
        vm_root_pos = smooth_signal(vm_root_pos, smooth_window)
        vm_joints = smooth_signal(vm_joints, smooth_window)

    # ---- Remap joints (23 -> 27) ----
    ada_joints = remap_joints(vm_joints)
    print(f"Joint mapping: {vm_joints.shape[1]} -> {ada_joints.shape[1]}")

    # ---- Resample to target fps ----
    # Resample root_pos, root_quat, and joints, then recompute FK on resampled data.
    if abs(src_fps - target_fps) > 0.01:
        print(f"Resampling {src_fps}fps -> {target_fps}fps ...")
        resample_data = {
            'root_pos': vm_root_pos,
            'joints': ada_joints,
        }
        resampled = resample_motion(resample_data, src_fps, target_fps)
        vm_root_pos = resampled['root_pos']
        ada_joints = resampled['joints']

        # Resample root quaternion via slerp
        from scipy.spatial.transform import Rotation as R, Slerp
        src_times = np.arange(T) / src_fps
        new_T = vm_root_pos.shape[0]
        dst_times = np.arange(new_T) / target_fps
        dst_times = np.clip(dst_times, src_times[0], src_times[-1])

        root_rots = R.from_quat(vm_root_quat)
        slerp = Slerp(src_times, root_rots)
        vm_root_quat = slerp(dst_times).as_quat()

        T = new_T
        src_fps = target_fps
        print(f"  After resample: {T} frames at {target_fps}fps (duration: {T/target_fps:.2f}s)")

    # ---- Compute keyframe link positions/orientations via FK ----
    print("Computing forward kinematics for 17 keyframe links...")
    ada_link_pos, ada_link_quat = compute_keyframe_fk(
        ada_joints, vm_root_pos, vm_root_quat, urdf_path=urdf_path,
    )
    print(f"FK output: link positions {ada_link_pos.shape}, orientations {ada_link_quat.shape}")

    dt = 1.0 / src_fps

    # ---- Convert root orientation: quat (XYZW) -> RPY (xyz euler) ----
    base_pose = quat_to_rpy(vm_root_quat).astype(np.float32)

    # ---- Convert link orientations: quat -> RPY ----
    link_rpy = quat_to_rpy(ada_link_quat).astype(np.float32)

    # ---- Compute velocities ----
    base_velocity = compute_velocity(vm_root_pos, dt).astype(np.float32)
    base_angular_velocity = compute_velocity(base_pose, dt).astype(np.float32)
    joint_velocity = compute_velocity(ada_joints, dt).astype(np.float32)
    link_velocity = compute_velocity(ada_link_pos, dt).astype(np.float32)
    link_angular_velocity = compute_velocity(link_rpy, dt).astype(np.float32)

    # ---- Assemble data.pt ----
    data = {
        'base_position': torch.tensor(vm_root_pos.astype(np.float32)),
        'base_pose': torch.tensor(base_pose),
        'base_velocity': torch.tensor(base_velocity),
        'base_angular_velocity': torch.tensor(base_angular_velocity),
        'joint_position': torch.tensor(ada_joints),
        'joint_velocity': torch.tensor(joint_velocity),
        'link_position': torch.tensor(ada_link_pos.astype(np.float32)),
        'link_orientation': torch.tensor(link_rpy),
        'link_velocity': torch.tensor(link_velocity),
        'link_angular_velocity': torch.tensor(link_angular_velocity),
    }

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'data.pt')
    torch.save(data, out_path)

    # Copy joint_id.txt if available
    joint_id_src = os.path.join(os.path.dirname(output_dir), 'joint_id.txt')
    joint_id_dst = os.path.join(output_dir, 'joint_id.txt')
    ada_joint_id_src = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'AdaMimic', 'legged_gym', 'resources',
        'dataset', 'g1_dof27_data', 'joint_id.txt',
    )
    if not os.path.exists(joint_id_dst):
        if os.path.exists(joint_id_src):
            import shutil
            shutil.copy2(joint_id_src, joint_id_dst)
        elif os.path.exists(ada_joint_id_src):
            import shutil
            shutil.copy2(ada_joint_id_src, joint_id_dst)

    # ---- Summary ----
    print(f"\nSaved to: {out_path}")
    print(f"{'Field':<30s} {'Shape':<25s} {'dtype'}")
    print("-" * 65)
    for k, v in data.items():
        print(f"{k:<30s} {str(list(v.shape)):<25s} {v.dtype}")

    # ---- Sanity checks ----
    print("\n--- Sanity checks ---")
    print(f"base_position Z range: [{data['base_position'][:,2].min():.4f}, "
          f"{data['base_position'][:,2].max():.4f}]")
    print(f"base_pose (RPY) range: [{data['base_pose'].min():.4f}, "
          f"{data['base_pose'].max():.4f}]")
    print(f"joint_position range: [{data['joint_position'].min():.4f}, "
          f"{data['joint_position'].max():.4f}]")
    print(f"Zero wrist joints (L): {data['joint_position'][0, 17:20].numpy()}")
    print(f"Zero wrist joints (R): {data['joint_position'][0, 24:27].numpy()}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert VideoMimic retarget h5 to AdaMimic data.pt')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to retarget_poses_g1.h5')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for data.pt')
    parser.add_argument('--task_name', default='converted',
                        help='Task name (for logging)')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to reduce noise')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='Smoothing window size (default: 5)')
    parser.add_argument('--target_fps', type=float, default=30.0,
                        help='Target frame rate for output (default: 30)')
    parser.add_argument('--source_fps', type=float, default=None,
                        help='Override source fps (default: read from h5)')
    parser.add_argument('--urdf_path', type=str, default=None,
                        help='Path to G1 27-DoF URDF (default: auto-detect)')
    args = parser.parse_args()

    convert(
        h5_path=args.input,
        output_dir=args.output_dir,
        task_name=args.task_name,
        do_smooth=args.smooth,
        smooth_window=args.smooth_window,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        urdf_path=args.urdf_path,
    )


if __name__ == '__main__':
    main()
