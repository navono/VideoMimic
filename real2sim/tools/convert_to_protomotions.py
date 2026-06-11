#!/usr/bin/env python3
"""Convert VideoMimic real2sim retarget output to ProtoMotions motion format.

Usage:
    conda activate vm1rs
    python convert_to_protomotions.py \
        --input /path/to/retarget_poses_g1.h5 \
        --output_dir /path/to/output_npz/ \
        [--task_name sitting_standing] \
        [--smooth] \
        [--output_contacts]

Reads retarget_poses_g1.h5 (VideoMimic stage4 output) and produces NPZ files
compatible with ProtoMotions' convert_pyroki_retargeted_robot_motions_to_proto.py.

Output NPZ format:
    base_frame_pos:    (T, 3)  root position in meters
    base_frame_wxyz:   (T, 4)  root rotation quaternion (w, x, y, z)
    joint_angles:      (T, 29) joint angles in radians

Optionally produces contact NPZ:
    foot_contacts:     (T, 2)  [left, right] binary contact labels

Then run ProtoMotions' converter:
    python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
        --retargeted-motion-dir <output_npz>/ \
        --output-dir <output_motion>/ \
        --robot-type g1 \
        --contact-labels-dir <output_contacts>/
"""

import argparse
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# ---------------------------------------------------------------------------
# Joint mapping: VideoMimic 23 joints -> ProtoMotions 29 joints
# ---------------------------------------------------------------------------
# VideoMimic joint names (23 total):
#   0-5:   left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   6-11:  right leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   12:    waist_yaw
#   13:    waist_roll
#   14:    waist_pitch
#   15-18: left arm   (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   19-22: right arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#
# ProtoMotions G1 MJCF 29-DoF joint order:
#   0-14:  same legs + waist (identical to VideoMimic)
#   15-18: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   19-21: left wrist  (roll, pitch, yaw) <-- VideoMimic has NO wrist joints
#   22-25: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
#   26-28: right wrist (roll, pitch, yaw) <-- VideoMimic has NO wrist joints

VM_TO_PROTO_JOINTS = np.array([
    0, 1, 2, 3, 4, 5,        # left leg   (0-5)
    6, 7, 8, 9, 10, 11,       # right leg  (6-11)
    12, 13, 14,                # waist      (12-14)
    15, 16, 17, 18,            # left arm   (15-18)
    -1, -1, -1,                # left wrist: roll, pitch, yaw (19-21) -> zero
    19, 20, 21, 22,            # right arm  (22-25)
    -1, -1, -1,                # right wrist: roll, pitch, yaw (26-28) -> zero
])

NUM_PROTO_JOINTS = 29


def read_h5_fps(f: h5py.File) -> float:
    """Read fps from h5 attributes, trying common key names."""
    for key in ('fps', '/fps', 'FPS'):
        if key in f.attrs:
            return float(f.attrs[key])
    return 30.0


def remap_joints(vm_joints: np.ndarray) -> np.ndarray:
    """Remap VideoMimic 23 joints to ProtoMotions 29 joints.

    Inserts zero wrist joints (6 total).
    """
    T = vm_joints.shape[0]
    proto_joints = np.zeros((T, NUM_PROTO_JOINTS), dtype=vm_joints.dtype)
    for proto_idx, vm_idx in enumerate(VM_TO_PROTO_JOINTS):
        if vm_idx >= 0:
            proto_joints[:, proto_idx] = vm_joints[:, vm_idx]
    return proto_joints


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
    """Resample XYZW quaternions via SLERP.

    Args:
        quat_xyzw: (T, 4) XYZW quaternions
        src_fps: source frame rate
        dst_fps: target frame rate

    Returns:
        resampled quaternions in XYZW format
    """
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
    output_contacts: bool = True,
):
    """Main conversion: h5 -> NPZ (ProtoMotions retargeted format)."""
    # ---- Read h5 ----
    with h5py.File(h5_path, 'r') as f:
        vm_root_pos = f['root_pos'][:].astype(np.float64)
        vm_root_quat = f['root_quat'][:].astype(np.float64)  # XYZW
        vm_joints = f['joints'][:].astype(np.float64)
        h5_fps = read_h5_fps(f)

        # Read contacts if available
        vm_left_contact = None
        vm_right_contact = None
        if 'contacts' in f:
            if 'left_foot' in f['contacts']:
                vm_left_contact = f['contacts/left_foot'][:].astype(np.float64)
            if 'right_foot' in f['contacts']:
                vm_right_contact = f['contacts/right_foot'][:].astype(np.float64)

    # Determine actual source fps
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

    # ---- Remap joints (23 -> 29) ----
    proto_joints = remap_joints(vm_joints)
    print(f"Joint mapping: {vm_joints.shape[1]} -> {proto_joints.shape[1]}")

    # ---- Resample to target fps ----
    if abs(src_fps - target_fps) > 0.01:
        print(f"Resampling {src_fps}fps -> {target_fps}fps ...")

        resample_data = {
            'root_pos': vm_root_pos,
            'joints': proto_joints,
        }
        resampled = resample_motion(resample_data, src_fps, target_fps)
        vm_root_pos = resampled['root_pos']
        proto_joints = resampled['joints']

        vm_root_quat = resample_quaternion_slerp(vm_root_quat, src_fps, target_fps)

        # Resample contacts
        if vm_left_contact is not None:
            contact_data = {
                'left': vm_left_contact,
                'right': vm_right_contact,
            }
            resampled_contacts = resample_motion(contact_data, src_fps, target_fps)
            vm_left_contact = resampled_contacts['left']
            vm_right_contact = resampled_contacts['right']

        T = vm_root_pos.shape[0]
        src_fps = target_fps
        print(f"  After resample: {T} frames at {target_fps}fps (duration: {T/target_fps:.2f}s)")

    # ---- Convert quaternion: XYZW -> WXYZ ----
    root_quat_wxyz = vm_root_quat[:, [3, 0, 1, 2]]

    # ---- Cast to float32 ----
    base_frame_pos = vm_root_pos.astype(np.float32)
    base_frame_wxyz = root_quat_wxyz.astype(np.float32)
    joint_angles = proto_joints.astype(np.float32)

    # ---- Save NPZ ----
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{task_name}.npz")
    np.savez(npz_path,
             base_frame_pos=base_frame_pos,
             base_frame_wxyz=base_frame_wxyz,
             joint_angles=joint_angles)

    print(f"\nSaved motion NPZ to: {npz_path}")
    print(f"  base_frame_pos:   {base_frame_pos.shape}  {base_frame_pos.dtype}")
    print(f"  base_frame_wxyz:  {base_frame_wxyz.shape}  {base_frame_wxyz.dtype}")
    print(f"  joint_angles:     {joint_angles.shape}  {joint_angles.dtype}")

    # ---- Save contact NPZ ----
    if output_contacts and vm_left_contact is not None and vm_right_contact is not None:
        contacts_dir = os.path.join(output_dir, "contacts")
        os.makedirs(contacts_dir, exist_ok=True)

        foot_contacts = np.stack([
            (vm_left_contact > 0.5).astype(np.float32),
            (vm_right_contact > 0.5).astype(np.float32),
        ], axis=-1)  # (T, 2)

        contact_path = os.path.join(contacts_dir, f"{task_name}_contacts.npz")
        np.savez(contact_path, foot_contacts=foot_contacts)
        print(f"  foot_contacts:    {foot_contacts.shape}  {foot_contacts.dtype}")
        print(f"  Saved contacts to: {contact_path}")

    # ---- Sanity checks ----
    print("\n--- Sanity checks ---")
    print(f"base_frame_pos Z range: [{base_frame_pos[:,2].min():.4f}, "
          f"{base_frame_pos[:,2].max():.4f}]")
    quat_norms = np.linalg.norm(base_frame_wxyz, axis=-1)
    print(f"quaternion norms: [{quat_norms.min():.6f}, {quat_norms.max():.6f}] "
          f"(should be ~1.0)")
    print(f"joint_angles range: [{joint_angles.min():.4f}, {joint_angles.max():.4f}]")
    print(f"Zero wrist joints (L): {joint_angles[0, 19:22]}")
    print(f"Zero wrist joints (R): {joint_angles[0, 26:29]}")

    # ---- Next steps ----
    print(f"\n--- Next steps ---")
    print(f"1. Run ProtoMotions converter:")
    print(f"   python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \\")
    print(f"       --retargeted-motion-dir {os.path.abspath(output_dir)} \\")
    print(f"       --output-dir <output_motion_dir> \\")
    print(f"       --robot-type g1")
    if output_contacts and vm_left_contact is not None:
        print(f"       --contact-labels-dir {os.path.abspath(os.path.join(output_dir, 'contacts'))}")
    print(f"2. Package into MotionLib .pt:")
    print(f"   python protomotions/components/motion_lib.py \\")
    print(f"       --motion-path <output_motion_dir> \\")
    print(f"       --output-file <task_name>.pt")
    print(f"3. Train with ProtoMotions:")
    print(f"   python protomotions/train_agent.py \\")
    print(f"       --robot-name g1 --simulator isaacgym \\")
    print(f"       --experiment-path examples/experiments/mimic/mlp.py \\")
    print(f"       --motion-file <task_name>.pt")

    return npz_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert VideoMimic retarget h5 to ProtoMotions NPZ format')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to retarget_poses_g1.h5')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for NPZ files')
    parser.add_argument('--task_name', default='converted',
                        help='Task name (used as NPZ filename)')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to reduce noise')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='Smoothing window size (default: 5)')
    parser.add_argument('--target_fps', type=float, default=30.0,
                        help='Target frame rate for output (default: 30)')
    parser.add_argument('--source_fps', type=float, default=None,
                        help='Override source fps (default: read from h5)')
    parser.add_argument('--output_contacts', action='store_true', default=True,
                        help='Save contact labels NPZ (default: True)')
    parser.add_argument('--no_contacts', action='store_true',
                        help='Skip contact labels output')
    args = parser.parse_args()

    output_contacts = args.output_contacts and not args.no_contacts

    convert(
        h5_path=args.input,
        output_dir=args.output_dir,
        task_name=args.task_name,
        do_smooth=args.smooth,
        smooth_window=args.smooth_window,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        output_contacts=output_contacts,
    )


if __name__ == '__main__':
    main()
