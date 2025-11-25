import viser
import time
import trimesh
import numpy as np
from sklearn.linear_model import RANSACRegressor
import h5py
from scipy.spatial.transform import Rotation


def get_rotation_matrix_to_align_vectors(vec1, vec2):
    """Get rotation matrix to align vec1 with vec2, assuming rotation should be less than 90 degrees"""
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # If vectors are parallel, return identity
    if np.allclose(vec1, vec2):
        return np.eye(3)
    
    # If vectors are nearly anti-parallel, flip vec1 to get the smaller rotation
    if np.dot(vec1, vec2) < 0:
        vec1 = -vec1
    
    v = np.cross(vec1, vec2)
    s = np.linalg.norm(v)
    c = np.dot(vec1, vec2)
    
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    
    R = np.eye(3) + v_x + v_x.dot(v_x) * (1 - c) / (s * s)
    return R


def main() -> None:
    # Initialize viser server
    server = viser.ViserServer()

    folder = "/home/arthur/code/retargeted_data/IMG_7811_new"

    # load the mesh from trimesh
    mesh = trimesh.load(f"{folder}/background_mesh.obj")
    vertices = mesh.vertices

    # Find the bottom height
    bottom_height = np.min(vertices[:, 2])
    
    # Get points below 0.5m from bottom
    mask_low = vertices[:, 2] < (bottom_height + 0.5)
    low_points = vertices[mask_low]
    
    if len(low_points) > 0:
        # Prepare data for RANSAC
        X = low_points[:, [0, 1]]  # x,y coordinates
        y = low_points[:, 2]       # z coordinates
        
        # Fit plane using RANSAC
        ransac = RANSACRegressor(random_state=0)
        ransac.fit(X, y)
        
        # Get plane equation z = ax + by + c
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        
        # Compute plane normal vector (a, b, -1) for z = ax + by + c
        plane_normal = np.array([a, b, -1])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # Get rotation matrix to align plane normal with z-axis (0, 0, 1)
        z_axis = np.array([0, 0, 1])
        R = get_rotation_matrix_to_align_vectors(plane_normal, z_axis)
        
        # Apply rotation to all vertices
        vertices = (R @ vertices.T).T
        
        # After rotation, replace points close to the XY plane
        threshold = 0.1  # Distance threshold to replace points
        for i in range(len(vertices)):
            if vertices[i, 2] < (bottom_height + 0.5):
                if abs(vertices[i, 2] - bottom_height) < threshold:
                    vertices[i, 2] = bottom_height

        # Now process the H5 motion data with the same rotation
        h5_path = f"{folder}/retarget_poses_g1.h5"
        with h5py.File(h5_path, 'r') as f:
            # Load all data first
            data = {}
            def copy_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[:]
            f.visititems(copy_dataset)
            # Copy attributes
            attrs = dict(f.attrs)
        
        # Convert rotation matrix to quaternion (scipy uses wxyz format)
        R_scipy = Rotation.from_matrix(R)
        rot_quat = R_scipy.as_quat()  # wxyz format
        rot_quat = np.array([rot_quat[1], rot_quat[2], rot_quat[3], rot_quat[0]])  # convert to xyzw
        
        # Rotate positions
        data['root_pos'] = (R @ data['root_pos'].T).T
        data['link_pos'] = (R @ data['link_pos'].reshape(-1, 3).T).T.reshape(data['link_pos'].shape)
        
        # Rotate quaternions
        # Convert xyzw to scipy format (wxyz) for multiplication
        root_quat_scipy = np.column_stack([data['root_quat'][:, 3], data['root_quat'][:, 0:3]])
        
        # Handle link quaternions maintaining original shape
        original_link_shape = data['link_quat'].shape
        link_quat_flat = data['link_quat'].reshape(-1, 4)  # Flatten to 2D array
        link_quat_scipy = np.column_stack([link_quat_flat[:, 3], link_quat_flat[:, 0:3]])
        
        # Apply rotation
        root_rot = Rotation.from_quat(root_quat_scipy)
        link_rot = Rotation.from_quat(link_quat_scipy)
        new_root_rot = R_scipy * root_rot
        new_link_rot = R_scipy * link_rot
        
        # Convert back to xyzw format
        new_root_quat = new_root_rot.as_quat()  # wxyz format
        new_link_quat = new_link_rot.as_quat()  # wxyz format
        
        data['root_quat'] = np.column_stack([new_root_quat[:, 1:4], new_root_quat[:, 0]])  # convert to xyzw
        # Reshape link quaternions back to original shape
        data['link_quat'] = np.column_stack([new_link_quat[:, 1:4], new_link_quat[:, 0]]).reshape(original_link_shape)
        
        # Save transformed data
        output_path = f"{folder}/retarget_poses_g1_fit.h5"
        with h5py.File(output_path, 'w') as f:
            # Recursively create groups and datasets
            def create_dataset_recursive(name, data):
                if isinstance(data, dict):
                    group = f.create_group(name)
                    for key, value in data.items():
                        create_dataset_recursive(f"{name}/{key}" if name else key, value)
                else:
                    f.create_dataset(name, data=data)
            
            # Create all datasets
            for key, value in data.items():
                create_dataset_recursive(key, value)
            
            # Copy original attributes
            for key, value in attrs.items():
                f.attrs[key] = value

    faces = mesh.faces
    
    # Create new mesh with processed vertices
    processed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Save the processed mesh, ensuring we save as a single mesh
    processed_mesh.export(f"{folder}/background_mesh_fit.obj", file_type='obj')
    
    # load mesh for visualization
    mesh = server.scene.add_mesh_simple("/mesh", vertices, faces, side="double")
    print(f'done export')
    while True:

        # fit mesh to terrain
        # server.scene.fit_mesh_to_terrain(mesh)

        time.sleep(0.01)


if __name__ == "__main__":
    main()

