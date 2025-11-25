import os
import pickle
import numpy as np
import trimesh
from glob import glob
import argparse
from scipy import interpolate

def inspect_data_structure(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("\nData structure:")
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: numpy array with shape {value.shape} and dtype {value.dtype}")
            else:
                print(f"{key}: {type(value)}")
    return data

def create_rough_terrain(width, length, difficulty=0, resolution=1600):
# def create_rough_terrain(width, length, difficulty=0, resolution=50):
    """
    Generate a rough terrain mesh with given dimensions and difficulty.
    
    Args:
        width: Width of terrain in meters
        length: Length of terrain in meters
        difficulty: Integer 0-4 representing difficulty levels (0=flat, 4=hardest)
        resolution: Number of vertices per side
    """
    # Create a grid of points
    x = np.linspace(-width/2, width/2, resolution)
    y = np.linspace(-length/2, length/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    if difficulty == 0:
        return create_flat_terrain(width, length)
    
    # Scale difficulty to height and frequency parameters
    # max_height = 0.01 * (difficulty + 1)  # 5cm, 10cm, 15cm, 20cm
    # max_height = 0.03#0.01 * (difficulty + 1)  # 5cm, 10cm, 15cm, 20cm
    max_height = 0.05#0.01 * (difficulty + 1)  # 5cm, 10cm, 15cm, 20cm
    print(f"Max height: {max_height}")
    
    # Adjust frequencies based on difficulty
    # base_frequencies = [1, 2, 4, 8]
    # base_frequencies = [2, 4, 6, 8]
    # base_amplitudes = [1.0, 0.5, 0.25, 0.125]
    # base_amplitudes = [1.0, 1.0, 1.0, 1.0]
    base_frequencies = [4, 8]
    base_amplitudes = [1.0, 1.0]

    # Add more high frequencies for higher difficulties
    if difficulty > 3:
        base_frequencies.extend([16, 32])
        base_amplitudes.extend([0.0625, 0.03125])
    
    # Generate random height field using multiple frequencies of noise
    Z = np.zeros((resolution, resolution))
    
    for freq, amp in zip(base_frequencies, base_amplitudes):
        noise = np.random.rand(resolution // freq + 1, resolution // freq + 1)
        noise_interp = interpolate_bilinear(noise, resolution)
        Z += noise_interp * amp
    
    # Normalize and scale to max height, then center around zero
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * max_height
    Z = Z - (max_height / 2)  # Shift down by half the max height
    
    # Generate vertices and faces
    vertices = []
    faces = []
    
    # Add vertices
    for i in range(resolution):
        for j in range(resolution):
            vertices.append([X[i,j], Y[i,j], Z[i,j]])
    
    # Add faces (triangles)
    for i in range(resolution-1):
        for j in range(resolution-1):
            v0 = i * resolution + j
            v1 = v0 + 1
            v2 = (i + 1) * resolution + j
            v3 = v2 + 1
            
            # Add two triangles for each quad
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return np.array(vertices), np.array(faces)

def create_flat_terrain(width, length):
    """Create a flat terrain with given dimensions."""
    vertices = np.array([
        [-width/2, -length/2, 0],  # bottom left
        [width/2, -length/2, 0],   # bottom right
        [width/2, length/2, 0],    # top right
        [-width/2, length/2, 0],   # top left
    ])
    
    faces = np.array([
        [0, 1, 2],  # first triangle
        [0, 2, 3]   # second triangle
    ])
    
    return vertices, faces

def interpolate_bilinear(array, new_size):
    """Simple bilinear interpolation to resize an array."""
    x = np.linspace(0, array.shape[0]-1, new_size)
    y = np.linspace(0, array.shape[1]-1, new_size)
    interpolating_function = interpolate.interp2d(np.arange(array.shape[1]), 
                                                np.arange(array.shape[0]), 
                                                array, 
                                                kind='linear')
    return interpolating_function(y, x)

def create_ground_mesh(max_x, max_y, difficulty=0):
    # Round up to nearest meter
    width = np.ceil(max_x * 2)  # *2 because we want to center at 0,0
    length = np.ceil(max_y * 2)
    
    vertices, faces = create_rough_terrain(width, length, difficulty=difficulty)
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def load_and_analyze_amass_files(directory):
    max_x = float('-inf')
    max_y = float('-inf')
    
    # Get all pkl files in the directory
    pkl_files = glob(os.path.join(directory, '*.pkl'))
    
    # First inspect one file
    if pkl_files:
        print(f"\nInspecting first file: {os.path.basename(pkl_files[0])}")
        first_data = inspect_data_structure(pkl_files[0])
    
    for file_path in pkl_files:
        print(f"\nProcessing {os.path.basename(file_path)}")
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # We'll print the keys of the first file to understand the structure
            if file_path == pkl_files[0]:
                print("Available keys:", list(data.keys()) if isinstance(data, dict) else "Not a dictionary")
            
            # Check root position
            if 'root_pos' in data:
                root_pos = data['root_pos']
                max_x = max(max_x, np.max(root_pos[:, 0]))
                max_y = max(max_y, np.max(root_pos[:, 1]))
            
            # Check all link positions
            if 'link_pos' in data:
                link_pos = data['link_pos']
                max_x = max(max_x, np.max(link_pos[..., 0]))
                max_y = max(max_y, np.max(link_pos[..., 1]))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return max_x, max_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground mesh for AMASS data')
    parser.add_argument('--difficulty', type=int, choices=[0, 1, 2, 3, 4], default=0,
                      help='Difficulty level (0=flat, 1=5cm, 2=10cm, 3=15cm, 4=20cm)')
    parser.add_argument('--directory', type=str, default=os.path.expanduser("~/Desktop/ACCAD_export_retargeted"),
                      help='Directory containing AMASS data')
    args = parser.parse_args()
    
    max_x, max_y = load_and_analyze_amass_files(args.directory)
    print(f"\nMaximum coordinates found:")
    print(f"Max X: {max_x:.3f}")
    print(f"Max Y: {max_y:.3f}")
    
    # Create and save the ground mesh
    ground_mesh = create_ground_mesh(max_x, max_y, difficulty=args.difficulty)
    
    # Create output directory if it doesn't exist
    output_dir = "resources/motions/amass"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the mesh
    difficulty_name = "flat" if args.difficulty == 0 else f"rough_d{args.difficulty}"
    output_path = os.path.join(output_dir, f"ground_mesh_{difficulty_name}.obj")
    ground_mesh.export(output_path)
    print(f"\nGround mesh saved to {output_path}")
    print(f"Mesh dimensions: {ground_mesh.bounds[1] - ground_mesh.bounds[0]}")
