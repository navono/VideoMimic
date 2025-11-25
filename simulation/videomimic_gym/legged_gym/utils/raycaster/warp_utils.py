"""Warp mesh utilities for raycasting."""

import torch
import warp as wp
import numpy as np
from typing import Optional, Tuple

def convert_to_warp_mesh(points: np.ndarray, indices: np.ndarray, device: str) -> wp.Mesh:
    """Create a warp mesh object with a mesh defined from vertices and triangles.

    Args:
        points: The vertices of the mesh. Shape is (N, 3), where N is the number of vertices.
        indices: The triangles of the mesh as references to vertices for each triangle.
            Shape is (M, 3), where M is the number of triangles / faces.
        device: The device to use for the mesh.

    Returns:
        The warp mesh object.
    """
    return wp.Mesh(
        points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(indices.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )

def raycast_mesh(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    mesh: wp.Mesh,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Performs ray-casting against a mesh.

    Note that the `ray_starts` and `ray_directions`, and `ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    Args:
        ray_starts: The starting position of the rays. Shape (N, 3).
        ray_directions: The ray directions for each ray. Shape (N, 3).
        mesh: The warp mesh to ray-cast against.
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.

    Returns:
        The ray hit position. Shape (N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
    """
    # extract device and shape information
    original_shape = ray_starts.shape  # Store original shape
    device = ray_starts.device
    # device of the mesh
    torch_device = wp.device_to_torch(mesh.device)
    # reshape the tensors
    ray_starts = ray_starts.to(torch_device).reshape(-1, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).reshape(-1, 3).contiguous()
    num_rays = ray_starts.shape[0]
    # create output tensor for the ray hits
    ray_hits = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()

    # map the memory to warp arrays
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
    else:
        ray_distance = None
        ray_distance_wp = wp.empty((1,), dtype=wp.float32, device=torch_device)

    if return_normal:
        ray_normal = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1,), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.ones((num_rays,), dtype=torch.int32, device=torch_device).contiguous() * (-1)
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1,), dtype=wp.int32, device=torch_device)

    # launch the warp kernel
    wp.launch(
        kernel=raycast_mesh_kernel,
        dim=num_rays,
        inputs=[
            mesh.id,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distance_wp,
            ray_normal_wp,
            ray_face_id_wp,
            float(max_dist),
            int(return_distance),
            int(return_normal),
            int(return_face_id),
        ],
        device=mesh.device,
    )
    wp.synchronize()

    # Restore original shapes
    ray_hits = ray_hits.to(device).reshape(original_shape)
    if return_distance:
        # For distances, we want a single value per ray
        ray_distance = ray_distance.to(device).reshape(original_shape[:-1])  # Remove the last dimension
    if return_normal:
        ray_normal = ray_normal.to(device).reshape(original_shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).reshape(original_shape[:-1])  # Remove the last dimension

    return ray_hits, ray_distance, ray_normal, ray_face_id

@wp.kernel(enable_backward=False)
def raycast_mesh_kernel(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    ray_normal: wp.array(dtype=wp.vec3),
    ray_face_id: wp.array(dtype=wp.int32),
    max_dist: float = 1e6,
    return_distance: int = False,
    return_normal: int = False,
    return_face_id: int = False,
):
    """Performs ray-casting against a mesh.

    This function performs ray-casting against the given mesh using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    Note that the `ray_starts`, `ray_directions`, and `ray_hits` arrays should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    The function utilizes the `mesh_query_ray` method from the `wp` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to `1e6` units.

    Args:
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (N, 3).
        ray_directions: The input ray directions. Shape is (N, 3).
        ray_hits: The output ray hit positions. Shape is (N, 3).
        ray_distance: The output ray hit distances. Shape is (N,), if `return_distance` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (N, 3), if `return_normal` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (N,), if `return_face_id` is True. Otherwise,
            this array is not used.
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_distance: Whether to return the ray hit distances. Defaults to False.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
    """
    # get the thread id
    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # ray cast against the mesh and store the hit position
    hit_success = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)
    # if the ray hit, store the hit data
    if hit_success:
        ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]
        if return_distance == 1:
            ray_distance[tid] = t
        if return_normal == 1:
            ray_normal[tid] = n
        if return_face_id == 1:
            ray_face_id[tid] = f 