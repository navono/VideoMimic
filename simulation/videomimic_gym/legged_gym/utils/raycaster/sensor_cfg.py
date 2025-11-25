from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import torch
import numpy as np

@dataclass
class RaycastingSensorCfg:
    """Base configuration for raycasting sensors."""
    
    name: str = ""
    """Name of the sensor."""
    
    type: str = field(init=False)
    """Type of sensor (set by subclass)."""
    
    enabled: bool = True
    """Whether the sensor is enabled."""
    
    body_name: str = ""
    """Name of the body the sensor is attached to."""
    
    max_distance: float = 5.0
    """Maximum distance for raycasting."""
    
    only_heading: bool = False
    """Whether to only use the heading (yaw) rotation of the body."""

    max_delay: int = 0
    """Maximum delay for the sensor."""

    device: str = 'cuda' # TODO -- should probably inherit from robot

    update_frequency_min: int = 0
    """Minimum update frequency for the sensor."""

    update_frequency_max: int = 0
    """Maximum update frequency for the sensor."""
    
    def __post_init__(self):
        self.type = self.__class__.__name__.lower().replace("sensorcfg", "")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

@dataclass
class HeightfieldSensorCfg(RaycastingSensorCfg):
    """Configuration for heightfield sensor."""
    
    size: Tuple[float, float] = (1.0, 1.0)
    """Size of the heightfield grid (length, width) in meters."""
    
    resolution: float = 0.1
    """Resolution of the heightfield grid in meters."""
    
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Direction of the rays."""
    
    ordering: str = "xy"
    """Ordering of the grid points ('xy' or 'yx')."""
    
    use_float: bool = False
    """Whether to use float32 instead of uint8 for the depth map."""

    white_noise_scale: float = 0.0
    """Scale of the Gaussian noise to add to the depth map (per-timestep)."""

    offset_noise_scale: float = 0.0
    """Scale of the offset noise to add to the depth map (per-env)."""

    roll_noise_scale: float = 0.0
    """Scale of the roll noise to add to the sensor orientation (radians)."""

    pitch_noise_scale: float = 0.0
    """Scale of the pitch noise to add to the sensor orientation (radians)."""

    yaw_noise_scale: float = 0.0
    """Scale of the yaw noise to add to the sensor orientation (radians)."""

    bad_distance_prob: float = 0.0
    """Probability of a bad distance (i.e. a completely random value) to be returned."""

@dataclass
class MultiLinkHeightSensorCfg(RaycastingSensorCfg):
    """Configuration for multi-link height sensor."""
    
    link_names: List[str] = field(default_factory=list)
    """List of link names to measure height for."""
    
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Direction of the rays."""
    
    use_float: bool = True
    """Whether to use float32 for the depth map."""

@dataclass
class DepthCameraSensorCfg(RaycastingSensorCfg):
    """Configuration for depth camera sensor."""
    
    width: int = 320
    """Width of the depth image in pixels."""
    
    height: int = 240
    """Height of the depth image in pixels."""
    
    downsample_factor: int = 1
    """Factor to downsample the image (must be integer)."""

    intrinsic_matrix: Optional[np.ndarray] = None
    """Camera intrinsic matrix (3x3). If None, uses default."""
    
    # Default D435 intrinsics (adjust as needed)
    default_intrinsic_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [384.0, 0.0, 320.0],
        [0.0, 384.0, 240.0],
        [0.0, 0.0, 1.0]
    ]), repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.downsample_factor > 1:
            self.width //= self.downsample_factor
            self.height //= self.downsample_factor
            if self.intrinsic_matrix is None:
                self.intrinsic_matrix = self.default_intrinsic_matrix.copy()
            # Adjust intrinsics for downsampling
            self.intrinsic_matrix[0, 0] /= self.downsample_factor  # fx
            self.intrinsic_matrix[1, 1] /= self.downsample_factor  # fy
            self.intrinsic_matrix[0, 2] /= self.downsample_factor  # cx
            self.intrinsic_matrix[1, 2] /= self.downsample_factor  # cy

@dataclass
class LeggedRobotSensorsCfg:
    """Container for sensor configurations."""
    sensor_cfgs: List[Union[HeightfieldSensorCfg, DepthCameraSensorCfg, MultiLinkHeightSensorCfg]] = field(default_factory=list) 