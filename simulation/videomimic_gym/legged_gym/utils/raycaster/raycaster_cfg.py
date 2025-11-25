from __future__ import annotations
import torch
from legged_gym.utils.configclass import configclass
from typing import Callable
from dataclasses import MISSING

from .raycaster_patterns import grid_pattern, pinhole_camera_pattern

@configclass
class PatternBaseCfg:
    """Base configuration for a pattern."""

    func: Callable = MISSING
    """Function to generate the pattern.

    The function should take in the configuration and the device name as arguments. It should return
    the pattern's starting positions and directions as a tuple of torch.Tensor.
    """


@configclass
class GridPatternCfg(PatternBaseCfg):
    """Configuration for the grid pattern for ray-casting.

    Defines a 2D grid of rays in the coordinates of the sensor.

    .. attention::
        The points are ordered based on the :attr:`ordering` attribute.

    """

    func: Callable = grid_pattern

    resolution: float = MISSING
    """Grid resolution (in meters)."""

    size: tuple[float, float] = MISSING
    """Grid size (length, width) (in meters)."""

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Ray direction. Defaults to (0.0, 0.0, -1.0)."""

    ordering: str = "xy"
    """Specifies the ordering of points in the generated grid. Defaults to ``"xy"``.

    Consider a grid pattern with points at :math:`(x, y)` where :math:`x` and :math:`y` are the grid indices.
    The ordering of the points can be specified as "xy" or "yx". This determines the outer and inner loop order
    when iterating over the grid points.

    * If *"xy"* is selected, the points are ordered with outer loop over "x" and inner loop over "y".
    * If *"yx"* is selected, the points are ordered with outer loop over "y" and inner loop over "x".

    For example, the grid pattern points with :math:`X = (0, 1, 2)` and :math:`Y = (3, 4)`:

    * *"xy"* ordering: :math:`[(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]`
    * *"yx"* ordering: :math:`[(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (2, 4)]`
    """



@configclass
class PinholeCameraPatternCfg(PatternBaseCfg):
    """Configuration for a pinhole camera depth image pattern for ray-casting.

    .. caution::
        Focal length as well as the aperture sizes and offsets are set as a tenth of the world unit. In our case, the
        world unit is meters, so all of these values are in cm. For more information, please check:
        https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html
    """

    func: Callable = pinhole_camera_pattern

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0cm.

    Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
    """

    horizontal_aperture: float = 20.955
    """Horizontal aperture (in cm). Defaults to 20.955 cm.

    Emulates sensor/film width on a camera.

    Note:
        The default value is the horizontal aperture of a 35 mm spherical projector.
    """
    vertical_aperture: float | None = None
    r"""Vertical aperture (in cm). Defaults to None.

    Emulates sensor/film height on a camera. If None, then the vertical aperture is calculated based on the
    horizontal aperture and the aspect ratio of the image to maintain squared pixels. In this case, the vertical
    aperture is calculated as:

    .. math::
        \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
    """

    horizontal_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""

    vertical_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate vertically. Defaults to 0.0."""

    width: int = MISSING
    """Width of the image (in pixels)."""

    height: int = MISSING
    """Height of the image (in pixels)."""

    @classmethod
    def from_intrinsic_matrix(
        cls,
        intrinsic_matrix: list[float],
        width: int,
        height: int,
        focal_length: float = 24.0,
    ) -> PinholeCameraPatternCfg:
        r"""Create a :class:`PinholeCameraPatternCfg` class instance from an intrinsic matrix.

        The intrinsic matrix is a 3x3 matrix that defines the mapping between the 3D world coordinates and
        the 2D image. The matrix is defined as:

        .. math::
            I_{cam} = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix},

        where :math:`f_x` and :math:`f_y` are the focal length along x and y direction, while :math:`c_x` and :math:`c_y` are the
        principle point offsets along x and y direction respectively.

        Args:
            intrinsic_matrix: Intrinsic matrix of the camera in row-major format.
                The matrix is defined as [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]. Shape is (9,).
            width: Width of the image (in pixels).
            height: Height of the image (in pixels).
            focal_length: Focal length of the camera (in cm). Defaults to 24.0 cm.

        Returns:
            An instance of the :class:`PinholeCameraPatternCfg` class.
        """
        # extract parameters from matrix
        f_x = intrinsic_matrix[0]
        c_x = intrinsic_matrix[2]
        f_y = intrinsic_matrix[4]
        c_y = intrinsic_matrix[5]
        # resolve parameters for usd camera
        horizontal_aperture = width * focal_length / f_x
        vertical_aperture = height * focal_length / f_y
        horizontal_aperture_offset = (c_x - width / 2) / f_x
        vertical_aperture_offset = (c_y - height / 2) / f_y

        return cls(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            horizontal_aperture_offset=horizontal_aperture_offset,
            vertical_aperture_offset=vertical_aperture_offset,
            width=width,
            height=height,
        )

