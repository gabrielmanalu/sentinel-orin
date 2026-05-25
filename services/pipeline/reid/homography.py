"""Image-to-ground-plane projection using WILDTRACK homographies.

Projects a detection's bottom-center pixel to world (X, Y)
coordinates on the shared ground plane, so detections from different
cameras can be compared in a common frame.
"""
from __future__ import annotations

import numpy as np


def project_to_ground(point_xy: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Project an image pixel (x, y) to world ground-plane (X, Y).

    Args:
        point_xy: pixel coordinate (bottom-center of a bbox).
        homography: 3x3 image->world homography matrix for the camera.

    Returns:
        (X, Y) world coordinate in meters.
    """
    px = np.array([point_xy[0], point_xy[1], 1.0])
    world = homography @ px
    if abs(world[2]) < 1e-9:
        raise ValueError("Degenerate homography projection (w ~= 0)")
    return float(world[0] / world[2]), float(world[1] / world[2])


def bbox_bottom_center(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bottom-center pixel of an [x1, y1, x2, y2] bbox."""
    x1, _y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)
