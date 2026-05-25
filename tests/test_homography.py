import numpy as np

from services.pipeline.reid.homography import bbox_bottom_center, project_to_ground


def test_identity_homography_returns_input():
    H = np.eye(3)
    x, y = project_to_ground((4.0, 7.0), H)
    assert abs(x - 4.0) < 1e-9
    assert abs(y - 7.0) < 1e-9


def test_scaling_homography():
    H = np.diag([2.0, 3.0, 1.0])
    x, y = project_to_ground((1.0, 1.0), H)
    assert abs(x - 2.0) < 1e-9
    assert abs(y - 3.0) < 1e-9


def test_bbox_bottom_center():
    cx, cy = bbox_bottom_center((10.0, 20.0, 30.0, 80.0))
    assert cx == 20.0  # (10+30)/2
    assert cy == 80.0  # bottom edge


def test_degenerate_homography_raises():
    import pytest
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    with pytest.raises(ValueError):
        project_to_ground((1.0, 1.0), H)
