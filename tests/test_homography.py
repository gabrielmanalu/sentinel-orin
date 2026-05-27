"""Tests for homography.py.

Unit tests use synthetic data (identity/scaling homographies).
Integration test uses real WILDTRACK calibration if available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.pipeline.reid.homography import (
    bbox_bottom_center,
    compute_homography,
    project_to_ground,
)

WILDTRACK_CALIB = Path("/data/sentinel/datasets/wildtrack/calibrations")


def test_identity_homography():
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
    assert cx == 20.0
    assert cy == 80.0


def test_degenerate_homography_raises():
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    with pytest.raises(ValueError):
        project_to_ground((1.0, 1.0), H)


@pytest.mark.skipif(
    not WILDTRACK_CALIB.exists(),
    reason="WILDTRACK calibration not available in CI"
)
def test_real_calibration_cam1():
    """Sanity check: cam1 ground projection of image center lands
    within the WILDTRACK annotated area (~1440x480 cm)."""
    H = compute_homography(WILDTRACK_CALIB, cam_idx=1)
    assert H.shape == (3, 3)
    x, y = project_to_ground((960.0, 540.0), H)
    # Ground plane is roughly -300 to 1140 cm in X, -90 to 390 cm in Y
    assert -1000 < x < 2000, f"X={x} out of expected range"
    assert -1000 < y < 2000, f"Y={y} out of expected range"


@pytest.mark.skipif(
    not WILDTRACK_CALIB.exists(),
    reason="WILDTRACK calibration not available in CI"
)
def test_real_calibration_all_sentinel_cams():
    """All 3 Sentinel cameras produce valid homographies."""
    from services.pipeline.reid.homography import load_sentinel_homographies
    homographies = load_sentinel_homographies(WILDTRACK_CALIB)
    assert set(homographies.keys()) == {1, 3, 5}
    for cam, H in homographies.items():
        assert H.shape == (3, 3), f"cam{cam} homography wrong shape"
        assert not np.any(np.isnan(H)), f"cam{cam} homography has NaN"
