"""Ground-plane homography computation from WILDTRACK calibration files.

WILDTRACK provides per-camera extrinsic (rvec, tvec) and intrinsic
(camera_matrix) calibration in OpenCV XML format. This module computes
the image->ground-plane homography for each camera.

Camera name mapping:
    C1 / cam1 -> extr_CVLab1.xml / intr_CVLab1.xml
    C2 / cam2 -> extr_CVLab2.xml / intr_CVLab2.xml
    C3 / cam3 -> extr_CVLab3.xml / intr_CVLab3.xml
    C4 / cam4 -> extr_CVLab4.xml / intr_CVLab4.xml
    C5 / cam5 -> extr_IDIAP1.xml / intr_IDIAP1.xml
    C6 / cam6 -> extr_IDIAP2.xml / intr_IDIAP2.xml
    C7 / cam7 -> extr_IDIAP3.xml / intr_IDIAP3.xml

Ground plane coordinate system: z=0, origin at (-300, -90) cm,
grid 1440x480 cm with 2.5 cm step (from WILDTRACK paper).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# Camera index (1-7) to calibration file stem
_CAM_NAMES = {
    1: "CVLab1",
    2: "CVLab2",
    3: "CVLab3",
    4: "CVLab4",
    5: "IDIAP1",
    6: "IDIAP2",
    7: "IDIAP3",
}

# Cameras used in Sentinel (cam1, cam3, cam5 -> indices 1, 3, 5)
SENTINEL_CAMS = (1, 3, 5)


def _parse_vec(xml_path: Path, tag: str) -> np.ndarray:
    """Parse a whitespace-separated float array from an XML element."""
    root = ET.parse(xml_path).getroot()
    el = root.find(tag)
    if el is None:
        raise ValueError(f"Tag '{tag}' not found in {xml_path}")
    return np.array([float(v) for v in el.text.split()], dtype=np.float64)


def _parse_matrix(xml_path: Path, tag: str, rows: int, cols: int) -> np.ndarray:
    """Parse an OpenCV matrix from XML."""
    root = ET.parse(xml_path).getroot()
    el = root.find(tag)
    if el is None:
        raise ValueError(f"Tag '{tag}' not found in {xml_path}")
    data_el = el.find("data")
    if data_el is None:
        raise ValueError(f"No <data> under '{tag}' in {xml_path}")
    vals = [float(v) for v in data_el.text.split()]
    return np.array(vals, dtype=np.float64).reshape(rows, cols)


def compute_homography(
    calib_dir: Path,
    cam_idx: int,
) -> np.ndarray:
    """Compute image->ground-plane homography for a WILDTRACK camera.

    Args:
        calib_dir: Path to the WILDTRACK calibrations/ directory.
        cam_idx: Camera index 1-7.

    Returns:
        3x3 homography matrix H such that:
            [X, Y, 1]^T ~ H @ [u, v, 1]^T
        where (u,v) is a pixel and (X,Y) is the ground-plane position in cm.
    """
    name = _CAM_NAMES[cam_idx]
    extr_path = calib_dir / "extrinsic" / f"extr_{name}.xml"
    intr_path = calib_dir / "intrinsic_zero" / f"intr_{name}.xml"

    rvec = _parse_vec(extr_path, "rvec").reshape(3, 1)
    tvec = _parse_vec(extr_path, "tvec").reshape(3, 1)
    K = _parse_matrix(intr_path, "camera_matrix", 3, 3)

    # Rotation matrix from rvec
    R, _ = cv2.Rodrigues(rvec)

    # Homography from image to ground plane (Z=0):
    # H = K @ [r1 | r2 | t]  (drop r3 since Z=0)
    # Then invert to get image->world
    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    H_world_to_img = K @ np.hstack([r1, r2, tvec])
    H_img_to_world = np.linalg.inv(H_world_to_img)
    return H_img_to_world


def load_sentinel_homographies(calib_dir: Path) -> dict[int, np.ndarray]:
    """Load homographies for cam1, cam3, cam5.

    Returns:
        Dict mapping camera index to 3x3 homography matrix.
    """
    return {
        cam: compute_homography(calib_dir, cam)
        for cam in SENTINEL_CAMS
    }


def project_to_ground(
    pixel: tuple[float, float],
    H: np.ndarray,
) -> tuple[float, float]:
    """Project an image pixel to ground-plane (X, Y) in centimeters.

    Args:
        pixel: (u, v) pixel coordinate (e.g. bbox bottom-center).
        H: 3x3 image->ground homography.

    Returns:
        (X, Y) in centimeters on the ground plane.
    """
    p = np.array([pixel[0], pixel[1], 1.0])
    w = H @ p
    if abs(w[2]) < 1e-9:
        raise ValueError("Degenerate homography projection")
    return float(w[0] / w[2]), float(w[1] / w[2])


def bbox_bottom_center(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return bottom-center pixel of an [x1, y1, x2, y2] bbox."""
    x1, _y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, y2
