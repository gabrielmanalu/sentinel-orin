"""INT8 PTQ calibration frame selection for YOLO26n.

Stratified sampling: 80 frames/camera x 3 cameras = 240 frames,
spread across crowd densities. A/B comparison vs 500-frame set
documented in BENCHMARKS.md. Frame indices committed for reproducibility.
"""
from __future__ import annotations

N_PER_CAMERA = 80
CAMERAS = ("cam01", "cam03", "cam05")
