"""Event construction + cross-camera-aware deduplication.

Emits the five Sentinel event types:
  restricted_zone_entry, restricted_zone_dwell, crowding_in_zone,
  line_crossing, cross_camera_handoff.
"""
from __future__ import annotations

EVENT_TYPES = (
    "restricted_zone_entry",
    "restricted_zone_dwell",
    "crowding_in_zone",
    "line_crossing",
    "cross_camera_handoff",
)
