"""ROS 2 node: events -> RViz2 markers on a 2D ground-plane map.

Subscribes /sentinel/events, publishes visualization_msgs/MarkerArray
to /sentinel/markers. Color: green=active, yellow=in-zone, red=dwell,
blue=just-handed-off.
"""
from __future__ import annotations
