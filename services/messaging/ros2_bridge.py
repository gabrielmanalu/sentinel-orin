"""MQTT -> ROS 2 bridge node.

Subscribes to MQTT sentinel/events, republishes to ROS 2 topic
/sentinel/events for downstream robotics consumers.
"""
from __future__ import annotations
