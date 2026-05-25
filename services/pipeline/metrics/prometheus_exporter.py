"""Prometheus metric definitions + HTTP exporter.

Exposes pipeline FPS, event counts, re-ID match scores, and Jetson
hardware telemetry (RAM, GPU%, temperature, power) on :9100.
"""
from __future__ import annotations
