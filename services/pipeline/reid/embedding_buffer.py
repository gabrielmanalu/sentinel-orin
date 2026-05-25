"""Rolling buffer of recent track observations across cameras.

Holds (camera_id, track_id, embedding, world_xy, timestamp)
tuples within a bounded time window, for cross-camera matching.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    camera_id: str
    track_id: int
    embedding: np.ndarray  # 512-dim OSNet appearance vector
    world_xy: tuple[float, float]
    timestamp: float


class EmbeddingBuffer:
    """Time-bounded buffer of cross-camera observations."""

    def __init__(self, retention_seconds: float = 30.0, max_len: int = 4096):
        self.retention = retention_seconds
        self._buf: deque[Observation] = deque(maxlen=max_len)

    def add(self, obs: Observation) -> None:
        self._buf.append(obs)

    def prune(self, now: float) -> None:
        cutoff = now - self.retention
        while self._buf and self._buf[0].timestamp < cutoff:
            self._buf.popleft()

    def candidates(self, exclude_camera: str) -> list[Observation]:
        """Observations from cameras other than the given one."""
        return [o for o in self._buf if o.camera_id != exclude_camera]
