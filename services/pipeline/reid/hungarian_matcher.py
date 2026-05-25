"""Hungarian assignment over appearance + spatial match scores.

Combines cosine similarity of OSNet embeddings with a spatial
gate (ground-plane distance) and solves optimal assignment.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHA = 0.7  # appearance weight
BETA = 0.3   # spatial weight
SPATIAL_SIGMA = 1.5  # meters
SCORE_THRESHOLD = 0.55


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def match_score(emb_a: np.ndarray, xy_a: tuple[float, float],
                emb_b: np.ndarray, xy_b: tuple[float, float]) -> float:
    appearance = cosine_similarity(emb_a, emb_b)
    dist2 = (xy_a[0] - xy_b[0]) ** 2 + (xy_a[1] - xy_b[1]) ** 2
    spatial = np.exp(-dist2 / (2 * SPATIAL_SIGMA ** 2))
    return ALPHA * appearance + BETA * spatial


def solve(score_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (row, col) matches above SCORE_THRESHOLD.

    Maximizes total score via Hungarian algorithm on the negated matrix.
    """
    if score_matrix.size == 0:
        return []
    rows, cols = linear_sum_assignment(-score_matrix)
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=False)
            if score_matrix[r, c] >= SCORE_THRESHOLD]
