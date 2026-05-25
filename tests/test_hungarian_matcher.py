import numpy as np

from services.pipeline.reid.hungarian_matcher import (
    cosine_similarity,
    match_score,
    solve,
)


def test_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_match_score_high_for_same_person():
    emb = np.array([1.0, 0.0, 0.0])
    score = match_score(emb, (0.0, 0.0), emb, (0.1, 0.1))
    assert score > 0.9  # same appearance, near location


def test_match_score_low_for_distant_different():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    score = match_score(a, (0.0, 0.0), b, (50.0, 50.0))
    assert score < 0.55  # different appearance + far apart


def test_solve_returns_above_threshold_only():
    m = np.array([[0.9, 0.1], [0.2, 0.8]])
    matches = solve(m)
    assert (0, 0) in matches
    assert (1, 1) in matches


def test_solve_empty_matrix():
    assert solve(np.array([])) == []
