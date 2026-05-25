import numpy as np

from services.pipeline.reid.embedding_buffer import EmbeddingBuffer, Observation


def _obs(cam, tid, t):
    return Observation(cam, tid, np.zeros(512), (0.0, 0.0), t)


def test_add_and_candidates_excludes_same_camera():
    buf = EmbeddingBuffer()
    buf.add(_obs("cam01", 1, 100.0))
    buf.add(_obs("cam03", 2, 100.0))
    cands = buf.candidates(exclude_camera="cam01")
    assert len(cands) == 1
    assert cands[0].camera_id == "cam03"


def test_prune_removes_old_observations():
    buf = EmbeddingBuffer(retention_seconds=10.0)
    buf.add(_obs("cam01", 1, 100.0))
    buf.add(_obs("cam01", 2, 105.0))
    buf.prune(now=120.0)  # cutoff = 110, both older
    assert len(buf.candidates(exclude_camera="cam99")) == 0


def test_prune_keeps_recent():
    buf = EmbeddingBuffer(retention_seconds=10.0)
    buf.add(_obs("cam01", 1, 100.0))
    buf.add(_obs("cam01", 2, 115.0))
    buf.prune(now=120.0)  # cutoff = 110, keeps the 115 one
    assert len(buf.candidates(exclude_camera="cam99")) == 1
