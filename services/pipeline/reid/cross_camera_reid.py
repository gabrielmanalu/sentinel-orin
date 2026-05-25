"""Cross-camera global-ID coordinator (the project's technical core).

Maintains global identities across cameras using appearance
embeddings + ground-plane geometry. Emits cross_camera_handoff events.

Global ID lifecycle: NEW -> MATCHED -> LOST -> RETIRED.
"""
from __future__ import annotations

# Implementation: See docs/REID_DESIGN.md.
