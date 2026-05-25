"""FastAPI events + evidence API.

Read-only JSON API over the SQLite event store. Visualization lives in
Grafana; this is the programmatic interface.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError:  # allows import during host-side lint without fastapi
    FastAPI = None  # type: ignore

if FastAPI is not None:
    app = FastAPI(title="Sentinel Orin API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}
