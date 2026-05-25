"""Main DeepStream pipeline entry point.

Builds the GStreamer pipeline:
  3x uridecodebin -> nvstreammux -> nvinfer(YOLO26n PGIE)
  -> nvtracker(NvDCF) -> nvinfer(OSNet SGIE) -> pad probe
  -> nvdsanalytics -> fakesink

The pad probe extracts OSNet embeddings, runs cross-camera re-ID,
emits zone events, and publishes Prometheus metrics.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Pipeline assembled+.")


if __name__ == "__main__":
    main()
