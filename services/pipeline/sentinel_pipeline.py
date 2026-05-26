#!/usr/bin/env python3
"""Sentinel Orin — DeepStream detection + tracking pipeline.

Pipeline:
    uridecodebin -> nvstreammux -> nvinfer (YOLO26n PGIE FP16)
    -> nvtracker (NvDCF) -> fakesink

    Pad probe on nvtracker src pad logs per-camera track IDs to stdout.

Usage:
    # Single stream:
    python3 pipeline/sentinel_pipeline.py \
        --videos /data/sentinel/videos/wildtrack_v1/cam01_1080p60.mp4

    # Three streams:
    python3 pipeline/sentinel_pipeline.py \
        --videos /data/sentinel/videos/wildtrack_v1/cam01_1080p60.mp4 \
                 /data/sentinel/videos/wildtrack_v1/cam03_1080p60.mp4 \
                 /data/sentinel/videos/wildtrack_v1/cam05_1080p60.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

try:
    import pyds
except ImportError:
    sys.exit(
        "pyds not found. Run this script inside the DeepStream container.\n"
        "  docker run -it --rm --runtime=nvidia \\\n"
        "    -v $(pwd):/opt/sentinel \\\n"
        "    -v /data/sentinel:/data/sentinel \\\n"
        "    nvcr.io/nvidia/deepstream:7.1-samples-multiarch \\\n"
        "    python3 /opt/sentinel/services/pipeline/sentinel_pipeline.py --videos ..."
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DS_ROOT = Path("/opt/sentinel/deepstream")
PGIE_CONFIG = DS_ROOT / "configs" / "yolo26n_pgie.txt"
PARSER_LIB = DS_ROOT / "custom_parser" / "libnvdsinfer_custom_yolo26.so"
TRACKER_CONFIG = DS_ROOT / "configs" / "tracker_nvdcf_perf.yml"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_frame_count = 0
_detect_count = 0
_t_start = 0.0


# ---------------------------------------------------------------------------
# Tracking pad probe
# ---------------------------------------------------------------------------
def _on_tracker_src(pad, info, _user_data):
    global _frame_count, _detect_count

    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        _frame_count += 1

        n_obj = 0
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            n_obj += 1
            _detect_count += 1

            # Log one tracked object per frame every 60 frames
            if _frame_count % 60 == 0 and n_obj == 1:
                r = obj.rect_params
                print(
                    f"[frame {_frame_count:>6}] "
                    f"cam={frame_meta.pad_index} "
                    f"track_id={obj.object_id} "
                    f"conf={obj.confidence:.2f} "
                    f"box=({r.left:.0f},{r.top:.0f},{r.width:.0f},{r.height:.0f})"
                )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # FPS report every 300 frames
        if _frame_count % 300 == 0 and _frame_count > 0:
            elapsed = time.time() - _t_start
            fps = _frame_count / elapsed if elapsed > 0 else 0
            print(
                f"[fps] frames={_frame_count} "
                f"detections={_detect_count} "
                f"fps={fps:.1f}"
            )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def _make_element(factory: str, name: str) -> Gst.Element:
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        sys.exit(f"Could not create GStreamer element: {factory} ({name})")
    return el


def build_pipeline(video_paths: list[str]) -> tuple[Gst.Pipeline, GLib.MainLoop]:
    Gst.init(None)

    n_sources = len(video_paths)
    print(f"[pipeline] sources={n_sources}")
    for p in video_paths:
        print(f"  {p}")

    for path in [PGIE_CONFIG, PARSER_LIB, TRACKER_CONFIG]:
        if not path.exists():
            sys.exit(f"Required file not found: {path}")

    pipeline = Gst.Pipeline.new("sentinel-pipeline")

    # streammux
    streammux = _make_element("nvstreammux", "streammux")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", n_sources)
    streammux.set_property("batched-push-timeout", 16666)
    streammux.set_property("live-source", 0)
    pipeline.add(streammux)

    # sources
    for i, path in enumerate(video_paths):
        uri = path if path.startswith("file://") else f"file://{path}"
        src = _make_element("uridecodebin", f"src-{i}")
        src.set_property("uri", uri)

        def _on_pad_added(element, pad, idx=i):
            if pad.get_current_caps() is None:
                return
            if "video" not in pad.get_current_caps().to_string():
                return
            sink_pad = streammux.get_request_pad(f"sink_{idx}")
            if sink_pad and not sink_pad.is_linked():
                pad.link(sink_pad)

        src.connect("pad-added", _on_pad_added)
        pipeline.add(src)

    # nvinfer PGIE
    pgie = _make_element("nvinfer", "pgie")
    pgie.set_property("config-file-path", str(PGIE_CONFIG))
    pipeline.add(pgie)

    # nvtracker NvDCF
    tracker = _make_element("nvtracker", "tracker")
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", str(TRACKER_CONFIG))
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("display-tracking-id", 1)
    pipeline.add(tracker)

    # fakesink
    sink = _make_element("fakesink", "sink")
    sink.set_property("sync", 0)
    pipeline.add(sink)

    # link: streammux -> pgie -> tracker -> sink
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sink)

    # pad probe on tracker src (not pgie src — we want post-tracking metadata)
    tracker_src = tracker.get_static_pad("src")
    if not tracker_src:
        sys.exit("Could not get tracker src pad")
    tracker_src.add_probe(Gst.PadProbeType.BUFFER, _on_tracker_src, None)

    # bus
    loop = GLib.MainLoop()

    def _on_bus_message(bus, msg, _loop):
        if msg.type == Gst.MessageType.EOS:
            print("\n[pipeline] End of stream.")
            _loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[pipeline] ERROR: {err.message}")
            if dbg:
                print(f"[pipeline] debug: {dbg}")
            _loop.quit()
        elif msg.type == Gst.MessageType.WARNING:
            warn, _ = msg.parse_warning()
            print(f"[pipeline] WARNING: {warn.message}")

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _on_bus_message, loop)

    return pipeline, loop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Sentinel Orin detection + tracking pipeline")
    ap.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Path(s) to input video file(s). One for smoke test, three for full pipeline.",
    )
    args = ap.parse_args()

    for p in args.videos:
        if not Path(p).exists():
            sys.exit(f"Video not found: {p}")

    if len(args.videos) > 3:
        sys.exit("Maximum 3 video sources (cam01, cam03, cam05).")

    pipeline, loop = build_pipeline(args.videos)

    global _t_start
    _t_start = time.time()

    print("\n[pipeline] Starting...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted.")
    finally:
        elapsed = time.time() - _t_start
        fps = _frame_count / elapsed if elapsed > 0 else 0
        print(
            f"\n[summary] frames={_frame_count} detections={_detect_count} "
            f"elapsed={elapsed:.1f}s fps={fps:.1f}"
        )
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
