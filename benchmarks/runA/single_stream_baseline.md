# Run A — Detection + Tracking baseline

Establishes throughput ceiling for the detection and tracking pipeline
before adding OSNet SGIE and cross-camera re-ID.

Hardware: Jetson Orin Nano Super 8GB
Power mode: 25W (sustained — MAXN_SUPER triggers over-current throttle under 3-stream load)
Clocks: jetson_clocks locked
Engine: yolo26n_fp16_b3_640 (FP16, batch=3, [1,300,6] end-to-end format)
Parser: libnvdsinfer_custom_yolo26.so

---

## Single stream — cam01, 1080p60, detection only

| Metric | Value |
|---|---|
| Stream | cam01_1080p60.mp4 |
| Resolution | 1920x1080 |
| Source FPS | 59.94 |
| Inference FPS (stable) | 184.3 |
| Total frames | 98,685 |
| Total detections | 916,255 |
| Avg detections/frame | ~9.3 |
| Elapsed | 535.5s |

Warm-up: 126.9 FPS at frame 300 → stable 184.3 FPS from ~frame 98,000.

---

## 3 streams — cam01 + cam03 + cam05, 1080p60, detection only

| Metric | Value |
|---|---|
| Streams | cam01, cam03, cam05 |
| Resolution | 1920x1080 per stream |
| Source FPS | 59.94 per stream |
| Aggregate FPS (stable) | 234.1 |
| Total frames | 190,306 |
| Total detections | 848,284 |
| Avg detections/frame | ~4.5 |
| Elapsed | 813.0s |

Warm-up: 166.6 FPS at frame 300 → stable 234.1 FPS from ~frame 50,000.

---

## 3 streams — cam01 + cam03 + cam05, 1080p60, detection + NvDCF tracker

| Metric | Value |
|---|---|
| Streams | cam01, cam03, cam05 |
| Aggregate FPS (stable) | 234.3 |
| FPS cost vs detection-only | -0.8 FPS (~0.3% overhead) |
| Total frames | 107,210 |
| Total detections | 448,779 |
| Elapsed | 457.6s |
| Tracker | NvDCF, tracker-width=640, tracker-height=384 |

NvDCF overhead is negligible — within measurement noise of the
detection-only baseline (234.1 FPS).

Note: track ID count inflates rapidly (~5000+ over partial run) due to
low-confidence detections flickering in/out creating short-lived tracks.
Confidence threshold and probationAge tuning deferred to re-ID evaluation.

---

## Observations

**Batch efficiency:** Adding 2 streams to a batch=3 engine costs only ~50 FPS
aggregate (184 → 234). The GPU was underutilized on a single stream — the
batch=3 engine amortizes inference cost across all 3 frames per forward pass.

**25W vs MAXN_SUPER:** MAXN_SUPER triggered over-current throttling under
3-stream sustained load. 25W is the correct benchmark mode — stable,
reproducible, and realistic for a deployed appliance.

**Headroom:** 234 FPS detection+tracking gives substantial budget for OSNet
SGIE and re-ID. Target aggregate FPS with full pipeline is 45+. Current
headroom is 5.2× above that target.

---

## Cumulative pipeline FPS

| Pipeline stage | Aggregate FPS |
|---|---|
| Detection only (1 stream) | 184.3 |
| Detection only (3 streams) | 234.1 |
| + NvDCF tracker (3 streams) | 234.3 |
| + OSNet SGIE (3 streams) | TBD |
| + Cross-camera re-ID | TBD |
| + Zone analytics + events | TBD |
| Full pipeline | TBD |