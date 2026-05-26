# Run A — Detection-only baseline

YOLO26n FP16, no tracker, no re-ID. Establishes the detection pipeline
throughput ceiling before adding downstream components.

Hardware: Jetson Orin Nano Super 8GB
Power mode: 25W (sustained, stable — MAXN_SUPER throttles under multi-stream load)
Clocks: jetson_clocks locked
Engine: yolo26n_fp16_b3_640 (FP16, batch=3, [1,300,6] end-to-end format)
Parser: libnvdsinfer_custom_yolo26.so

---

## Single stream — cam01, 1080p60

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

## 3 streams — cam01 + cam03 + cam05, 1080p60

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

## Observations

**Batch efficiency:** Adding 2 streams to a batch=3 engine costs only ~50 FPS
aggregate (184 → 234). The GPU was underutilized on a single stream — the
batch=3 engine amortizes inference cost across all 3 frames per forward pass,
making 3-stream significantly more GPU-efficient than 3 separate single-stream
runs would be.

**25W vs MAXN_SUPER:** MAXN_SUPER triggered over-current throttling under
3-stream sustained load. 25W is the correct benchmark mode for this hardware —
stable, reproducible, and realistic for a deployed appliance.

**Headroom:** 234.1 FPS detection-only gives substantial budget for downstream
components (tracker, OSNet SGIE, re-ID, zone analytics). Target aggregate FPS
with full pipeline is 45+. Current headroom is 5.2× above that target.

---

## Cumulative pipeline FPS (updated as components are added)

| Pipeline stage | Power mode | Aggregate FPS |
|---|---|---|
| Detection only (1 stream) | 25W | 184.3 |
| Detection only (3 streams) | 25W | 234.1 |
| + NvDCF tracker | 25W | TBD |
| + OSNet SGIE | 25W | TBD |
| + Cross-camera re-ID | 25W | TBD |
| + Zone analytics + events | 25W | TBD |
| Full pipeline | 25W | TBD |