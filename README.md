# Sentinel Orin

Multi-camera edge AI analytics appliance running on a single Jetson Orin Nano Super 8GB.

Three synchronized camera views are processed in real time through YOLO26n detection,
NvDCF tracking, and OSNet cross-camera re-identification — assigning a single global ID
to each person as they move between overlapping cameras. Zone-violation events are
detected, evidence is captured asynchronously, metrics stream to Grafana, and events are
republished over MQTT and ROS 2 to a live RViz2 map.

Validated on the WILDTRACK seven-camera HD pedestrian benchmark, with cross-camera
tracking accuracy measured against its ground-truth identities. The architecture is
general multi-camera edge analytics — applicable to industrial safety, retail, and
smart-city monitoring.

> **Status:** Active development, following
> [EdgeDrive Perception](https://github.com/gabrielmanalu/EdgeDrive-Perception).

---

## Why this project

Most multi-camera projects stop at per-camera tracking, where the same person gets a
different ID in each view. Sentinel projects detections onto a shared ground plane using
WILDTRACK's calibration, matches appearance embeddings across cameras, and reports
measured MOTA against ground truth — a verifiable accuracy number, not just a demo.

Cross-camera tracking is implemented manually (OSNet re-ID + ground-plane geometry)
rather than via DeepStream's MV3DT, because MV3DT requires DeepStream 8.0+ / JetPack 7,
which only supports Jetson Thor. On Orin Nano (DeepStream 7.1), hand-rolled re-ID is both
necessary and a more demonstrable engineering contribution.

---

## Architecture

```
3x camera streams  ->  nvstreammux (batch=3)
                       -> YOLO26n PGIE (FP16, person only)
                       -> NvDCF tracker (per-camera IDs)
                       -> OSNet SGIE (512-dim embeddings)
                       -> cross-camera re-ID -> global IDs
                       -> zone analytics -> events
                       -> nvdsanalytics -> fakesink

events -> SQLite + FastAPI | async evidence | MQTT -> ROS 2 -> RViz2
metrics -> Prometheus -> Grafana
```

All services run as Docker Compose on the Jetson.

---

## Hardware

| Component | Spec | Cost |
|---|---|---|
| Edge compute | Jetson Orin Nano Super 8GB Developer Kit | ~$250 |
| **Total** | | **~$250** |

Software: JetPack 6.2.x · DeepStream 7.1 · TensorRT 10.x · ROS 2 Humble

---

## Models

| Role | Model | Notes |
|---|---|---|
| Detector (PGIE) | YOLO26n, COCO-pretrained | Person class only; NMS-free head |
| Re-ID (SGIE) | OSNet x0.25, Market-1501 | 512-dim appearance embeddings, FP16 |

YOLO26n uses the COCO-pretrained person class rather than a fine-tuned model: WILDTRACK
is pedestrian footage, COCO's person class already covers the domain.

---

## Quick start

> Requires a Jetson Orin Nano Super flashed with JetPack 6.2.x.

```bash
git clone https://github.com/gabrielmanalu/sentinel-orin
cd sentinel-orin
docker compose up --build
```

Access points:

```
http://<jetson-ip>:8080    FastAPI events
http://<jetson-ip>:3000    Grafana dashboards
http://<jetson-ip>:9090    Prometheus
```

---

## License

AGPL-3.0, for compatibility with Ultralytics YOLO. OSNet (deep-person-reid) retains its
MIT license. See [`LICENSE`](LICENSE).