# Jetson Deployment Guide

Target hardware: Jetson Orin Nano Super 8GB Developer Kit
JetPack: 6.2.x (Jetson Linux 36.4.x / 36.5.x)

---

## 1. Flash JetPack 6.2.x

Use NVIDIA SDK Manager on a host Ubuntu 22.04 machine.

1. Download SDK Manager: https://developer.nvidia.com/sdk-manager
2. Put the Jetson into recovery mode:
   - Power off
   - Hold FORCE RECOVERY button (or jumper pins 9-10 on J14 header)
   - Connect USB-C from Jetson to host
   - Power on
   - Run `lsusb` on host — confirm "NVIDIA Corp" device appears
3. Launch SDK Manager, select:
   - Target: **Jetson Orin Nano 8GB Developer Kit** (no separate "Super" entry)
   - JetPack version: **6.2.2**
   - Storage target: **NVMe** (the P34A60 SSD)
4. Let SDK Manager flash and install (~30-60 min)

---

## 2. First boot configuration

### Activate Super performance mode

```bash
sudo nvpmodel -m 2
sudo nvpmodel -q
# Expected: NV Power Mode: MAXN_SUPER
```

### Lock clocks for benchmarking (optional, disable during normal use)

```bash
sudo jetson_clocks
```

Note: `jetson_clocks` locks all clocks at maximum. Use only during benchmark
runs for consistent numbers. Skip during normal development — let DVFS manage
clocks dynamically.

### Verify CUDA and driver

```bash
nvidia-smi
# Expected: Driver 540.x, CUDA 12.6, GPU: Orin (nvgpu)
```

---

## 3. Install Docker and NVIDIA Container Runtime

Docker and the NVIDIA runtime ship with JetPack 6.2.x. Verify:

```bash
docker --version
# Docker version 29.x or higher

docker compose version
# Docker Compose version v2.x or higher

docker info | grep -i runtime
# Runtimes: io.containerd.runc.v2 nvidia runc
# Default Runtime: nvidia   <-- must be nvidia
```

If Default Runtime is not `nvidia`, add to `/etc/docker/daemon.json`:

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

Then restart: `sudo systemctl restart docker`

---

## 4. Create data directory structure

```bash
sudo mkdir -p /data/sentinel/{datasets/wildtrack,videos/wildtrack_v1,models/yolo26n,models/osnet,engines,events,evidence/snapshots,evidence/clips,logs,benchmarks/runA,benchmarks/runB,benchmarks/runC,benchmarks/runD,benchmarks/runE}
sudo chown -R $USER:$USER /data/sentinel
```

---

## 5. Pull DeepStream 7.1 image

```bash
docker pull nvcr.io/nvidia/deepstream:7.1-samples-multiarch
```

**Note:** DeepStream 8.0+ requires JetPack 7 / Jetson Thor. DeepStream 7.1
is the ceiling for Jetson Orin Nano on JetPack 6.2.x.

**Note:** The `triton-multiarch` variant requires driver 560.28+. Jetson ships
driver 540.x. Always use `samples-multiarch` for Orin Nano.

---

## 6. Build the custom YOLO26 parser

The `samples-multiarch` container does not include TensorRT dev headers or
CUDA headers. These must be copied from the Jetson host before building.

### Step 1 — Start the container

```bash
cd ~/sentinel-orin
docker run -it --rm \
  --runtime=nvidia \
  -v $(pwd):/opt/sentinel \
  nvcr.io/nvidia/deepstream:7.1-samples-multiarch \
  bash
```

### Step 2 — On the host (second terminal), copy headers and libs

Get the container ID:

```bash
docker ps
# Note the container ID, e.g. 7f06ee97cd76
```

Copy everything the compiler needs:

```bash
# TensorRT + system headers (includes NvInfer.h and all dependencies)
docker cp /usr/include/aarch64-linux-gnu/. <container_id>:/usr/include/

# CUDA runtime headers
docker cp /usr/local/cuda-12.6/include/. <container_id>:/usr/local/cuda-12.6/include/

# TensorRT shared library
docker cp /usr/lib/aarch64-linux-gnu/libnvinfer.so.10 \
  <container_id>:/usr/lib/aarch64-linux-gnu/

# Unversioned symlink — linker needs libnvinfer.so, not libnvinfer.so.10
docker exec <container_id> ln -s \
  /usr/lib/aarch64-linux-gnu/libnvinfer.so.10 \
  /usr/lib/aarch64-linux-gnu/libnvinfer.so
```

**Note:** `-lnvparsers` is NOT used. It was removed in TensorRT 10.x and
merged into `libnvinfer`. The Makefile does not reference it.

### Step 3 — Inside the container, build

```bash
cd /opt/sentinel/deepstream/custom_parser
make
```

Expected output:

```
g++ -o libnvdsinfer_custom_yolo26.so nvdsinfer_custom_yolo26.cpp \
  -Wall -std=c++14 -shared -fPIC ... (no errors)
```

### Step 4 — Verify

```bash
ls -lh libnvdsinfer_custom_yolo26.so
# -rwxr-xr-x ... 44K ... libnvdsinfer_custom_yolo26.so
```

The `.so` is compiled into `deepstream/custom_parser/` inside the mounted
repo — immediately available on the host. It is gitignored (binary,
hardware-specific). Rebuild any time by repeating Steps 1-4.

---

## 7. Export YOLO26n to ONNX

Run on any machine with Ultralytics installed:

```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640 dynamic=False simplify=True opset=17
cp yolo26n.onnx /data/sentinel/models/yolo26n/
```

This exports the FP16-compatible end-to-end NMS-free format `[1, 300, 6]`.
The ONNX itself is FP32 — TensorRT converts to FP16 when building the engine
on first pipeline run, controlled by `network-mode=2` in the nvinfer config.

---

## 8. First pipeline run — TensorRT engine build

On first run, DeepStream calls TensorRT to build and cache the FP16 engine:

```bash
docker compose up pipeline
```

Engine build takes **5-9 minutes** on Orin Nano. This is normal — not a hang.
Engine is cached at:

```
/data/sentinel/engines/yolo26n_fp16_b3_640.engine
```

All subsequent runs load the cached engine (~10 second cold start).

---

## 9. Verify GPU access inside Sentinel pipeline container

```bash
docker run --rm --runtime=nvidia --gpus all \
  nvcr.io/nvidia/deepstream:7.1-samples-multiarch \
  nvidia-smi
# Expected: Driver 540.x, CUDA 12.6, GPU: Orin (nvgpu)
```

---

## Known issues and workarounds

| Issue | Cause | Fix |
|---|---|---|
| `triton-multiarch` won't start | Requires driver 560.28+, Jetson has 540.x | Use `samples-multiarch` always |
| `libnvinfer-dev` not installable in container | L4T repo unreachable from container network | Copy headers from host (Step 6) |
| `libnvparsers` not found | Removed in TensorRT 10.x | Already removed from Makefile — do not add back |
| Clocks at 300MHz idle | Normal DVFS behavior | Run `sudo jetson_clocks` for benchmarks only |
| TensorRT engine build takes 5-9 min | First-run calibration | Normal — cached on subsequent runs |
| INT8 end-to-end export broken | JetPack 6.2.x / TensorRT 10.x Jetson limitation | FP16 primary engine uses end-to-end export; INT8 benchmarking uses raw head export + separate parser |