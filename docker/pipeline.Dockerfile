# Sentinel Orin pipeline image
# Base: DeepStream 7.1 samples (samples-multiarch works on Jetson driver 540.x;
# triton-multiarch requires 560.28+ and won't start on Orin Nano).
#
# Prerequisites before docker build:
#   1. Compile the custom parser on the Jetson host:
#        cd deepstream/custom_parser && make
#      This produces libnvdsinfer_custom_yolo26.so which is copied in below.
#      See docs/DEPLOYMENT_JETSON.md for the full build procedure.
#   2. docker build -f docker/pipeline.Dockerfile -t sentinel-pipeline .
FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch

WORKDIR /opt/sentinel

# GStreamer Python bindings + pyds
# pyds is not bundled in the samples image and must be installed manually.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-gi \
        python3-gst-1.0 \
        python-gi-dev \
        libpython3.10 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl \
    && pip3 install --no-cache-dir pyds-1.2.0-cp310-cp310-linux_aarch64.whl \
    && rm pyds-1.2.0-cp310-cp310-linux_aarch64.whl

# Python deps
RUN pip3 install --no-cache-dir \
    scipy==1.13.1 \
    numpy==1.26.4 \
    paho-mqtt==2.1.0 \
    prometheus-client==0.21.0

# Pre-compiled parser .so (built on host via deepstream/custom_parser/make)
COPY deepstream/ ./deepstream/
COPY services/pipeline/ ./pipeline/
COPY services/evidence/ ./evidence/

CMD ["python3", "pipeline/sentinel_pipeline.py"]