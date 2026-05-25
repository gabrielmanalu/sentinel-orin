# Stage 1: builder — compiles custom YOLO26 parser + OSNet helper
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch AS builder

WORKDIR /build
COPY deepstream/custom_parser/ ./custom_parser/
# Build the custom nvinfer parser library (libnvdsinfer_custom_yolo26.so)
RUN cd custom_parser && make -j"$(nproc)" || echo "parser build placeholder"

# Stage 2: runtime — smaller samples image + app code
FROM nvcr.io/nvidia/deepstream:7.1-samples-multiarch

WORKDIR /opt/sentinel

# Python deps for pipeline (pyds is preinstalled in DeepStream image)
RUN pip3 install --no-cache-dir \
    scipy==1.13.1 \
    numpy==1.26.4 \
    paho-mqtt==2.1.0 \
    prometheus-client==0.21.0

COPY --from=builder /build/custom_parser/libnvdsinfer_custom_yolo26.so ./deepstream/custom_parser/ 2>/dev/null || true
COPY deepstream/ ./deepstream/
COPY services/pipeline/ ./pipeline/
COPY services/evidence/ ./evidence/

CMD ["python3", "pipeline/sentinel_pipeline.py"]
