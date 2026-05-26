# Builds libnvdsinfer_custom_yolo26.so for DeepStream nvinfer.
# FP16 end-to-end NMS-free parser for [1, 300, 6] output format.
#
# Tested on: DeepStream 7.1, TensorRT 10.3, CUDA 12.6, JetPack 6.2.x
#
# NOTE: libnvparsers was removed in TensorRT 10.x (merged into libnvinfer).
#       Do NOT add -lnvparsers — it no longer exists.
#
# NOTE: TensorRT headers are not bundled in the samples-multiarch container.
#       Copy them from the host before building (see docs/DEPLOYMENT_JETSON.md):
#         docker cp /usr/include/aarch64-linux-gnu/. <cid>:/usr/include/
#         docker cp /usr/local/cuda-12.6/include/.  <cid>:/usr/local/cuda-12.6/include/
#         docker cp /usr/lib/aarch64-linux-gnu/libnvinfer.so.10 <cid>:/usr/lib/aarch64-linux-gnu/
#         docker exec <cid> ln -s /usr/lib/aarch64-linux-gnu/libnvinfer.so.10 \
#                                  /usr/lib/aarch64-linux-gnu/libnvinfer.so
#
# Build:
#   cd deepstream/custom_parser && make
# Output:
#   libnvdsinfer_custom_yolo26.so

CUDA_VER     ?= 12.6
NVDS_VERSION ?= 7.1
CC           := g++

DS_ROOT  := /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)
DS_INC   := $(DS_ROOT)/sources/includes
CUDA_INC := /usr/local/cuda-$(CUDA_VER)/include
TRT_INC  := /usr/include
CUDA_LIB := /usr/local/cuda-$(CUDA_VER)/lib64

CFLAGS := -Wall -std=c++14 -shared -fPIC
CFLAGS += -I$(DS_INC) -I$(TRT_INC) -I$(CUDA_INC)

# -lnvparsers removed: merged into libnvinfer in TensorRT 10.x
LFLAGS := -lnvinfer -L$(CUDA_LIB) -lcudart

TARGET := libnvdsinfer_custom_yolo26.so
SRC    := nvdsinfer_custom_yolo26.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(CFLAGS) $(LFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean