# Builds libnvdsinfer_custom_yolo26.so for DeepStream nvinfer.
# FP16 end-to-end NMS-free parser for [1, 300, 6] output format.
#
# Build on the Jetson inside the DeepStream builder container:
#   cd deepstream/custom_parser && make
#
# Output: libnvdsinfer_custom_yolo26.so
# Copied to /opt/sentinel/deepstream/custom_parser/ by the Dockerfile.

CUDA_VER    ?= 12.6
NVDS_VERSION?= 7.1
CC          := g++

DS_ROOT := /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)
DS_INC  := $(DS_ROOT)/sources/includes
CUDA_INC:= /usr/local/cuda-$(CUDA_VER)/include
CUDA_LIB:= /usr/local/cuda-$(CUDA_VER)/lib64

CFLAGS := -Wall -std=c++14 -shared -fPIC
CFLAGS += -I$(DS_INC) -I$(CUDA_INC)
LFLAGS := -lnvinfer -lnvparsers -L$(CUDA_LIB) -lcudart

TARGET := libnvdsinfer_custom_yolo26.so
SRC    := nvdsinfer_custom_yolo26.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(CFLAGS) $(LFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean