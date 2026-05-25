# Builds libnvdsinfer_custom_yolo26.so for DeepStream nvinfer.
# Ported from EdgeDrive yolo26_decoder.cpp

CUDA_VER?=12.6
CC:=g++
NVDS_VERSION:=7.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
DS_INC:=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes

CFLAGS:=-Wall -std=c++14 -shared -fPIC -I$(DS_INC) -I/usr/local/cuda-$(CUDA_VER)/include
LFLAGS:=-lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart

TARGET:=libnvdsinfer_custom_yolo26.so
SRC:=nvdsinfer_custom_yolo26.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(CFLAGS) $(LFLAGS)

clean:
	rm -f $(TARGET)
