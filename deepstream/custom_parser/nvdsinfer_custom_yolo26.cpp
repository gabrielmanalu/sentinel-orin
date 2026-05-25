/*
 * Custom DeepStream nvinfer parser for YOLO26n output.
 *
 * ported from EdgeDrive's yolo26_decoder.cpp. Wraps the existing
 * 8400-anchor decode + cxcywh->xyxy + NMS logic in NVIDIA's
 * NvDsInferParseCustomFunc signature so DeepStream's nvinfer plugin can
 * call it. Person class (id 0) only.
 *
 * extern "C" bool NvDsInferParseCustomYolo26(
 *     std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
 *     NvDsInferNetworkInfo const& networkInfo,
 *     NvDsInferParseDetectionParams const& detectionParams,
 *     std::vector<NvDsInferParseObjectInfo>& objectList);
 */
