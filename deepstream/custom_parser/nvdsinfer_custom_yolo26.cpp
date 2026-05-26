/*
 * nvdsinfer_custom_yolo26.cpp — DeepStream nvinfer custom parser for YOLO26n
 * ==========================================================================
 * Handles the FP16 end-to-end NMS-free export format: [1, 300, 6].
 *
 * YOLO26n architecture advantage over YOLOv8: NMS-free one-to-one head
 * eliminates duplicate-box suppression at the architecture level. The ONNX
 * graph includes a TopK op that emits the top-300 decoded detections
 * directly; no post-hoc NMS is needed or applied here.
 *
 * Output tensor layout: [1, 300, 6], row-major over detections.
 *   Each row: [x1, y1, x2, y2, score, class_id]
 *   - Coordinates are in NETWORK INPUT SPACE (640x640), xyxy format.
 *   - Scores are pre-activated (sigmoid), already in [0, 1].
 *   - class_id is a float — cast to int for comparison.
 *
 * NMS strategy: NONE. In-graph TopK already handled it. Set cluster-mode=4
 * in the nvinfer config to tell DeepStream not to cluster our output.
 *
 * Model: COCO-pretrained YOLO26n, filtered to PERSON CLASS (class_id == 0).
 *
 * Precision: FP16 primary engine (network-mode=2 in nvinfer config).
 * For INT8 benchmarking, a separate raw-head parser will be written
 * for the [1, 84, 8400] format, since end-to-end export is broken on
 * JetPack 6.2.x for INT8 (TensorRT 10.x Jetson limitation).
 *
 * Hard-won lessons from EdgeDrive carried forward:
 *   1. Coordinates are already xyxy in 640-space — no cxcywh conversion needed.
 *   2. Scores are pre-sigmoid — do NOT apply sigmoid again.
 *   3. class_id is a float cast from int — use roundf() before comparing.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static constexpr int MAX_DETECTIONS = 300;  // TopK limit baked into the graph
static constexpr int FIELDS_PER_DET = 6;    // [x1, y1, x2, y2, score, class]
static constexpr int NETWORK_DIM    = 640;
static constexpr int PERSON_CLASS   = 0;    // COCO class 0

extern "C" bool NvDsInferParseCustomYolo26(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYolo26(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        return false;
    }

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* data = static_cast<const float*>(layer.buffer);
    if (data == nullptr) {
        return false;
    }

    // Confidence threshold for person class.
    const float thresh = detectionParams.perClassPreclusterThreshold.empty()
        ? 0.30f
        : detectionParams.perClassPreclusterThreshold[0];

    objectList.reserve(64);

    // Iterate over the 300 candidate detections.
    // Memory layout: row-major [det_0_x1, det_0_y1, ..., det_299_class_id]
    for (int i = 0; i < MAX_DETECTIONS; i++) {
        const float* det = data + i * FIELDS_PER_DET;

        const float score    = det[4];
        const int   class_id = static_cast<int>(roundf(det[5]));

        // Filter: person only, above threshold.
        if (class_id != PERSON_CLASS || score < thresh) {
            continue;
        }

        // Coordinates already xyxy in 640-space. Clamp to network bounds.
        float x1 = std::max(0.0f, std::min(det[0], static_cast<float>(NETWORK_DIM)));
        float y1 = std::max(0.0f, std::min(det[1], static_cast<float>(NETWORK_DIM)));
        float x2 = std::max(0.0f, std::min(det[2], static_cast<float>(NETWORK_DIM)));
        float y2 = std::max(0.0f, std::min(det[3], static_cast<float>(NETWORK_DIM)));

        float bw = x2 - x1;
        float bh = y2 - y1;
        if (bw <= 0.0f || bh <= 0.0f) {
            continue;
        }

        NvDsInferParseObjectInfo obj;
        obj.classId             = PERSON_CLASS;
        obj.left                = x1;
        obj.top                 = y1;
        obj.width               = bw;
        obj.height              = bh;
        obj.detectionConfidence = score;
        objectList.push_back(obj);
    }

    return true;
}

// Compile-time check that our function matches DeepStream's expected signature.
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26);