#pragma once

#include <cstdint>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocrplate/core/detection.h"

namespace yolo_detector {

struct LetterboxInfo {
	int orig_w = 0;
	int orig_h = 0;
	int in_w = 0;
	int in_h = 0;
	float scale = 1.0f;
	int pad_x = 0;
	int pad_y = 0;
};

struct InputSpec {
	ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	bool nchw = true; // true: NCHW, false: NHWC
	int64_t n = -1;   // batch; -1 nghĩa là dynamic/không rõ
	int64_t c = 3;
	int64_t h = 640;
	int64_t w = 640;
};

// Chạy YOLO để detect.
// Hỗ trợ standard YOLO output: (N, 4+num_classes, num_anchors) → transpose + NMS
std::vector<std::vector<Detection>> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold,
	float nms_iou_threshold = 0.35f);

std::vector<Detection> RunSingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	float conf_threshold,
	float nms_iou_threshold = 0.35f);

} // namespace yolo_detector
