#pragma once
// Internal helpers cho yolo_detector – KHÔNG dùng ngoài module này.

#include "yolo_detector.h"

#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace yolo_detector {
namespace detail {

// ── Preprocess ────────────────────────────────────────────────────────

cv::Mat LetterboxToSizeRGB(const cv::Mat& bgr, int target_w, int target_h, LetterboxInfo& info);

template <typename T>
void FillTensorFromRGB_NCHW(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

template <typename T>
void FillTensorFromRGB_NHWC(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

// ── Postprocess ───────────────────────────────────────────────────────

Detection MapBackToOriginal(const Detection& in, const LetterboxInfo& info);

float IoU(const Detection& a, const Detection& b);

std::vector<Detection> ApplyNMS(std::vector<Detection> dets, float iou_threshold);

std::vector<std::vector<Detection>> ParseOutput(
	Ort::Value& out0,
	const std::vector<LetterboxInfo>& infos,
	float conf_threshold,
	float nms_iou_threshold);

// ── Input spec ────────────────────────────────────────────────────────

InputSpec GetInputSpec(Ort::Session& session);

} // namespace detail
} // namespace yolo_detector
