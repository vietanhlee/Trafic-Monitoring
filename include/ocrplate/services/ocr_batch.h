#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace ocr_batch {

struct OcrText {
	std::string text;
	float conf_avg = 0.0f; // độ tin cậy trung bình cho chuỗi OCR (trung bình các timestep không phải blank)
};

// Input: batch ảnh RGB uint8 dạng HWC (H,W,3), yêu cầu continuous.
// OCR model kỳ vọng input: (N,64,128,3) uint8 NHWC (giống như repo hiện tại).
std::vector<OcrText> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& rgb_u8_hwc,
	const std::string& alphabet);

} // namespace ocr_batch
