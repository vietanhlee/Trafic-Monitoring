#pragma once

#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace brand_classifier {

struct BrandResult {
	int class_id = -1;
	float conf = 0.0f;
};

// Chạy phân loại hãng xe cho 1 ảnh BGR crop phương tiện.
// Model kỳ vọng input float32 NCHW với kích thước (3,224,224) hoặc (1,3,224,224).
BrandResult ClassifySingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	int input_h,
	int input_w);

// Chạy phân loại hãng xe theo batch ảnh BGR crop phương tiện.
// Model kỳ vọng input float32 NCHW: (N,3,H,W).
std::vector<BrandResult> ClassifyBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	int input_h,
	int input_w);

} // namespace brand_classifier
