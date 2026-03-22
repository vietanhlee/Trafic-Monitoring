/**
 * @file brand_classifier.h
 * @brief Khai báo API phân loại thương hiệu xe trên crop phương tiện.
 *
 * Module này nhận ảnh crop phương tiện (BGR OpenCV), preprocess về input model,
 * infer classifier ONNX và trả về class + confidence.
 */
#pragma once

#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace brand_classifier {

/**
 * @brief Kết quả phân loại thương hiệu cho một ảnh.
 */
struct BrandResult {
	/**
	 * @brief Chỉ số lớp dự đoán top-1 (argmax).
	 *
	 * Giá trị -1 nghĩa là không có kết quả hợp lệ.
	 */
	int class_id = -1;
	/**
	 * @brief Độ tin cậy của lớp class_id.
	 *
	 * Miền giá trị kỳ vọng: [0, 1].
	 */
	float conf = 0.0f;
};

/**
	 * @brief Phân loại thương hiệu cho một crop phương tiện.
	 *
	 * @param session Session ONNX Runtime của model classifier.
	 * @param bgr_image Ảnh BGR của crop phương tiện (OpenCV).
 * @param input_h Chiều cao input model.
 * @param input_w Chiều rộng input model.
	 * @return BrandResult Kết quả top-1 gồm class_id và confidence.
	 *
	 * @note Hàm tự preprocess ảnh về tensor float32 NCHW.
 */
BrandResult ClassifySingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	int input_h,
	int input_w);


/**
	 * @brief Phân loại thương hiệu theo batch.
	 *
	 * @param session Session ONNX Runtime của model classifier.
	 * @param bgr_images Danh sách ảnh crop BGR.
 * @param input_h Chiều cao input model.
 * @param input_w Chiều rộng input model.
	 * @return std::vector<BrandResult> Danh sách kết quả top-1 theo thứ tự đầu vào.
	 *
	 * @note Input tensor model kỳ vọng float32 NCHW shape (N,3,H,W).
 */
std::vector<BrandResult> ClassifyBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	int input_h,
	int input_w);

} // namespace brand_classifier
