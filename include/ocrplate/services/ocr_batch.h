/**
 * @file ocr_batch.h
 * @brief Khai báo API OCR theo lô (batch) cho danh sách ảnh biển số đã preprocess.
 *
 * File này định nghĩa dữ liệu đầu ra OCR và hàm infer batch OCR.
 * Mục tiêu là tách phần "gọi model" ra khỏi pipeline để code dễ test và tái sử dụng.
 */
#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace ocr_batch {

/**
 * @brief Kết quả OCR của một biển số.
 */
struct OcrText {
	/**
	 * @brief Chuỗi text sau khi decode CTC và hậu xử lý.
	 *
	 * Có thể rỗng nếu model không tự tin hoặc ảnh đầu vào không hợp lệ.
	 */
	std::string text;
	/**
	 * @brief Độ tin cậy trung bình của chuỗi OCR.
	 *
	 * Cách tính thường là trung bình confidence của các timestep non-blank.
	 * Miền giá trị kỳ vọng: [0, 1].
	 */
	float conf_avg = 0.0f;
};

/**
 * @brief Chạy OCR theo lô (batch) cho danh sách ảnh biển số.
 *
 * Hàm này infer model OCR ONNX cho nhiều ảnh cùng lúc.
 * Thứ tự kết quả đầu ra khớp 1-1 với thứ tự ảnh đầu vào.
 *
 * @param session Session ONNX Runtime của model OCR.
 * @param rgb_u8_hwc Danh sách ảnh RGB uint8 layout HWC, mỗi ảnh phải continuous.
 * @param alphabet Bảng ký tự CTC để map index thành ký tự.
 * @return std::vector<OcrText> Danh sách kết quả OCR theo từng ảnh đầu vào.
 *
 * @note Input model hiện tại kỳ vọng tensor NHWC uint8 có shape (N,64,128,3).
 */
std::vector<OcrText> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& rgb_u8_hwc,
	const std::string& alphabet);

} // namespace ocr_batch
