/**
 * @file ocr_batch.h
 * @brief Khai bao API OCR theo lo (batch) cho danh sach anh bịển số da preprocess.
 *
 * File nay định nghia dữ liệu đầu ra OCR va ham infer batch OCR.
 * Mục tiêu la tach phan "gọi model" ra khoi pipeline để code để test va để tai su dùng.
 */
#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace ocr_batch {

/**
 * @brief Kết quả OCR cua mot bịển số.
 */
struct OcrText {
	/**
	 * @brief Chuoi text sau khi decode CTC va hau xu ly.
	 *
	 * Có thể rong nếu model không tu tin hoặc anh đầu vào không hop le.
	 */
	std::string text;
	/**
	 * @brief Độ tin cậy trung binh cua chuoi OCR.
	 *
	 * Cach tinh thường la trung binh confidence cua cac timestep non-blank.
	 * Mien gia tri kỳ vọng: [0, 1].
	 */
	float conf_avg = 0.0f;
};

/**
 * @brief Chạy OCR theo lo (batch) cho danh sach anh bịển số.
 *
 * Ham nay infer model OCR ONNX cho nhiều anh cung luc.
 * Thứ tự kết quả đầu ra khớp 1-1 voi thứ tự anh đầu vào.
 *
 * @param session Session ONNX Runtime cua model OCR.
 * @param rgb_u8_hwc Danh sach anh RGB uint8 layout HWC, moi anh phai continuous.
 * @param alphabet Bang ký tự CTC để map index thanh ký tự.
 * @return std::vector<OcrText> Danh sach kết quả OCR theo tung anh đầu vào.
 *
 * @note Input model hiện tại kỳ vọng tensor NHWC uint8 co shape (N,64,128,3).
 */
std::vector<OcrText> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& rgb_u8_hwc,
	const std::string& alphabet);

} // namespace ocr_batch
