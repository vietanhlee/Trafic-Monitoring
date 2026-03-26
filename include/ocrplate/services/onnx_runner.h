/**
 * @file onnx_runner.h
 * @brief Khai báo helper infer ONNX OCR và decode top-1 theo timestep.
 *
 * Module này giúp gọi model OCR theo đầu vào NHWC uint8 và trả về chuỗi index
 * (kèm confidence nếu cần) để phần post-process bên ngoài xử lý tiếp.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace onnx_runner {

/**
 * @brief Kết quả argmax + confidence theo từng timestep.
 */
struct ArgMaxWithConfResult {
	/**
	 * @brief Dãy index top-1 theo từng timestep.
	 *
	 * Thường được đưa vào CTC decode để tạo chuỗi text.
	 */
	std::vector<int64_t> indices;
	/**
	 * @brief Confidence top-1 theo từng timestep.
	 *
	 * Miền giá trị kỳ vọng: [0, 1].
	 */
	std::vector<float> conf;
};

/**
 * @brief Chạy model OCR và trả về chuỗi argmax index theo timestep.
 *
 * @param env Ort::Env đã khởi tạo từ ứng dụng.
 * @param model_path Đường dẫn file model .onnx.
 * @param nhwc_u8 Con trỏ dữ liệu ảnh layout NHWC uint8.
 * @param h Chiều cao ảnh input.
 * @param w Chiều rộng ảnh input.
 * @param c Số kênh ảnh input (thường là 3).
 * @return std::vector<int64_t> Danh sách index top-1 theo từng timestep.
 */
std::vector<int64_t> RunModelGetArgMax(
 	Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c);

/**
 * @brief Giống RunModelGetArgMax nhưng trả thêm confidence từng timestep.
 *
 * @param env Ort::Env đã khởi tạo từ ứng dụng.
 * @param model_path Đường dẫn file model .onnx.
 * @param nhwc_u8 Con trỏ dữ liệu ảnh layout NHWC uint8.
 * @param h Chiều cao ảnh input.
 * @param w Chiều rộng ảnh input.
 * @param c Số kênh ảnh input (thường là 3).
 * @return ArgMaxWithConfResult Gồm index top-1 và confidence theo timestep.
 */
ArgMaxWithConfResult RunModelGetArgMaxAndConf(
	Ort::Env& env,
	const std::string& model_path,
	const uint8_t* nhwc_u8,
	int h,
	int w,
	int c);

} // namespace onnx_runner
