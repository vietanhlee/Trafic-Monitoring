/**
 * @file onnx_runner.h
 * @brief Khai bao helper infer ONNX OCR va decode top-1 theo timestep.
 *
 * Module nay giup gọi model OCR theo đầu vào NHWC uint8 va tra về chuoi index
 * (kem confidence nếu cần) để phan post-process ben ngoai xu ly tiep.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace onnx_runner {

/**
 * @brief Kết quả argmax + confidence theo tung timestep.
 */
struct ArgMaxWithConfResult {
    /**
     * @brief Day index top-1 theo tung timestep.
     *
     * Thường được dua vào CTC decode để tao chuoi text.
     */
    std::vector<int64_t> indices;
    /**
     * @brief Confidence top-1 theo tung timestep.
     *
     * Mien gia tri kỳ vọng: [0, 1].
     */
    std::vector<float> conf;
};

/**
 * @brief Chạy model OCR va tra về chuoi argmax index theo timestep.
 *
 * @param env Ort::Env da khoi tao tu ung dùng.
 * @param model_path Đường dẫn file model .onnx.
 * @param nhwc_u8 Con tro dữ liệu anh layout NHWC uint8.
 * @param h Chiều cao anh input.
 * @param w Chiều rộng anh input.
 * @param c So kenh anh input (thường la 3).
 * @return std::vector<int64_t> Danh sach index top-1 theo tung timestep.
 */
std::vector<int64_t> RunModelGetArgMax(
 	Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c);

/**
 * @brief Giong RunModelGetArgMax nhung tra them confidence tung timestep.
 *
 * @param env Ort::Env da khoi tao tu ung dùng.
 * @param model_path Đường dẫn file model .onnx.
 * @param nhwc_u8 Con tro dữ liệu anh layout NHWC uint8.
 * @param h Chiều cao anh input.
 * @param w Chiều rộng anh input.
 * @param c So kenh anh input (thường la 3).
 * @return ArgMaxWithConfResult Gom index top-1 va confidence theo timestep.
 */
ArgMaxWithConfResult RunModelGetArgMaxAndConf(
	Ort::Env& env,
	const std::string& model_path,
	const uint8_t* nhwc_u8,
	int h,
	int w,
	int c);

} // namespace onnx_runner
