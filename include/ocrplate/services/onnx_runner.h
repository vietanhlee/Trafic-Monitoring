#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace onnx_runner {

struct ArgMaxWithConfResult {
    std::vector<int64_t> indices;
	// Độ tin cậy theo từng timestep của class argmax: softmax(top1) trong [0,1]
    std::vector<float> conf;
};

// Chạy model ONNX với input ảnh uint8 NHWC (1,H,W,3).
// Trả về chuỗi indices (argmax theo classes) theo chiều thời gian T.
std::vector<int64_t> RunModelGetArgMax(
    Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c);

// Giống RunModelGetArgMax nhưng trả thêm confidence theo timestep (tính từ logits nếu cần).
ArgMaxWithConfResult RunModelGetArgMaxAndConf(
	Ort::Env& env,
	const std::string& model_path,
	const uint8_t* nhwc_u8,
	int h,
	int w,
	int c);

} // namespace onnx_runner
