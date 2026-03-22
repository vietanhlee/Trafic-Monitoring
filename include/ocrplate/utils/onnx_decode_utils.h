/*
 * Mô tả file: Tiện ích giải mã tensor ONNX output sang cấu trúc để xử lý tiếp.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <cstdint>

#include "ocrplate/services/onnx_runner.h"

namespace onnx_decode_utils {

// Tính argmax + confidence theo từng timestep từ buffer logits float32.
// Input được coi là ma trận [time_dim, class_dim] theo thứ tự liên tiếp trong bộ nhớ.
// Đầu ra:
// - indices[t] = class có logit lớn nhất tại timestep t.
// - conf[t] = softmax(top-1) tại timestep t.
onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const float* data, int64_t time_dim, int64_t class_dim);

// Bản overload cho logits double.
// Hành vi tương tự bản float để tái sử dụng với model/xuất tensor kiểu f64.
onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const double* data, int64_t time_dim, int64_t class_dim);

} // namespace onnx_decode_utils
