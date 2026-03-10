#pragma once

#include <cstdint>

#include "onnx_runner.h"

namespace onnx_decode_utils {

onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const float* data, int64_t time_dim, int64_t class_dim);
onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const double* data, int64_t time_dim, int64_t class_dim);

} // namespace onnx_decode_utils
