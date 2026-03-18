/*
 * Mo ta file: Tien ich giai ma tensor ONNX output sang cau truc de xu ly tiep.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <cstdint>

#include "ocrplate/services/onnx_runner.h"

namespace onnx_decode_utils {

onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const float* data, int64_t time_dim, int64_t class_dim);
onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const double* data, int64_t time_dim, int64_t class_dim);

} // namespace onnx_decode_utils
