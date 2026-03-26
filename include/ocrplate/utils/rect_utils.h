/*
 * Mô tả file: Tiện ích chuẩn hóa bounding box float về cv::Rect hợp lệ trong ảnh.
 */
#pragma once

#include <opencv2/core.hpp>

namespace rect_utils {

// Chuẩn hóa bbox theo miền ảnh [0, w-1] x [0, h-1], đảm bảo width/height không âm.
cv::Rect ToRectClamped(float x1, float y1, float x2, float y2, int w, int h);

} // namespace rect_utils
