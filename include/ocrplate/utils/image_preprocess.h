#pragma once

#include <filesystem>

#include <opencv2/core.hpp>

namespace image_preprocess {

// Đọc ảnh từ disk (BGR), đổi sang RGB, resize về (W,H), trả về cv::Mat dạng HWC, RGB, uint8.
cv::Mat ReadAndPreprocessImageRgbU8Hwc(const std::filesystem::path& image_path, int target_w, int target_h);

// Tương tự ReadAndPreprocessImageRgbU8Hwc nhưng nhận ảnh BGR đã có sẵn.
// Trả về RGB uint8 HWC (continuous).
cv::Mat PreprocessMatRgbU8Hwc(const cv::Mat& bgr, int target_w, int target_h);

} // namespace image_preprocess
