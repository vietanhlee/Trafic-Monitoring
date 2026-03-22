/*
 * Mô tả file: Tiện ích đọc/chuyển đổi ảnh đầu vào phục vụ inference.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <filesystem>

#include <opencv2/core.hpp>

namespace image_preprocess {

// Đọc ảnh từ disk, sau đó preprocess về đúng format model OCR.
// Quy trình:
// 1) imread ảnh BGR.
// 2) Convert BGR -> RGB.
// 3) Resize về kích thước target_w x target_h.
// Đầu ra:
// - cv::Mat HWC, uint8, RGB, bộ nhớ liên tục.
cv::Mat ReadAndPreprocessImageRgbU8Hwc(const std::filesystem::path& image_path, int target_w, int target_h);

// Preprocess từ ảnh BGR đã có sẵn trong RAM.
// Dùng khi pipeline đã có crop từ frame/video, không cần đọc lại từ đĩa.
// Đầu ra tương tự hàm ReadAndPreprocessImageRgbU8Hwc:
// - RGB uint8, layout HWC, continuous.
cv::Mat PreprocessMatRgbU8Hwc(const cv::Mat& bgr, int target_w, int target_h);

} // namespace image_preprocess
