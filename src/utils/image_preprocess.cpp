/*
 * Mô tả file: Triển khai các hàm xử lý ảnh tiền xử lý dùng chung trong pipeline.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#include "ocrplate/utils/image_preprocess.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <string>

namespace image_preprocess {

cv::Mat PreprocessMatRgbU8Hwc(const cv::Mat& bgr, int target_w, int target_h) {
    if (bgr.empty()) {
        throw std::runtime_error("Không đọc được anh (mất rong)");
    }
    if (bgr.channels() != 3) {
        throw std::runtime_error("Anh đầu vào cần 3 kenh");
    }

    cv::Mat rgb;
    // Đổi kênh màu về RGB để phù hợp các model OCR/brand trong pipeline.
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    // Đưa về kích thước model yêu cầu.
    cv::resize(rgb, resized, cv::Size(target_w, target_h), 0.0, 0.0, cv::INTER_LINEAR);

    if (resized.type() != CV_8UC3) {
        // Chốt type uint8 3 kênh để tạo tensor ONNX đơn giản và ổn định.
        resized.convertTo(resized, CV_8UC3);
    }
    if (!resized.isContinuous()) {
        // Clone để đảm bảo bộ nhớ liên tục cho memcpy vào tensor.
        resized = resized.clone();
    }
    return resized;
}

cv::Mat ReadAndPreprocessImageRgbU8Hwc(const std::filesystem::path& image_path, int target_w, int target_h) {
    // Đọc ảnh màu từ đĩa bằng OpenCV (mặc định là BGR).
    cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Không đọc được ảnh: " + image_path.string());
    }
	return PreprocessMatRgbU8Hwc(bgr, target_w, target_h);
}

} // namespace image_preprocess
