/*
 * Mo ta file: Trien khai cac ham xu ly anh tien de dung chung trong pipeline.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/utils/image_preprocess.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <string>

namespace image_preprocess {

cv::Mat PreprocessMatRgbU8Hwc(const cv::Mat& bgr, int target_w, int target_h) {
    if (bgr.empty()) {
        throw std::runtime_error("Khong doc duoc anh (mat rong)");
    }
    if (bgr.channels() != 3) {
        throw std::runtime_error("Anh dau vao can 3 kenh");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(target_w, target_h), 0.0, 0.0, cv::INTER_LINEAR);

    if (resized.type() != CV_8UC3) {
        resized.convertTo(resized, CV_8UC3);
    }
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }
    return resized;
}

cv::Mat ReadAndPreprocessImageRgbU8Hwc(const std::filesystem::path& image_path, int target_w, int target_h) {
    cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Không đọc được ảnh: " + image_path.string());
    }
	return PreprocessMatRgbU8Hwc(bgr, target_w, target_h);
}

} // namespace image_preprocess
