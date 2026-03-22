/**
 * @file yolo_preprocess.cpp
 * @brief Triển khai đọc input spec, letterbox và đóng gói tensor input cho YOLO.
 */
#include "ocrplate/services/yolo_detector_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <opencv2/imgproc.hpp>

namespace yolo_detector {
namespace detail {

// ── Input spec (cached per session pointer) ───────────────────────────

/**
 * @brief Đọc input metadata của session YOLO và cache thread-local.
 *
 * @param session Session ONNX Runtime YOLO.
 * @return InputSpec Cấu hình input tensor (type/layout/shape).
 */
InputSpec GetInputSpec(Ort::Session& session) {
	// Cache: tránh gọi lại metadata mỗi lần infer.
	static thread_local std::uintptr_t cached_session_id = 0;
	static thread_local InputSpec cached_spec;
	// session_id là khóa cache theo địa chỉ session hiện tại.
	const auto session_id = reinterpret_cast<std::uintptr_t>(&session);
	if (session_id == cached_session_id) {
		return cached_spec;
	}

	Ort::AllocatorWithDefaultOptions allocator;
	auto type_info = session.GetInputTypeInfo(0);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	const auto elem_type = tensor_info.GetElementType();
	const auto shape = tensor_info.GetShape();
	if (shape.size() != 4) {
		throw std::runtime_error("YOLO input rank không hợp lệ (cần 4), rank=" + std::to_string(shape.size()));
	}

	InputSpec spec;
	// type và n được đọc trực tiếp từ graph ONNX.
	spec.type = elem_type;
	spec.n = shape[0];

	const bool dim1_is_c = (shape[1] == 3);
	const bool dim3_is_c = (shape[3] == 3);
	if (dim1_is_c && !dim3_is_c) {
		// Dạng NCHW: [N, C, H, W]
		spec.nchw = true;
		spec.c = 3;
		spec.h = shape[2];
		spec.w = shape[3];
	} else if (dim3_is_c && !dim1_is_c) {
		// Dạng NHWC: [N, H, W, C]
		spec.nchw = false;
		spec.c = 3;
		spec.h = shape[1];
		spec.w = shape[2];
	} else {
		// Trường hợp mở ho: fallback NCHW để giu tương thích model cu.
		spec.nchw = true;
		spec.c = (shape[1] > 0 ? shape[1] : 3);
		spec.h = shape[2];
		spec.w = shape[3];
	}

	if (spec.c != 3) {
		throw std::runtime_error("YOLO input không hỗ trợ so kenh != 3");
	}
	if (spec.h <= 0 || spec.w <= 0) {
		// Fallback an toàn nếu shape dynamic/chưa xác định.
		spec.h = 640;
		spec.w = 640;
	}
	cached_session_id = session_id;
	cached_spec = spec;
	return spec;
}

// ── Letterbox ─────────────────────────────────────────────────────────

/**
 * @brief Resize + letterbox ảnh BGR về kích thước model và đổi sang RGB.
 *
 * @param bgr Ảnh đầu vào BGR.
 * @param target_w Chiều rộng input model.
 * @param target_h Chiều cao input model.
 * @param info [out] Thong tin scale/pad để map bbox về ảnh gốc.
 * @return cv::Mat Ảnh RGB đã letterbox.
 */
cv::Mat LetterboxToSizeRGB(const cv::Mat& bgr, int target_w, int target_h, LetterboxInfo& info) {
	if (bgr.empty()) {
		throw std::runtime_error("Anh rong");
	}
	info.orig_w = bgr.cols;
	info.orig_h = bgr.rows;
	info.in_w = target_w;
	info.in_h = target_h;

	const float r = std::min(static_cast<float>(target_w) / static_cast<float>(info.orig_w),
						static_cast<float>(target_h) / static_cast<float>(info.orig_h));
	// scale dòng nhất theo canh ngan hon để giu ty le hinh.
	info.scale = r;

	const int new_w = static_cast<int>(std::round(info.orig_w * r));
	const int new_h = static_cast<int>(std::round(info.orig_h * r));
	// pad_x/pad_y la lech cần giua noi dùng that va vien letterbox.
	info.pad_x = (target_w - new_w) / 2;
	info.pad_y = (target_h - new_h) / 2;

	cv::Mat resized;
	cv::resize(bgr, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);

	cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
	// Dien nen 114 theo convention YOLO letterbox.
	resized.copyTo(padded(cv::Rect(info.pad_x, info.pad_y, new_w, new_h)));

	cv::Mat rgb;
	cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
	if (!rgb.isContinuous()) {
		rgb = rgb.clone();
	}
	return rgb;
}

// ── Tensor fill ───────────────────────────────────────────────────────

/**
 * @brief Dong gọi batch anh RGB vào tensor NCHW.
 *
 * @tparam T Kieu tensor dich (float hoặc uint8).
 * @param rgbs_u8 Danh sach anh RGB da letterbox.
 * @param h Chiều cao input tensor.
 * @param w Chiều rộng input tensor.
 * @param out [out] Buffer tensor NCHW.
 * @param scale_01 true nếu cần scale [0,1].
 */
template <typename T>
void FillTensorFromRGB_NCHW(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01) {
	const size_t n = rgbs_u8.size();
	const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
	// out co tong kích thước N * C * H * W.
	out.resize(n * 3ull * plane);
	for (size_t i = 0; i < n; ++i) {
		const size_t base = i * 3ull * plane;
		if constexpr (std::is_same_v<T, float>) {
			// Dùng cv::split cho tốc độ
			cv::Mat float_img;
			rgbs_u8[i].convertTo(float_img, CV_32FC3, scale_01 ? (1.0 / 255.0) : 1.0);
			std::vector<cv::Mat> channels(3);
			cv::split(float_img, channels);
			for (int c = 0; c < 3; ++c) {
				std::memcpy(out.data() + base + c * plane,
					channels[c].ptr<float>(0), plane * sizeof(float));
			}
		} else {
			// uint8: giu nguyen gia tri pixel, split truc tiep theo kenh.
			std::vector<cv::Mat> channels(3);
			cv::split(rgbs_u8[i], channels);
			for (int c = 0; c < 3; ++c) {
				std::memcpy(out.data() + base + c * plane,
					channels[c].ptr<uint8_t>(0), plane * sizeof(uint8_t));
			}
		}
	}
}

/**
 * @brief Dong gọi batch anh RGB vào tensor NHWC.
 *
 * @tparam T Kieu tensor dich (float hoặc uint8).
 * @param rgbs_u8 Danh sach anh RGB da letterbox.
 * @param h Chiều cao input tensor.
 * @param w Chiều rộng input tensor.
 * @param out [out] Buffer tensor NHWC.
 * @param scale_01 true nếu cần scale [0,1].
 */
template <typename T>
void FillTensorFromRGB_NHWC(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01) {
	const size_t n = rgbs_u8.size();
	// out co tong kích thước N * H * W * C.
	out.resize(n * static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull);
	for (size_t i = 0; i < n; ++i) {
		const uint8_t* p = rgbs_u8[i].ptr<uint8_t>(0);
		const size_t base = i * static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull;
		if constexpr (std::is_same_v<T, float>) {
			const float k = scale_01 ? (1.0f / 255.0f) : 1.0f;
			for (size_t j = 0; j < static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull; ++j) {
				out[base + j] = static_cast<float>(p[j]) * k;
			}
		} else {
			for (size_t j = 0; j < static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull; ++j) {
				out[base + j] = static_cast<T>(p[j]);
			}
		}
	}
}

// Explicit template instantiations
template void FillTensorFromRGB_NCHW<float>(const std::vector<cv::Mat>&, int, int, std::vector<float>&, bool);
template void FillTensorFromRGB_NCHW<uint8_t>(const std::vector<cv::Mat>&, int, int, std::vector<uint8_t>&, bool);
template void FillTensorFromRGB_NHWC<float>(const std::vector<cv::Mat>&, int, int, std::vector<float>&, bool);
template void FillTensorFromRGB_NHWC<uint8_t>(const std::vector<cv::Mat>&, int, int, std::vector<uint8_t>&, bool);

} // namespace detail
} // namespace yolo_detector
