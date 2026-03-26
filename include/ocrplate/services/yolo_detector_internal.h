/**
 * @file yolo_detector_internal.h
 * @brief Khai báo helper nội bộ cho preprocess, parse output và NMS của YOLO.
 */
#pragma once
// Internal helpers cho yolo_detector - KHÔNG dùng ngoài module này.

#include "ocrplate/services/yolo_detector.h"

#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace yolo_detector {
namespace detail {

// ── Preprocess ────────────────────────────────────────────────────────

/**
 * @brief Resize + letterbox ảnh BGR về kích thước model rồi đổi sang RGB.
 *
 * @param bgr Ảnh đầu vào BGR.
 * @param target_w Chiều rộng input model.
 * @param target_h Chiều cao input model.
 * @param info [out] Thông tin scale/pad để map ngược bbox.
 * @return cv::Mat Ảnh RGB sau letterbox.
 */
cv::Mat LetterboxToSizeRGB(const cv::Mat& bgr, int target_w, int target_h, LetterboxInfo& info);

/**
 * @brief Nạp batch ảnh RGB vào tensor layout NCHW.
 *
 * @tparam T Kiểu dữ liệu tensor (float hoặc uint8).
 * @param rgbs_u8 Danh sách ảnh RGB đã letterbox, cùng kích thước h x w.
 * @param h Chiều cao input tensor.
 * @param w Chiều rộng input tensor.
 * @param out [out] Buffer tensor theo thứ tự [N][C][H][W].
 * @param scale_01 true nếu cần scale [0..255] -> [0..1].
 */
template <typename T>
void FillTensorFromRGB_NCHW(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

/**
 * @brief Nạp batch ảnh RGB vào tensor layout NHWC.
 *
 * @tparam T Kiểu dữ liệu tensor (float hoặc uint8).
 * @param rgbs_u8 Danh sách ảnh RGB đã letterbox, cùng kích thước h x w.
 * @param h Chiều cao input tensor.
 * @param w Chiều rộng input tensor.
 * @param out [out] Buffer tensor theo thứ tự [N][H][W][C].
 * @param scale_01 true nếu cần scale [0..255] -> [0..1].
 */
template <typename T>
void FillTensorFromRGB_NHWC(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

// ── Postprocess ───────────────────────────────────────────────────────

/**
 * @brief Map bbox từ hệ tọa độ letterbox về hệ tọa độ ảnh gốc.
 *
 * @param in BBox trong hệ tọa độ input model.
 * @param info Thông tin letterbox (scale/pad/original size).
 * @return Detection BBox đã map về tọa độ ảnh gốc.
 */
Detection MapBackToOriginal(const Detection& in, const LetterboxInfo& info);

/**
 * @brief Tính IoU giữa hai bbox detection.
 */
float IoU(const Detection& a, const Detection& b);

/**
 * @brief Áp dụng Non-Maximum Suppression (NMS) trên danh sách detection.
 */
std::vector<Detection> ApplyNMS(std::vector<Detection> dets, float iou_threshold);

/**
 * @brief Parse tensor output YOLO thành detection cho từng ảnh trong batch.
 *
 * @param out0 Tensor output đầu tiên của model YOLO.
 * @param infos Thông tin letterbox theo từng ảnh input.
 * @param conf_threshold Ngưỡng confidence để lọc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<std::vector<Detection>> Kết quả detection theo batch.
 */
std::vector<std::vector<Detection>> ParseOutput(
	Ort::Value& out0,
	const std::vector<LetterboxInfo>& infos,
	float conf_threshold,
	float nms_iou_threshold);

// ── Input spec ────────────────────────────────────────────────────────

/**
 * @brief Đọc metadata input của session YOLO.
 *
 * @param session Session ONNX Runtime của model YOLO.
 * @return InputSpec Cấu hình input tensor (type/layout/shape).
 */
InputSpec GetInputSpec(Ort::Session& session);

} // namespace detail
} // namespace yolo_detector
