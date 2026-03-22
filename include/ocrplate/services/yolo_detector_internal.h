/**
 * @file yolo_detector_internal.h
 * @brief Khai bao helper noi bo cho preprocess, parse output va NMS cua YOLO.
 */
#pragma once
// Internal helpers cho yolo_detector – KHÔNG dùng ngoài module này.

#include "ocrplate/services/yolo_detector.h"

#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

namespace yolo_detector {
namespace detail {

// ── Preprocess ────────────────────────────────────────────────────────

/**
 * @brief Resize + letterbox anh BGR về kích thước model roi doi sang RGB.
 *
 * @param bgr Anh đầu vào BGR.
 * @param target_w Chiều rộng input model.
 * @param target_h Chiều cao input model.
 * @param info [out] Thong tin scale/pad để map nguoc bbox.
 * @return cv::Mat Anh RGB sau letterbox.
 */
cv::Mat LetterboxToSizeRGB(const cv::Mat& bgr, int target_w, int target_h, LetterboxInfo& info);

/**
 * @brief Nap batch anh RGB vào tensor layout NCHW.
 *
 * @tparam T Kieu dữ liệu tensor (float hoặc uint8).
 * @param rgbs_u8 Danh sach anh RGB da letterbox, cung kích thước h x w.
 * @param h Chiều cao input tensor.
 * @param w Chiều rộng input tensor.
 * @param out [out] Buffer tensor theo thứ tự [N][C][H][W].
 * @param scale_01 true nếu cần scale [0..255] -> [0..1].
 */
template <typename T>
void FillTensorFromRGB_NCHW(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

/**
 * @brief Nap batch anh RGB vào tensor layout NHWC.
 *
 * @tparam T Kieu dữ liệu tensor (float hoặc uint8).
 * @param rgbs_u8 Danh sach anh RGB da letterbox, cung kích thước h x w.
 * @param h Chiều cao input tensor.
 * @param w Chiều rộng input tensor.
 * @param out [out] Buffer tensor theo thứ tự [N][H][W][C].
 * @param scale_01 true nếu cần scale [0..255] -> [0..1].
 */
template <typename T>
void FillTensorFromRGB_NHWC(const std::vector<cv::Mat>& rgbs_u8, int h, int w, std::vector<T>& out, bool scale_01);

// ── Postprocess ───────────────────────────────────────────────────────

/**
 * @brief Map bbox tu he tọa độ letterbox về he tọa độ anh goc.
 *
 * @param in BBox trong he tọa độ input model.
 * @param info Thong tin letterbox (scale/pad/original size).
 * @return Detection BBox da map về tọa độ anh goc.
 */
Detection MapBackToOriginal(const Detection& in, const LetterboxInfo& info);

/**
 * @brief Tinh IoU giua hai bbox detection.
 */
float IoU(const Detection& a, const Detection& b);

/**
 * @brief Ap dùng Non-Maximum Suppression (NMS) tren danh sach detection.
 */
std::vector<Detection> ApplyNMS(std::vector<Detection> dets, float iou_threshold);

/**
 * @brief Parse tensor output YOLO thanh detection cho tung anh trong batch.
 *
 * @param out0 Tensor output dau tien cua model YOLO.
 * @param infos Thong tin letterbox theo tung anh input.
 * @param conf_threshold Ngưỡng confidence để loc detection.
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
 * @brief Đọc metadata input cua session YOLO.
 *
 * @param session Session ONNX Runtime cua model YOLO.
 * @return InputSpec Cau hinh input tensor (type/layout/shape).
 */
InputSpec GetInputSpec(Ort::Session& session);

} // namespace detail
} // namespace yolo_detector
