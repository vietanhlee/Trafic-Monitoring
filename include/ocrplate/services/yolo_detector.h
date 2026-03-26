/**
 * @file yolo_detector.h
 * @brief Khai báo API detect YOLO cho ảnh đơn và batch ảnh.
 *
 * Module này nhận ảnh BGR OpenCV, preprocess theo input model,
 * infer ONNX, parse output, map bbox về hệ tọa độ ảnh gốc và áp NMS.
 */
#pragma once

#include <cstdint>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocrplate/core/detection.h"

namespace yolo_detector {

/**
 * @brief Thông tin letterbox để map tọa độ bbox về ảnh gốc.
 */
struct LetterboxInfo {
	/** @brief Chiều rộng ảnh gốc trước resize/letterbox. */
	int orig_w = 0;
	/** @brief Chiều cao ảnh gốc trước resize/letterbox. */
	int orig_h = 0;
	/** @brief Chiều rộng input tensor đưa vào model. */
	int in_w = 0;
	/** @brief Chiều cao input tensor đưa vào model. */
	int in_h = 0;
	/** @brief Tỉ lệ scale để fit ảnh gốc vào khung model (giữ tỉ lệ). */
	float scale = 1.0f;
	/** @brief Số pixel pad theo trục X (viền trái/phải). */
	int pad_x = 0;
	/** @brief Số pixel pad theo trục Y (viền trên/dưới). */
	int pad_y = 0;
};

/**
 * @brief Mô tả metadata input tensor của model YOLO.
 */
struct InputSpec {
	/** @brief Kiểu dữ liệu tensor input ONNX (float32/float16/uint8...). */
	ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	/** @brief true: NCHW, false: NHWC. */
	bool nchw = true;
	/**
	 * @brief Kích thước batch mong đợi.
	 *
	 * Giá trị -1 nghĩa là dynamic batch.
	 */
	int64_t n = -1;
	/** @brief Số kênh input (thường là 3). */
	int64_t c = 3;
	/** @brief Chiều cao tensor input model. */
	int64_t h = 640;
	/** @brief Chiều rộng tensor input model. */
	int64_t w = 640;
};

/**
 * @brief Detect YOLO theo batch ảnh BGR.
 *
 * @param session Session ONNX Runtime của model detect.
 * @param bgr_images Danh sách ảnh đầu vào BGR.
 * @param conf_threshold Ngưỡng confidence để lọc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<std::vector<Detection>> Kết quả detect theo từng ảnh.
 *
 * @note Hàm tự xử lý input layout NCHW/NHWC và type float/uint8 theo model.
 * @note BBox được map ngược về hệ tọa độ ảnh gốc sau letterbox.
 */
std::vector<std::vector<Detection>> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold,
	float nms_iou_threshold = 0.35f);

/**
 * @brief Detect YOLO cho một ảnh đơn.
 *
 * @param session Session ONNX Runtime của model detect.
 * @param bgr_image Ảnh đầu vào BGR.
 * @param conf_threshold Ngưỡng confidence để lọc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<Detection> Danh sách detection của ảnh đầu vào.
 */
std::vector<Detection> RunSingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	float conf_threshold,
	float nms_iou_threshold = 0.35f);

} // namespace yolo_detector
