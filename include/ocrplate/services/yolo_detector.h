/**
 * @file yolo_detector.h
 * @brief Khai bao API detect YOLO cho anh don va batch anh.
 *
 * Module nay nhan anh BGR OpenCV, preprocess theo input model,
 * infer ONNX, parse output, map bbox về he tọa độ anh goc va ap NMS.
 */
#pragma once

#include <cstdint>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocrplate/core/detection.h"

namespace yolo_detector {

/**
 * @brief Thong tin letterbox để map tọa độ bbox về anh goc.
 */
struct LetterboxInfo {
	/** @brief Chiều rộng anh goc trước resize/letterbox. */
	int orig_w = 0;
	/** @brief Chiều cao anh goc trước resize/letterbox. */
	int orig_h = 0;
	/** @brief Chiều rộng input tensor dua vào model. */
	int in_w = 0;
	/** @brief Chiều cao input tensor dua vào model. */
	int in_h = 0;
	/** @brief Ti le scale để fit anh goc vào khung model (giu ti le). */
	float scale = 1.0f;
	/** @brief So pixel pad theo truc X (vien trai/phai). */
	int pad_x = 0;
	/** @brief So pixel pad theo truc Y (vien tren/duoi). */
	int pad_y = 0;
};

/**
 * @brief Mô tả metadata input tensor cua model YOLO.
 */
struct InputSpec {
	/** @brief Kieu dữ liệu tensor input ONNX (float32/float16/uint8...). */
	ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	/** @brief true: NCHW, false: NHWC. */
	bool nchw = true;
	/**
	 * @brief Kích thước batch mong doi.
	 *
	 * Gia tri -1 nghia la dynamic batch.
	 */
	int64_t n = -1;
	/** @brief So kenh input (thường la 3). */
	int64_t c = 3;
	/** @brief Chiều cao tensor input model. */
	int64_t h = 640;
	/** @brief Chiều rộng tensor input model. */
	int64_t w = 640;
};

/**
 * @brief Detect YOLO theo batch anh BGR.
 *
 * @param session Session ONNX Runtime cua model detect.
 * @param bgr_images Danh sach anh đầu vào BGR.
 * @param conf_threshold Ngưỡng confidence để loc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<std::vector<Detection>> Kết quả detect theo tung anh.
 *
 * @note Ham tu xu ly input layout NCHW/NHWC va type float/uint8 theo model.
 * @note BBox được map nguoc về he tọa độ anh goc sau letterbox.
 */
std::vector<std::vector<Detection>> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold,
	float nms_iou_threshold = 0.35f);

/**
 * @brief Detect YOLO cho mot anh don.
 *
 * @param session Session ONNX Runtime cua model detect.
 * @param bgr_image Anh đầu vào BGR.
 * @param conf_threshold Ngưỡng confidence để loc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<Detection> Danh sach detection cua anh đầu vào.
 */
std::vector<Detection> RunSingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	float conf_threshold,
	float nms_iou_threshold = 0.35f);

} // namespace yolo_detector
