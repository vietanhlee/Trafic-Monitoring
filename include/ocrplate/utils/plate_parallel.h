/*
 * Mô tả file: Tiện ích song song hóa detect/crop/preprocess biển số trên nhiều xe.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <cstddef>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocrplate/services/yolo_detector.h"

namespace plate_parallel {

// Ứng viên biển số sau khi map box từ output detector về tọa độ trong crop xe.
struct PlateCandidate {
	// Chỉ số xe trong mảng vehicle_crops gốc.
	size_t vehicle_index = 0;
	// Detection biển số trong hệ tọa độ local của crop xe.
	yolo_detector::Detection plate_in_vehicle;
	// Rect đã được clamp và quy đổi sang cv::Rect để crop OCR.
	cv::Rect plate_rect_in_vehicle;
};

// Detect biển số trên mỗi crop xe bằng đa luồng.
// Đầu vào:
// - plate_sess: session model detect biển số.
// - vehicle_crops: danh sách crop phương tiện.
// - conf_threshold: ngưỡng score detect biển số.
// - nms_iou_threshold: ngưỡng NMS cho branch plate.
// Đầu ra:
// - vector 2 chiều, mỗi phần tử là danh sách plate detections của từng xe.
// - Thứ tự output giữ nguyên theo thứ tự vehicle_crops.
std::vector<std::vector<yolo_detector::Detection>> DetectPlatesPerVehicleParallel(
	Ort::Session& plate_sess,
	const std::vector<cv::Mat>& vehicle_crops,
	float conf_threshold,
	float nms_iou_threshold);

// Lọc và quy đổi kết quả detect biển số thành danh sách candidate OCR.
// Hàm sẽ bỏ qua box lỗi (âm kích thước, out of bound) và giữ lại candidate hợp lệ.
std::vector<PlateCandidate> BuildPlateCandidatesParallel(
	const std::vector<std::vector<yolo_detector::Detection>>& plates_per_vehicle,
	const std::vector<cv::Mat>& vehicle_crops,
	float min_plate_score);

// Crop biển số từ vehicle crops và preprocess về input OCR.
// Đầu ra là danh sách ảnh RGB uint8 HWC kích thước cố định out_w x out_h.
std::vector<cv::Mat> PreprocessPlatesParallel(
	const std::vector<PlateCandidate>& candidates,
	const std::vector<cv::Mat>& vehicle_crops,
	int out_w,
	int out_h);

} // namespace plate_parallel
