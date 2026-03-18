/*
 * Mo ta file: Tien ich song song hoa detect/crop/preprocess bien so tren nhieu xe.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <cstddef>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "ocrplate/services/yolo_detector.h"

namespace plate_parallel {

// Ung vien bien so sau khi map box tu output detector ve toa do trong crop xe.
struct PlateCandidate {
	size_t vehicle_index = 0;
	yolo_detector::Detection plate_in_vehicle;
	cv::Rect plate_rect_in_vehicle;
};

// Detect bien so tren moi crop xe bang da luong (moi crop la 1 inference).
// Thu tu output trung voi thu tu vehicle_crops.
std::vector<std::vector<yolo_detector::Detection>> DetectPlatesPerVehicleParallel(
	Ort::Session& plate_sess,
	const std::vector<cv::Mat>& vehicle_crops,
	float conf_threshold,
	float nms_iou_threshold);

// Loc detection hop le va quy doi thanh danh sach candidate de OCR.
std::vector<PlateCandidate> BuildPlateCandidatesParallel(
	const std::vector<std::vector<yolo_detector::Detection>>& plates_per_vehicle,
	const std::vector<cv::Mat>& vehicle_crops,
	float min_plate_score);

// Crop bien so va preprocess sang RGB uint8 HWC theo input OCR.
std::vector<cv::Mat> PreprocessPlatesParallel(
	const std::vector<PlateCandidate>& candidates,
	const std::vector<cv::Mat>& vehicle_crops,
	int out_w,
	int out_h);

} // namespace plate_parallel
