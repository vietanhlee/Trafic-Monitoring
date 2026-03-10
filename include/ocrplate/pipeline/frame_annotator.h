#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "yolo_detector.h"

struct VehicleOverlayResult {
	yolo_detector::Detection det;
	bool has_plate = false;
	int brand_id = -1;
};

struct PlateOverlayResult {
	yolo_detector::Detection det;
	std::string text;
	float conf_avg = 0.0f;
};

struct FrameOverlayResult {
	std::vector<VehicleOverlayResult> vehicles;
	std::vector<PlateOverlayResult> plates;
};

// Chay pipeline va tra ve ket qua de co the ve lai tren cac frame tiep theo.
bool InferFrameOverlay(
	const cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	FrameOverlayResult& out_overlay,
	bool verbose);

// Ve lai overlay tu ket qua da infer truoc do.
void DrawFrameOverlay(cv::Mat& bgr, const FrameOverlayResult& overlay);

// Vẽ FPS lên frame/video output.
void DrawFps(cv::Mat& bgr, double fps);

// Chạy pipeline annotate cho 1 frame:
// detect vehicle -> branch brand(batch) song song với plate detect(batch)+ocr(batch) -> vẽ kết quả.
bool AnnotateFrame(
	cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool verbose);
