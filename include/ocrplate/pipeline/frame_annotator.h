/*
 * Mo ta file: Giao dien cho bo xu ly ve khung hinh va hop nhat ket qua OCR/brand/tracking.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "ocrplate/services/yolo_detector.h"
#include "ocrplate/tracking/vehicle_identity_store.h"
#include "ocrplate/tracking/byte_track_tracker.h"

struct VehicleOverlayResult {
	yolo_detector::Detection det;
	int track_id = -1;
	bool has_plate = false;
	int brand_id = -1;
	std::string accepted_plate_text;
};

struct PlateOverlayResult {
	yolo_detector::Detection det;
	int track_id = -1;
	std::string text;
	float conf_avg = 0.0f;
};

// Lich su di chuyen cua 1 track id (ve thanh polyline tren frame).
struct TrackTraceOverlay {
	int track_id = -1;
	int cls = -1;
	bool has_plate = false;
	std::vector<cv::Point> points;
};

struct FrameOverlayResult {
	std::vector<VehicleOverlayResult> vehicles;
	std::vector<PlateOverlayResult> plates;
	std::vector<TrackTraceOverlay> traces;
};

struct TrackingRuntimeContext {
	vehicle_tracker::ByteTrackTracker tracker;
	vehicle_identity_store::VehicleIdentityStore identity_store;
	bool enable_predict_on_line_cross = false;
	cv::Point gate_line_p1{0, 0};
	cv::Point gate_line_p2{0, 0};

	// Luu vet di chuyen theo track_id (tam bbox) de ve trace.
	std::unordered_map<int, std::deque<cv::Point>> track_trace_points;

	TrackingRuntimeContext();
};

// Chay pipeline va tra ve ket qua de co the ve lai tren cac frame tiep theo.
bool InferFrameOverlay(
	const cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	FrameOverlayResult& out_overlay,
	bool verbose,
	TrackingRuntimeContext* tracking_ctx = nullptr);

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
	bool verbose,
	TrackingRuntimeContext* tracking_ctx = nullptr);
