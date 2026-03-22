/*
 * Mô tả file: Giao diện cho bộ xử lý về khung hình và hợp nhất kết quả OCR/brand/tracking.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
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
	// Bounding box phương tiện trong he tọa độ frame hiện tại.
	yolo_detector::Detection det;
	// Track ID được gan boi tracker. -1 nếu chưa gan được.
	int track_id = -1;
	// Co tim thay bịển số hop le cho xe nay hay không.
	bool has_plate = false;
	// Nhan brand da chap nhan. -1 nếu chưa co kết quả.
	int brand_id = -1;
	// Chuoi bịển số da chap nhan (sau OCR + hau xu ly).
	// Có thể rong nếu chưa được xac nhan.
	std::string accepted_plate_text;
};

struct PlateOverlayResult {
	// Bounding box bịển số (tọa độ trong frame goc).
	yolo_detector::Detection det;
	// Track ID cua xe chưa bịển số nay.
	int track_id = -1;
	// Text OCR cua bịển số sau hau xu ly.
	std::string text;
	// Độ tin cậy trung binh OCR cho chuoi text.
	float conf_avg = 0.0f;
};

// Lich su di chuyen cua 1 track id (về thanh polyline tren frame).
struct TrackTraceOverlay {
	// Track ID cua duong trace cần về.
	int track_id = -1;
	// Lop phương tiện (phuc vu chon mau về).
	int cls = -1;
	// Co bịển số hop le hay không (phuc vu style về trace).
	bool has_plate = false;
	// Danh sach diem tam bbox theo lich su thời gian.
	std::vector<cv::Point> points;
};

struct FrameOverlayResult {
	// Danh sach kết quả overlay theo phương tiện.
	std::vector<VehicleOverlayResult> vehicles;
	// Danh sach kết quả overlay theo bịển số.
	std::vector<PlateOverlayResult> plates;
	// Danh sach duong trace theo track.
	std::vector<TrackTraceOverlay> traces;
};

struct TrackingRuntimeContext {
	// Tracker được duy tri xuyen suot vong lap video.
	vehicle_tracker::ByteTrackTracker tracker;
	// Kho nho danh tinh (brand/plate) theo track_id.
	vehicle_identity_store::VehicleIdentityStore identity_store;
	// Bật/tắt co che chi cho predict khi track cat qua gate line.
	bool enable_predict_on_line_cross = false;
	// Diem 1 cua gate line trong he tọa độ ROI/local frame.
	cv::Point gate_line_p1{0, 0};
	// Diem 2 cua gate line trong he tọa độ ROI/local frame.
	cv::Point gate_line_p2{0, 0};

	// Luu vet di chuyen theo track_id (tam bbox) để về trace.
	std::unordered_map<int, std::deque<cv::Point>> track_trace_points;

	// Khoi tao context voi bo tham số tracker/identity mac định tu app_config.
	TrackingRuntimeContext();
};

// Chạy pipeline infer cho 1 frame va tra về cau truc overlay da tong hop.
// Luong tong quat:
// 1) Vehicle detect + tracking.
// 2) Brand classify theo xe (có thể batch).
// 3) Plate detect + OCR + chap nhan kết quả theo track.
// 4) Tong hop dữ liệu để về lai o frame hiện tại/nhung frame cache tiep theo.
// Tra về true nếu frame co kết quả hop le để về, false nếu không co gi cần overlay.
bool InferFrameOverlay(
	const cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	FrameOverlayResult& out_overlay,
	bool verbose,
	TrackingRuntimeContext* tracking_ctx = nullptr);

// Vẽ overlay len frame tu kết quả da tinh san trong FrameOverlayResult.
// Ham nay không chạy infer, chi render.
void DrawFrameOverlay(cv::Mat& bgr, const FrameOverlayResult& overlay);

// Vẽ FPS len frame để debug throughput realtime.
void DrawFps(cv::Mat& bgr, double fps);

// API muc cao cho chế độ "annotate truc tiep" tren frame.
// Ham nay thường được dùng o mode anh don va mot so flow video don gian.
// Luong xu ly: detect vehicle -> brand/plate+ocr -> về kết quả truc tiep len bgr.
bool AnnotateFrame(
	cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool verbose,
	TrackingRuntimeContext* tracking_ctx = nullptr);
