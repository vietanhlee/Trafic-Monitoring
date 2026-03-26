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
	// Bounding box phương tiện trong hệ tọa độ frame hiện tại.
	yolo_detector::Detection det;
	// Track ID được gán bởi tracker. -1 nếu chưa gán được.
	int track_id = -1;
	// Có tìm thấy biển số hợp lệ cho xe này hay không.
	bool has_plate = false;
	// Nhãn brand đã chấp nhận. -1 nếu chưa có kết quả.
	int brand_id = -1;
	// Chuỗi biển số đã chấp nhận (sau OCR + hậu xử lý).
	// Có thể rỗng nếu chưa được xác nhận.
	std::string accepted_plate_text;
};

struct PlateOverlayResult {
	// Bounding box biển số (tọa độ trong frame gốc).
	yolo_detector::Detection det;
	// Track ID của xe chứa biển số này.
	int track_id = -1;
	// Text OCR của biển số sau hậu xử lý.
	std::string text;
	// Độ tin cậy trung bình OCR cho chuỗi text.
	float conf_avg = 0.0f;
};

// Lịch sử di chuyển của 1 track id (vẽ thành polyline trên frame).
struct TrackTraceOverlay {
	// Track ID của đường trace cần vẽ.
	int track_id = -1;
	// Lớp phương tiện (phục vụ chọn màu vẽ).
	int cls = -1;
	// Có biển số hợp lệ hay không (phục vụ style vẽ trace).
	bool has_plate = false;
	// Danh sách điểm tâm bbox theo lịch sử thời gian.
	std::vector<cv::Point> points;
};

struct FrameOverlayResult {
	// Danh sách kết quả overlay theo phương tiện.
	std::vector<VehicleOverlayResult> vehicles;
	// Danh sách kết quả overlay theo biển số.
	std::vector<PlateOverlayResult> plates;
	// Danh sách đường trace theo track.
	std::vector<TrackTraceOverlay> traces;
};

struct TrackingRuntimeContext {
	// Tracker được duy trì xuyên suốt vòng lặp video.
	vehicle_tracker::ByteTrackTracker tracker;
	// Kho nhớ danh tính (brand/plate) theo track_id.
	vehicle_identity_store::VehicleIdentityStore identity_store;
	// Bật/tắt cơ chế chỉ cho predict khi track cắt qua gate line.
	bool enable_predict_on_line_cross = false;
	// Điểm 1 của gate line trong hệ tọa độ ROI/local frame.
	cv::Point gate_line_p1{0, 0};
	// Điểm 2 của gate line trong hệ tọa độ ROI/local frame.
	cv::Point gate_line_p2{0, 0};

	// Lưu vết di chuyển theo track_id (tâm bbox) để vẽ trace.
	std::unordered_map<int, std::deque<cv::Point>> track_trace_points;

	// Khởi tạo context với bộ tham số tracker/identity mặc định từ app_config.
	TrackingRuntimeContext();
};

// Chạy pipeline infer cho 1 frame và trả về cấu trúc overlay đã tổng hợp.
// Luồng tổng quát:
// 1) Vehicle detect + tracking.
// 2) Brand classify theo xe (có thể batch).
// 3) Plate detect + OCR + chấp nhận kết quả theo track.
// 4) Tổng hợp dữ liệu để vẽ lại ở frame hiện tại/những frame cache tiếp theo.
// Trả về true nếu frame có kết quả hợp lệ để vẽ, false nếu không có gì cần overlay.
bool InferFrameOverlay(
	const cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	FrameOverlayResult& out_overlay,
	bool verbose,
	TrackingRuntimeContext* tracking_ctx = nullptr);

// Vẽ overlay lên frame từ kết quả đã tính sẵn trong FrameOverlayResult.
// Hàm này không chạy infer, chỉ render.
void DrawFrameOverlay(cv::Mat& bgr, const FrameOverlayResult& overlay);

// Vẽ FPS lên frame để debug throughput realtime.
void DrawFps(cv::Mat& bgr, double fps);

// API mức cao cho chế độ "annotate trực tiếp" trên frame.
// Hàm này thường được dùng ở mode ảnh đơn và một số flow video đơn giản.
// Luồng xử lý: detect vehicle -> brand/plate+ocr -> vẽ kết quả trực tiếp lên bgr.
bool AnnotateFrame(
	cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool verbose,
	TrackingRuntimeContext* tracking_ctx = nullptr);
