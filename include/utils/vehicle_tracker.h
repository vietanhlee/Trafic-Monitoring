/*
 * Mô tả file: Header tương thích cũ cho vehicle tracker ở nhánh utils.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "yolo_detector.h"

namespace vehicle_tracker {

struct TrackState {
	// ID duy nhất của track.
	int track_id = -1;
	// Detection hiện tại đại diện cho track.
	yolo_detector::Detection det;
	// Số lần match thành công với detection.
	int hit_count = 0;
	// Số frame liên tiếp không match detection.
	int missed_count = 0;
	// Track đã đủ điều kiện ổn định hay chưa.
	bool is_confirmed = false;

	// Vận tốc mô hình constant-velocity trong không gian (cx, cy, w, h).
	float vx = 0.0f;
	float vy = 0.0f;
	float vw = 0.0f;
	float vh = 0.0f;

	// Số frame kể từ lần đo đạc thật gần nhất.
	int frames_since_update = 0;

	// Có đo đạc thật gần nhất hay chưa.
	bool has_last_meas = false;
	// Tâm bbox ở lần measurement gần nhất.
	float last_meas_cx = 0.0f;
	// Tâm bbox ở lần measurement gần nhất.
	float last_meas_cy = 0.0f;
	// Chiều rộng bbox ở lần measurement gần nhất.
	float last_meas_w = 0.0f;
	// Chiều cao bbox ở lần measurement gần nhất.
	float last_meas_h = 0.0f;
};

class ByDetectionTracker {
public:
	// Khởi tạo tracker theo kiểu by-detection.
	ByDetectionTracker(
		float iou_threshold,
		int max_missed_frames,
		int min_confirmed_hits,
		float high_score_threshold = 0.0f,
		float low_score_threshold = 0.0f,
		float iou_threshold_low = 0.0f);

	// Tiến model tracker lên 1 frame (predict-only).
	void AdvanceFrame();

	// Cập nhật tracker bằng detections frame hiện tại.
	// Trả về mảng track_id theo thứ tự detections đầu vào.
	std::vector<int> Update(const std::vector<yolo_detector::Detection>& detections);

	// Reset toàn bộ state tracker.
	void Reset();

	// Lấy danh sách track hiện có.
	const std::vector<TrackState>& GetTracks() const { return tracks_; }

private:
	// Ngưỡng IoU stage 1.
	float iou_threshold_ = 0.3f;
	// Ngưỡng IoU stage 2 low-score.
	float iou_threshold_low_ = 0.0f;
	// Ngưỡng high-score detection.
	float high_score_threshold_ = 0.0f;
	// Ngưỡng low-score detection.
	float low_score_threshold_ = 0.0f;
	// Số frame missed tối đa.
	int max_missed_frames_ = 8;
	// Số hit tối thiểu để confirm track.
	int min_confirmed_hits_ = 1;
	// ID tiếp theo cho track mới.
	int next_track_id_ = 1;
	// Bộ đếm frame nội bộ.
	std::int64_t frame_index_ = 0;
	// Danh sách track dạng quản lý.
	std::vector<TrackState> tracks_;
};

} // namespace vehicle_tracker
