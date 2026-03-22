/*
 * Mô tả file: Giao diện bộ theo vệt ByteTrack cho phương tiện qua nhiều frame.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "ocrplate/core/detection.h"

namespace vehicle_tracker {

struct TrackState {
	// ID định danh track duy nhất trong vòng đời tracker.
	// Giá trị > 0 là hợp lệ; -1 là chưa khởi tạo.
	int track_id = -1;
	// Detection hiện tại đại diện cho track (thường là box đã predict hoặc đo đạc mới nhất).
	yolo_detector::Detection det;
	// Số lần track được match thành công với detection.
	// Dùng để quyết định trạng thái confirmed.
	int hit_count = 0;
	// Số frame liên tiếp track không được match.
	// Vượt ngưỡng sẽ bị xóa track.
	int missed_count = 0;
	// Đánh dấu track đã "ổn định" hay chưa.
	// Track unconfirmed thường dễ bị xóa nhanh hơn để tránh noise.
	bool is_confirmed = false;

	// Vận tốc mô hình constant-velocity trong không gian (cx, cy, w, h).
	// Dùng cho bước predict giữa các frame khi chưa có detection mới.
	float vx = 0.0f;
	float vy = 0.0f;
	float vw = 0.0f;
	float vh = 0.0f;

	// Số frame thực đã trôi qua kể từ lần match detection gần nhất.
	// Dùng để tính dt cho bước predict/update vận tốc.
	int frames_since_update = 0;

	// Có thông tin đo đạc thật (measurement) hay chưa.
	// Nếu false, tracker chưa có cơ sở để tính vận tốc.
	bool has_last_meas = false;
	// Tâm bbox đo được từ lần match thật gần nhất.
	float last_meas_cx = 0.0f;
	// Tâm bbox đo được từ lần match thật gần nhất.
	float last_meas_cy = 0.0f;
	// Chiều rộng bbox đo được từ lần match thật gần nhất.
	float last_meas_w = 0.0f;
	// Chiều cao bbox đo được từ lần match thật gần nhất.
	float last_meas_h = 0.0f;
};

class ByteTrackTracker {
public:
	// Khoi tao tracker voi bo tham số ghep noi va vong doi track.
	// - iou_threshold: ngưỡng IoU cho stage 1 (high-score detections).
	// - max_missed_frames: so frame tối đa cho phep track confirmed bị mất trước khi xóa.
	// - min_confirmed_hits: so hit tới thieu để track được xem la confirmed.
	// - high_score_threshold: ngưỡng tach nhom detection high.
	// - low_score_threshold: ngưỡng duoi cua nhom detection low để cuu track.
	// - iou_threshold_low: ngưỡng IoU cho stage 2 (match low-score).
	ByteTrackTracker(
		float iou_threshold,
		int max_missed_frames,
		int min_confirmed_hits,
		float high_score_threshold = 0.0f,
		float low_score_threshold = 0.0f,
		float iou_threshold_low = 0.0f);

	// Tien tracker len 1 frame thời gian thuc (predict-only).
	// Cần gọi o moi frame video, ke ca frame không chạy detector,
	// để mở hinh van toc va missed_count phan anh dùng nhip thời gian.
	void AdvanceFrame();

	// Cap nhất tracker bang detections frame hiện tại.
	// Đầu ra la mang track_id theo cung thứ tự detections đầu vào:
	// - track_id >= 1: detection da được gan vào mot track.
	// - -1: detection không được gan track hop le.
	std::vector<int> Update(const std::vector<yolo_detector::Detection>& detections);

	// Xóa toàn bộ trang thai noi bo va reset bo dem ID về trang thai ban dau.
	void Reset();

	// Lấy danh sach track hiện co (chi đọc).
	// Huu ich cho debug, overlay va thống kê.
	const std::vector<TrackState>& GetTracks() const { return tracks_; }

private:
	// Ngưỡng IoU cho stage 1 matching (confirmed/unconfirmed voi high detections).
	float iou_threshold_ = 0.3f;
	// Ngưỡng IoU cho stage 2 matching với low-score detections.
	float iou_threshold_low_ = 0.0f;
	// Ngưỡng score bắt đầu nhóm high detections.
	float high_score_threshold_ = 0.0f;
	// Ngưỡng score bắt đầu nhóm low detections.
	float low_score_threshold_ = 0.0f;
	// Số frame missed tối đa cho track confirmed.
	int max_missed_frames_ = 8;
	// Số hit tối thiểu để chuyển track sang confirmed.
	int min_confirmed_hits_ = 1;
	// ID tiếp theo sẽ cấp cho track mới.
	int next_track_id_ = 1;
	// Bộ đếm frame nội bộ (phục vụ thống kê/quan sát).
	std::int64_t frame_index_ = 0;
	// Danh sách trạng thái tất cả track dạng được quản lý.
	std::vector<TrackState> tracks_;
};

} // namespace vehicle_tracker
