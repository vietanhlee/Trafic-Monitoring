/*
 * Mô tả file: Khai báo tiện ích theo vệt (trace) để hiển thị lịch sử chuyển động của đối tượng.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "ocrplate/pipeline/frame_annotator.h"

// Cập nhật lịch sử vị trí cho từng track và tạo danh sách trace để render.
// Đầu vào:
// - bgr: frame hiện tại (dùng để clamp tọa độ vào biên frame).
// - vehicles: danh sách phương tiện đã có track_id.
// - tracking_ctx: nơi lưu map track_id -> deque điểm lịch sử.
// Đầu ra:
// - out_traces: dữ liệu trace sẵn sàng để vẽ (1 phần tử/track dạng active).
// Hàm sẽ đồng thời dọn dẹp track cũ không còn active để tránh map phình vô hạn.
void UpdateTrackTraces(
	const cv::Mat& bgr,
	const std::vector<VehicleOverlayResult>& vehicles,
	TrackingRuntimeContext& tracking_ctx,
	std::vector<TrackTraceOverlay>& out_traces);

// Vẽ polyline trace lên frame.
// Thường được gọi trước khi vẽ bbox/text để giữ trace ở lớp nền.
void DrawTrackTraces(cv::Mat& bgr, const std::vector<TrackTraceOverlay>& traces);
