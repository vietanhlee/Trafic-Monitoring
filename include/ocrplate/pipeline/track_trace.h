/*
 * Mo ta file: Khai bao tien ich theo vet (trace) de hien thi lich su chuyen dong cua doi tuong.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "ocrplate/pipeline/frame_annotator.h"

// Cập nhật lịch sử vị trí (tâm bbox) cho từng track_id và xuất ra danh sách trace để vẽ.
// - Chỉ hoạt động khi có tracking (TrackingRuntimeContext).
// - Lịch sử được lưu trong tracking_ctx.track_trace_points.
void UpdateTrackTraces(
	const cv::Mat& bgr,
	const std::vector<VehicleOverlayResult>& vehicles,
	TrackingRuntimeContext& tracking_ctx,
	std::vector<TrackTraceOverlay>& out_traces);

// Vẽ polyline trace lên frame (vẽ trước bbox/text).
void DrawTrackTraces(cv::Mat& bgr, const std::vector<TrackTraceOverlay>& traces);
