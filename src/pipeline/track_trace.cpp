/*
 * Mo ta file: Trien khai ve duong di track len khung hinh de quan sat hanh vi.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/pipeline/track_trace.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ocrplate/core/app_config.h"

namespace {

cv::Point CenterPointClamped(const yolo_detector::Detection& det, int w, int h) {
	const float cx = 0.5f * (det.x1 + det.x2);
	const float cy = 0.5f * (det.y1 + det.y2);
	const int ix = std::max(0, std::min(static_cast<int>(std::lround(cx)), std::max(0, w - 1)));
	const int iy = std::max(0, std::min(static_cast<int>(std::lround(cy)), std::max(0, h - 1)));
	return cv::Point(ix, iy);
}

cv::Scalar VehicleTraceColor(int cls, bool has_plate) {
	const bool is_motor = (cls == 1);
	return is_motor
		? (has_plate ? cv::Scalar(255, 0, 255) : cv::Scalar(170, 0, 170))
		: (has_plate ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255));
}

} // namespace

void UpdateTrackTraces(
	const cv::Mat& bgr,
	const std::vector<VehicleOverlayResult>& vehicles,
	TrackingRuntimeContext& tracking_ctx,
	std::vector<TrackTraceOverlay>& out_traces) {
	out_traces.clear();

	std::unordered_set<int> active_ids;
	active_ids.reserve(vehicles.size());
	for (const auto& v : vehicles) {
		if (v.track_id <= 0) {
			continue;
		}
		active_ids.insert(v.track_id);
		auto& dq = tracking_ctx.track_trace_points[v.track_id];
		const cv::Point c = CenterPointClamped(v.det, bgr.cols, bgr.rows);
		if (dq.empty() || dq.back() != c) {
			dq.push_back(c);
		}
		while (static_cast<int>(dq.size()) > std::max(0, app_config::kTrackTraceMaxPoints)) {
			dq.pop_front();
		}
	}

	// Dọn các track không còn active để tránh phình map vô hạn.
	for (auto it = tracking_ctx.track_trace_points.begin(); it != tracking_ctx.track_trace_points.end();) {
		if (active_ids.find(it->first) == active_ids.end()) {
			it = tracking_ctx.track_trace_points.erase(it);
		} else {
			++it;
		}
	}

	out_traces.reserve(active_ids.size());
	std::unordered_set<int> pushed;
	pushed.reserve(active_ids.size());
	for (const auto& v : vehicles) {
		if (v.track_id <= 0 || pushed.find(v.track_id) != pushed.end()) {
			continue;
		}
		pushed.insert(v.track_id);
		TrackTraceOverlay t;
		t.track_id = v.track_id;
		t.cls = v.det.cls;
		t.has_plate = v.has_plate;
		const auto it = tracking_ctx.track_trace_points.find(v.track_id);
		if (it != tracking_ctx.track_trace_points.end()) {
			t.points.assign(it->second.begin(), it->second.end());
		}
		out_traces.push_back(std::move(t));
	}
}

void DrawTrackTraces(cv::Mat& bgr, const std::vector<TrackTraceOverlay>& traces) {
	for (const auto& t : traces) {
		if (t.track_id <= 0 || t.points.size() < 2) {
			continue;
		}
		const cv::Scalar color = VehicleTraceColor(t.cls, t.has_plate);
		const int thickness = std::max(1, app_config::kTrackTraceThickness);
		for (size_t i = 1; i < t.points.size(); ++i) {
			cv::line(bgr, t.points[i - 1], t.points[i], color, thickness, cv::LINE_AA);
		}
	}
}
