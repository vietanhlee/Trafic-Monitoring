#pragma once

#include <cstdint>
#include <vector>

#include "ocrplate/core/detection.h"

namespace vehicle_tracker {

struct TrackState {
	int track_id = -1;
	yolo_detector::Detection det;
	int hit_count = 0;
	int missed_count = 0;
	bool is_confirmed = false;

	// Simple constant-velocity model in (cx, cy, w, h) space.
	float vx = 0.0f;
	float vy = 0.0f;
	float vw = 0.0f;
	float vh = 0.0f;

	// Number of real frames since the last associated detection.
	int frames_since_update = 0;

	// Last measurement (not predicted) in (cx, cy, w, h).
	bool has_last_meas = false;
	float last_meas_cx = 0.0f;
	float last_meas_cy = 0.0f;
	float last_meas_w = 0.0f;
	float last_meas_h = 0.0f;
};

class ByteTrackTracker {
public:
	ByteTrackTracker(
		float iou_threshold,
		int max_missed_frames,
		int min_confirmed_hits,
		float high_score_threshold = 0.0f,
		float low_score_threshold = 0.0f,
		float iou_threshold_low = 0.0f);

	// Advance internal track predictions by 1 real frame.
	// This should be called for every video frame, even if you only run inference every N frames.
	void AdvanceFrame();

	// Update tracker from detections of current frame and return track_id per detection.
	std::vector<int> Update(const std::vector<yolo_detector::Detection>& detections);

	void Reset();

	const std::vector<TrackState>& GetTracks() const { return tracks_; }

private:
	float iou_threshold_ = 0.3f;
	float iou_threshold_low_ = 0.0f;
	float high_score_threshold_ = 0.0f;
	float low_score_threshold_ = 0.0f;
	int max_missed_frames_ = 8;
	int min_confirmed_hits_ = 1;
	int next_track_id_ = 1;
	std::int64_t frame_index_ = 0;
	std::vector<TrackState> tracks_;
};

} // namespace vehicle_tracker
