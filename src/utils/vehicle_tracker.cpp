/*
 * Mô tả file: Triển khai vehicle tracker ở nhánh utils để tương thích với code cũ.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */

namespace {

float IoU(const yolo_detector::Detection& a, const yolo_detector::Detection& b) {
	const float x1 = std::max(a.x1, b.x1);
	const float y1 = std::max(a.y1, b.y1);
	const float x2 = std::min(a.x2, b.x2);
	const float y2 = std::min(a.y2, b.y2);
	const float w = std::max(0.0f, x2 - x1);
	const float h = std::max(0.0f, y2 - y1);
	const float inter = w * h;
	if (inter <= 0.0f) {
		return 0.0f;
	}
	const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
	const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
	const float uni = area_a + area_b - inter;
	return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

struct CenterSizeBox {
	float cx = 0.0f;
	float cy = 0.0f;
	float w = 0.0f;
	float h = 0.0f;
};

CenterSizeBox ToCenterSize(const yolo_detector::Detection& d) {
	CenterSizeBox b;
	const float w = std::max(0.0f, d.x2 - d.x1);
	const float h = std::max(0.0f, d.y2 - d.y1);
	b.cx = d.x1 + w * 0.5f;
	b.cy = d.y1 + h * 0.5f;
	b.w = w;
	b.h = h;
	return b;
}

yolo_detector::Detection FromCenterSize(const CenterSizeBox& b, float score, int cls) {
	yolo_detector::Detection d;
	const float w = std::max(1.0f, b.w);
	const float h = std::max(1.0f, b.h);
	d.x1 = b.cx - w * 0.5f;
	d.y1 = b.cy - h * 0.5f;
	d.x2 = b.cx + w * 0.5f;
	d.y2 = b.cy + h * 0.5f;
	d.score = score;
	d.cls = cls;
	return d;
}

void InitializeTrackState(TrackState& t, const yolo_detector::Detection& det) {
	const CenterSizeBox b = ToCenterSize(det);
	t.det = det;
	t.frames_since_update = 0;
	t.has_last_meas = true;
	t.last_meas_cx = b.cx;
	t.last_meas_cy = b.cy;
	t.last_meas_w = b.w;
	t.last_meas_h = b.h;
	t.vx = 0.0f;
	t.vy = 0.0f;
	t.vw = 0.0f;
	t.vh = 0.0f;
}

void PredictOneFrame(TrackState& t) {
	if (!t.has_last_meas) {
		return;
	}
	// Predict from the last measurement to avoid accumulating numerical drift.
	const float dt = static_cast<float>(std::max(0, t.frames_since_update));
	CenterSizeBox pred;
	pred.cx = t.last_meas_cx + t.vx * dt;
	pred.cy = t.last_meas_cy + t.vy * dt;
	pred.w = t.last_meas_w + t.vw * dt;
	pred.h = t.last_meas_h + t.vh * dt;
	t.det = FromCenterSize(pred, t.det.score, t.det.cls);
}

void UpdateWithMeasurement(TrackState& t, const yolo_detector::Detection& meas) {
	const CenterSizeBox b = ToCenterSize(meas);
	if (t.has_last_meas) {
		const float dt = static_cast<float>(std::max(1, t.frames_since_update));
		const float new_vx = (b.cx - t.last_meas_cx) / dt;
		const float new_vy = (b.cy - t.last_meas_cy) / dt;
		const float new_vw = (b.w - t.last_meas_w) / dt;
		const float new_vh = (b.h - t.last_meas_h) / dt;
		// Smooth velocities a bit to reduce jitter.
		constexpr float kAlpha = 0.7f;
		t.vx = t.vx * kAlpha + new_vx * (1.0f - kAlpha);
		t.vy = t.vy * kAlpha + new_vy * (1.0f - kAlpha);
		t.vw = t.vw * kAlpha + new_vw * (1.0f - kAlpha);
		t.vh = t.vh * kAlpha + new_vh * (1.0f - kAlpha);
	}
	t.has_last_meas = true;
	t.last_meas_cx = b.cx;
	t.last_meas_cy = b.cy;
	t.last_meas_w = b.w;
	t.last_meas_h = b.h;
	t.frames_since_update = 0;
	t.det = meas;
}

// Hungarian / assignment (min-cost) for rectangular matrix with n <= m.
// 1-indexed internally.
std::vector<int> SolveAssignmentMinCost(const std::vector<std::vector<float>>& a) {
	// a is 1..n, 1..m
	const int n = static_cast<int>(a.size()) - 1;
	const int m = static_cast<int>(a[0].size()) - 1;
	const float INF = std::numeric_limits<float>::infinity();

	std::vector<float> u(n + 1, 0.0f);
	std::vector<float> v(m + 1, 0.0f);
	std::vector<int> p(m + 1, 0);
	std::vector<int> way(m + 1, 0);

	for (int i = 1; i <= n; ++i) {
		p[0] = i;
		int j0 = 0;
		std::vector<float> minv(m + 1, INF);
		std::vector<char> used(m + 1, false);
		do {
			used[j0] = true;
			const int i0 = p[j0];
			float delta = INF;
			int j1 = 0;
			for (int j = 1; j <= m; ++j) {
				if (used[j]) {
					continue;
				}
				const float cur = a[i0][j] - u[i0] - v[j];
				if (cur < minv[j]) {
					minv[j] = cur;
					way[j] = j0;
				}
				if (minv[j] < delta) {
					delta = minv[j];
					j1 = j;
				}
			}
			if (!std::isfinite(delta)) {
				// No improving path; break to avoid NaNs. This cần happen if all remaining costs are INF.
				break;
			}
			for (int j = 0; j <= m; ++j) {
				if (used[j]) {
					u[p[j]] += delta;
					v[j] -= delta;
				} else {
					minv[j] -= delta;
				}
			}
			j0 = j1;
		} while (p[j0] != 0);

		// Augmenting path.
		do {
			const int j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
		} while (j0 != 0);
	}

	std::vector<int> ans(n + 1, 0);
	for (int j = 1; j <= m; ++j) {
		if (p[j] != 0) {
			ans[p[j]] = j;
		}
	}
	return ans;
}

struct AssignmentResult {
	std::vector<std::pair<size_t, size_t>> matches; // (track_idx_in_input, det_idx_in_input)
	std::vector<size_t> unmatched_track_indices;
	std::vector<size_t> unmatched_det_indices;
};

AssignmentResult AssignTracksToDets(
	const std::vector<size_t>& track_indices,
	const std::vector<size_t>& det_indices,
	const std::vector<TrackState>& tracks,
	const std::vector<yolo_detector::Detection>& detections,
	float iou_threshold) {
	AssignmentResult out;
	const float INF = 1e9f;
	const size_t n = track_indices.size();
	const size_t m_real = det_indices.size();
	if (n == 0) {
		out.unmatched_det_indices = det_indices;
		return out;
	}
	// Add n dummy columns so every track cần choose "unmatched" without stealing a real detection.
	const size_t m = m_real + n;
	// 1-indexed matrix a[n][m].
	std::vector<std::vector<float>> a(n + 1, std::vector<float>(m + 1, 0.0f));
	for (size_t i = 0; i < n; ++i) {
		const TrackState& t = tracks[track_indices[i]];
		for (size_t j = 0; j < m_real; ++j) {
			const auto& d = detections[det_indices[j]];
			if (t.det.cls != d.cls) {
				a[i + 1][j + 1] = INF;
				continue;
			}
			const float iou = IoU(t.det, d);
			if (iou < iou_threshold) {
				a[i + 1][j + 1] = INF;
				continue;
			}
			// Minimize cost = 1 - IoU.
			a[i + 1][j + 1] = 1.0f - iou;
		}
		// Dummy columns cost.
		for (size_t j = m_real; j < m; ++j) {
			a[i + 1][j + 1] = 1.0f;
		}
	}

	std::vector<int> assign = SolveAssignmentMinCost(a); // size n+1, value in [1..m]
	std::vector<char> det_used(m_real, false);
	std::vector<char> track_used(n, false);

	for (size_t i = 0; i < n; ++i) {
		const int col = assign[i + 1];
		if (col <= 0) {
			continue;
		}
		const size_t j = static_cast<size_t>(col - 1);
		if (j >= m_real) {
			continue; // matched to dummy
		}
		const float cost = a[i + 1][j + 1];
		if (cost >= INF * 0.5f) {
			continue;
		}
		track_used[i] = true;
		det_used[j] = true;
		out.matches.push_back({i, j});
	}

	for (size_t i = 0; i < n; ++i) {
		if (!track_used[i]) {
			out.unmatched_track_indices.push_back(track_indices[i]);
		}
	}
	for (size_t j = 0; j < m_real; ++j) {
		if (!det_used[j]) {
			out.unmatched_det_indices.push_back(det_indices[j]);
		}
	}
	return out;
}

} // namespace

ByDetectionTracker::ByDetectionTracker(
	float iou_threshold,
	int max_missed_frames,
	int min_confirmed_hits,
	float high_score_threshold,
	float low_score_threshold,
	float iou_threshold_low)
	: iou_threshold_(iou_threshold),
	  iou_threshold_low_(iou_threshold_low),
	  high_score_threshold_(high_score_threshold),
	  low_score_threshold_(low_score_threshold),
	  max_missed_frames_(max_missed_frames),
	  min_confirmed_hits_(min_confirmed_hits) {
	if (iou_threshold_ <= 0.0f || iou_threshold_ > 1.0f) {
		throw std::runtime_error("Tracker IoU threshold không hop le");
	}
	if (iou_threshold_low_ <= 0.0f) {
		iou_threshold_low_ = std::min(0.20f, iou_threshold_);
	}
	if (iou_threshold_low_ <= 0.0f || iou_threshold_low_ > 1.0f) {
		throw std::runtime_error("Tracker IoU low threshold không hop le");
	}
	if (max_missed_frames_ < 0) {
		throw std::runtime_error("Tracker max missed frames không hop le");
	}
	if (min_confirmed_hits_ <= 0) {
		throw std::runtime_error("Tracker min confirmed hits không hop le");
	}
	if (high_score_threshold_ < 0.0f || high_score_threshold_ > 1.0f) {
		throw std::runtime_error("Tracker high score threshold không hop le");
	}
	if (low_score_threshold_ < 0.0f || low_score_threshold_ > 1.0f) {
		throw std::runtime_error("Tracker low score threshold không hop le");
	}
	if (low_score_threshold_ > high_score_threshold_) {
		// Allow but clamp to make ranges sensible.
		low_score_threshold_ = high_score_threshold_;
	}
}

void ByDetectionTracker::AdvanceFrame() {
	++frame_index_;
	for (auto& t : tracks_) {
		t.frames_since_update += 1;
		PredictOneFrame(t);
	}
}

std::vector<int> ByDetectionTracker::Update(const std::vector<yolo_detector::Detection>& detections) {
	std::vector<int> track_ids(detections.size(), -1);
	// Split detections by confidence (ByteTrack-style: high then low).
	std::vector<size_t> high_det_indices;
	std::vector<size_t> low_det_indices;
	high_det_indices.reserve(detections.size());
	low_det_indices.reserve(detections.size());
	for (size_t i = 0; i < detections.size(); ++i) {
		const float s = detections[i].score;
		if (s >= high_score_threshold_) {
			high_det_indices.push_back(i);
		} else if (s >= low_score_threshold_) {
			low_det_indices.push_back(i);
		}
	}

	if (detections.empty()) {
		for (auto& t : tracks_) {
			t.missed_count += 1;
		}
		tracks_.erase(
			std::remove_if(
				tracks_.begin(),
				tracks_.end(),
				[&](const TrackState& t) { return t.missed_count > max_missed_frames_; }),
			tracks_.end());
		return track_ids;
	}

	std::vector<size_t> confirmed_tracks;
	std::vector<size_t> unconfirmed_tracks;
	confirmed_tracks.reserve(tracks_.size());
	unconfirmed_tracks.reserve(tracks_.size());
	for (size_t ti = 0; ti < tracks_.size(); ++ti) {
		if (tracks_[ti].is_confirmed) {
			confirmed_tracks.push_back(ti);
		} else {
			unconfirmed_tracks.push_back(ti);
		}
	}

	std::vector<char> det_taken(detections.size(), false);
	std::vector<char> track_updated(tracks_.size(), false);

	auto mark_match = [&](size_t track_idx, size_t det_idx) {
		TrackState& t = tracks_[track_idx];
		UpdateWithMeasurement(t, detections[det_idx]);
		t.hit_count += 1;
		t.missed_count = 0;
		t.is_confirmed = (t.hit_count >= min_confirmed_hits_);
		track_ids[det_idx] = t.track_id;
		track_updated[track_idx] = true;
		det_taken[det_idx] = true;
	};

	// Stage 1: match confirmed tracks with high-score detections.
	std::vector<size_t> high_available;
	high_available.reserve(high_det_indices.size());
	for (size_t di : high_det_indices) {
		if (!det_taken[di]) {
			high_available.push_back(di);
		}
	}
	AssignmentResult stage1 = AssignTracksToDets(confirmed_tracks, high_available, tracks_, detections, iou_threshold_);
	for (const auto& m : stage1.matches) {
		const size_t track_idx = confirmed_tracks[m.first];
		const size_t det_idx = high_available[m.second];
		mark_match(track_idx, det_idx);
	}

	// Stage 1b: match unconfirmed tracks with remaining high detections (a bit stricter).
	std::vector<size_t> high_remaining;
	high_remaining.reserve(high_det_indices.size());
	for (size_t di : high_det_indices) {
		if (!det_taken[di]) {
			high_remaining.push_back(di);
		}
	}
	const float unconfirmed_iou_thr = std::min(0.80f, iou_threshold_ + 0.05f);
	AssignmentResult stage1b = AssignTracksToDets(unconfirmed_tracks, high_remaining, tracks_, detections, unconfirmed_iou_thr);
	for (const auto& m : stage1b.matches) {
		const size_t track_idx = unconfirmed_tracks[m.first];
		const size_t det_idx = high_remaining[m.second];
		mark_match(track_idx, det_idx);
	}

	// Stage 2: match unmatched confirmed tracks with low-score detections (or leftover highs) using lower IoU.
	std::vector<size_t> confirmed_unmatched;
	confirmed_unmatched.reserve(confirmed_tracks.size());
	for (size_t ti : confirmed_tracks) {
		if (!track_updated[ti]) {
			confirmed_unmatched.push_back(ti);
		}
	}
	std::vector<size_t> low_available;
	low_available.reserve(low_det_indices.size() + high_det_indices.size());
	for (size_t di : low_det_indices) {
		if (!det_taken[di]) {
			low_available.push_back(di);
		}
	}
	for (size_t di : high_det_indices) {
		if (!det_taken[di]) {
			low_available.push_back(di);
		}
	}
	AssignmentResult stage2 = AssignTracksToDets(confirmed_unmatched, low_available, tracks_, detections, iou_threshold_low_);
	for (const auto& m : stage2.matches) {
		const size_t track_idx = confirmed_unmatched[m.first];
		const size_t det_idx = low_available[m.second];
		mark_match(track_idx, det_idx);
	}

	// Create new tracks for remaining high-score detections.
	for (size_t di : high_det_indices) {
		if (di >= detections.size() || det_taken[di]) {
			continue;
		}
		TrackState t;
		t.track_id = next_track_id_++;
		InitializeTrackState(t, detections[di]);
		t.hit_count = 1;
		t.missed_count = 0;
		t.is_confirmed = (t.hit_count >= min_confirmed_hits_);
		tracks_.push_back(t);
		track_updated.push_back(true);
		track_ids[di] = t.track_id;
		det_taken[di] = true;
	}

	// Mark missed tracks and prune.
	for (size_t ti = 0; ti < tracks_.size(); ++ti) {
		if (ti < track_updated.size() && track_updated[ti]) {
			continue;
		}
		tracks_[ti].missed_count += 1;
	}

	tracks_.erase(
		std::remove_if(
			tracks_.begin(),
			tracks_.end(),
			[&](const TrackState& t) {
				// Unconfirmed tracks are removed quickly to avoid ID churn.
				if (!t.is_confirmed) {
					return t.missed_count >= 1;
				}
				return t.missed_count > max_missed_frames_;
			}),
		tracks_.end());

	return track_ids;
}

void ByDetectionTracker::Reset() {
	tracks_.clear();
	next_track_id_ = 1;
	frame_index_ = 0;
}

} // namespace vehicle_tracker
