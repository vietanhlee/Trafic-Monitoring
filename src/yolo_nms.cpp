#include "yolo_detector_internal.h"

#include <algorithm>
#include <vector>

namespace yolo_detector {
namespace detail {

float IoU(const Detection& a, const Detection& b) {
	const float xx1 = std::max(a.x1, b.x1);
	const float yy1 = std::max(a.y1, b.y1);
	const float xx2 = std::min(a.x2, b.x2);
	const float yy2 = std::min(a.y2, b.y2);
	const float w = std::max(0.0f, xx2 - xx1);
	const float h = std::max(0.0f, yy2 - yy1);
	const float inter = w * h;
	const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
	const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
	const float uni = area_a + area_b - inter;
	if (uni <= 0.0f) {
		return 0.0f;
	}
	return inter / uni;
}

static constexpr size_t kMaxNmsInput = 300;

static void NmsSingleClass(
		const std::vector<Detection>& dets,
		const std::vector<size_t>& indices,
		const std::vector<float>& areas,
		float iou_threshold,
		std::vector<Detection>& kept) {
	const size_t n = indices.size();
	if (n == 0) return;

	const size_t limit = std::min(n, kMaxNmsInput);
	std::vector<bool> suppressed(limit, false);

	for (size_t ii = 0; ii < limit; ++ii) {
		if (suppressed[ii]) continue;
		const size_t i = indices[ii];
		kept.push_back(dets[i]);

		const float ix1 = dets[i].x1;
		const float iy1 = dets[i].y1;
		const float ix2 = dets[i].x2;
		const float iy2 = dets[i].y2;
		const float area_i = areas[i];

		for (size_t jj = ii + 1; jj < limit; ++jj) {
			if (suppressed[jj]) continue;
			const size_t j = indices[jj];

			const float xx1 = std::max(ix1, dets[j].x1);
			const float yy1 = std::max(iy1, dets[j].y1);
			const float xx2 = std::min(ix2, dets[j].x2);
			const float yy2 = std::min(iy2, dets[j].y2);

			if (xx2 <= xx1 || yy2 <= yy1) continue;

			const float inter = (xx2 - xx1) * (yy2 - yy1);
			const float uni = area_i + areas[j] - inter;
			if (uni > 0.0f && inter >= iou_threshold * uni) {
				suppressed[jj] = true;
			}
		}
	}
}

std::vector<Detection> ApplyNMS(std::vector<Detection> dets, float iou_threshold) {
	if (dets.size() <= 1) {
		return dets;
	}

	std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
		return a.score > b.score;
	});

	const size_t n = dets.size();
	std::vector<float> areas(n);
	for (size_t i = 0; i < n; ++i) {
		areas[i] = std::max(0.0f, dets[i].x2 - dets[i].x1) * std::max(0.0f, dets[i].y2 - dets[i].y1);
	}

	int max_cls = 0;
	for (size_t i = 0; i < n; ++i) {
		if (dets[i].cls > max_cls) max_cls = dets[i].cls;
	}

	if (max_cls <= 2) {
		std::vector<std::vector<size_t>> cls_indices(static_cast<size_t>(max_cls + 1));
		for (size_t i = 0; i < n; ++i) {
			cls_indices[static_cast<size_t>(dets[i].cls)].push_back(i);
		}

		std::vector<Detection> kept;
		kept.reserve(std::min(n, size_t(64)));
		for (auto& indices : cls_indices) {
			NmsSingleClass(dets, indices, areas, iou_threshold, kept);
		}
		std::sort(kept.begin(), kept.end(), [](const Detection& a, const Detection& b) {
			return a.score > b.score;
		});
		return kept;
	}

	const size_t limit = std::min(n, kMaxNmsInput);
	std::vector<Detection> kept;
	kept.reserve(limit);
	std::vector<bool> suppressed(limit, false);
	for (size_t i = 0; i < limit; ++i) {
		if (suppressed[i]) continue;
		kept.push_back(dets[i]);
		const float ix1 = dets[i].x1;
		const float iy1 = dets[i].y1;
		const float ix2 = dets[i].x2;
		const float iy2 = dets[i].y2;
		const float area_i = areas[i];
		for (size_t j = i + 1; j < limit; ++j) {
			if (suppressed[j]) continue;
			const float xx1 = std::max(ix1, dets[j].x1);
			const float yy1 = std::max(iy1, dets[j].y1);
			const float xx2 = std::min(ix2, dets[j].x2);
			const float yy2 = std::min(iy2, dets[j].y2);
			if (xx2 <= xx1 || yy2 <= yy1) continue;
			const float inter = (xx2 - xx1) * (yy2 - yy1);
			const float uni = area_i + areas[j] - inter;
			if (uni > 0.0f && inter >= iou_threshold * uni) {
				suppressed[j] = true;
			}
		}
	}
	return kept;
}

} // namespace detail
} // namespace yolo_detector
