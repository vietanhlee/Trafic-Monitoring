/*
 * Mo ta file: Trien khai Non-Maximum Suppression cho ket qua detect YOLO.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/services/yolo_detector_internal.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace yolo_detector {
namespace detail {

float IoU(const Detection& a, const Detection& b) {
	// Tinh phan giao / phan hop giua 2 bbox.
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
static constexpr float kContainmentSuppression = 0.72f;

static inline bool ShouldSuppress(
		float ix1,
		float iy1,
		float ix2,
		float iy2,
		float area_i,
		float jx1,
		float jy1,
		float jx2,
		float jy2,
		float area_j,
		float iou_threshold) {
	const float xx1 = std::max(ix1, jx1);
	const float yy1 = std::max(iy1, jy1);
	const float xx2 = std::min(ix2, jx2);
	const float yy2 = std::min(iy2, jy2);
	if (xx2 <= xx1 || yy2 <= yy1) {
		return false;
	}

	const float inter = (xx2 - xx1) * (yy2 - yy1);
	const float uni = area_i + area_j - inter;
	if (uni > 0.0f && inter >= iou_threshold * uni) {
		// Dieu kien NMS chuan theo IoU.
		return true;
	}

	const float min_area = std::max(1e-6f, std::min(area_i, area_j));
	const float containment = inter / min_area;
	// Them rang buoc containment de loai box long nhau qua nhieu.
	return containment >= kContainmentSuppression;
}
std::vector<Detection> ApplyNMS(std::vector<Detection> dets, float iou_threshold) {
	if (dets.size() <= 1) {
		return dets;
	}

	std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
		// Sort giam dan theo score de NMS greedy hoat dong dung.
		return a.score > b.score;
	});

	// iou_threshold <= 0 duoc xem la tat NMS.
	// Van giu thu tu score giam dan de cac nhanh o tren co the lay top-1 on dinh.
	if (iou_threshold <= 0.0f) {
		return dets;
	}

	const size_t n = dets.size();
	std::vector<float> areas(n);
	for (size_t i = 0; i < n; ++i) {
		areas[i] = std::max(0.0f, dets[i].x2 - dets[i].x1) * std::max(0.0f, dets[i].y2 - dets[i].y1);
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
			if (dets[j].cls != dets[i].cls) {
				// NMS theo class: khong suppress box cua lop khac.
				continue;
			}
			if (ShouldSuppress(
					ix1,
					iy1,
					ix2,
					iy2,
					area_i,
					dets[j].x1,
					dets[j].y1,
					dets[j].x2,
					dets[j].y2,
					areas[j],
					iou_threshold)) {
				suppressed[j] = true;
			}
		}
	}
	return kept;
}

} // namespace detail
} // namespace yolo_detector
