/**
 * @file yolo_nms.cpp
 * @brief Triển khai IoU và Non-Maximum Suppression cho detection YOLO.
 */
#include "ocrplate/services/yolo_detector_internal.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace yolo_detector {
namespace detail {

/**
 * @brief Tính IoU giữa hai bounding box.
 *
 * @param a Bounding box thứ nhất.
 * @param b Bounding box thứ hai.
 * @return float IoU trong [0,1].
 */
float IoU(const Detection& a, const Detection& b) {
	// Tính phần giao / phần hợp giữa 2 bbox.
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

/**
 * @brief Kiểm tra box j có bị suppress bởi box i hay không.
 *
 * @param ix1 Tọa độ x1 box i.
 * @param iy1 Tọa độ y1 box i.
 * @param ix2 Tọa độ x2 box i.
 * @param iy2 Tọa độ y2 box i.
 * @param area_i Dien tích box i.
 * @param jx1 Tọa độ x1 box j.
 * @param jy1 Tọa độ y1 box j.
 * @param jx2 Tọa độ x2 box j.
 * @param jy2 Tọa độ y2 box j.
 * @param area_j Dien tích box j.
 * @param iou_threshold Ngưỡng IoU để suppress.
 * @return true Nếu box j bị suppress.
 * @return false Nếu box j được giữ lại.
 */
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
		// Điều kiện NMS chuẩn theo IoU.
		return true;
	}

	const float min_area = std::max(1e-6f, std::min(area_i, area_j));
	// containment độ dài của box nhỏ hơn trong box lớn hơn.
	const float containment = inter / min_area;
	// Them rang buoc containment để loại box dài nhau quá nhiều.
	return containment >= kContainmentSuppression;
}

/**
 * @brief Ap dùng NMS theo class cho danh sách detection.
 *
 * @param dets Danh sách detection trước NMS.
 * @param iou_threshold Ngưỡng IoU suppress.
 * @return std::vector<Detection> Danh sách detection sau NMS.
 */
std::vector<Detection> ApplyNMS(std::vector<Detection> dets, float iou_threshold) {
	if (dets.size() <= 1) {
		return dets;
	}

	std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
		// Sort giảm dần theo score để NMS greedy hoạt động dùng.
		return a.score > b.score;
	});

	// iou_threshold <= 0 được xem là tắt NMS.
	// Van giu thứ tự score giảm dần để các nhánh o trên có thể lấy top-1 on định.
	if (iou_threshold <= 0.0f) {
		return dets;
	}

	const size_t n = dets.size();
	// areas cache diện tích để tránh tính lại trong vòng lặp đối.
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
		// i la box giu lai tiep theo theo thứ tự score giảm dần.
		kept.push_back(dets[i]);
		const float ix1 = dets[i].x1;
		const float iy1 = dets[i].y1;
		const float ix2 = dets[i].x2;
		const float iy2 = dets[i].y2;
		const float area_i = areas[i];
		for (size_t j = i + 1; j < limit; ++j) {
			if (suppressed[j]) continue;
			if (dets[j].cls != dets[i].cls) {
				// NMS theo class: không suppress box của lớp khác.
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
