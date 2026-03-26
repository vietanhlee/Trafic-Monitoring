/*
 * Mô tả file: Triển khai song song hóa các bước liên quan đến biển số.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#include "ocrplate/utils/plate_parallel.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <utility>
#include <vector>

#include "ocrplate/core/app_config.h"
#include "ocrplate/utils/image_preprocess.h"
#include "ocrplate/utils/parallel_utils.h"
#include "ocrplate/utils/rect_utils.h"

namespace {

// Chia việc theo kiểu "work stealing" đơn giản: mỗi worker lấy index tiếp theo
// bằng atomic để cân bằng tải khi mỗi crop có độ phức tạp khác nhau.
template <typename Fn>
void ParallelForEach(
	size_t item_count,
	size_t max_workers,
	Fn&& fn) {
	if (item_count == 0) {
		return;
	}

	const size_t resolved = parallel_utils::ResolveWorkerCount(item_count);
	const size_t worker_count = std::max<size_t>(1, std::min(resolved, max_workers));
	if (worker_count <= 1 || item_count <= 1) {
		for (size_t i = 0; i < item_count; ++i) {
			fn(i);
		}
		return;
	}

	std::atomic<size_t> next_index{0};
	std::vector<std::thread> workers;
	workers.reserve(worker_count);
	for (size_t w = 0; w < worker_count; ++w) {
		workers.emplace_back([&]() {
			while (true) {
				const size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
				if (i >= item_count) {
					break;
				}
				fn(i);
			}
		});
	}
	for (auto& t : workers) {
		t.join();
	}
}

} // namespace

namespace plate_parallel {

std::vector<std::vector<yolo_detector::Detection>> DetectPlatesPerVehicleParallel(
	Ort::Session& plate_sess,
	const std::vector<cv::Mat>& vehicle_crops,
	float conf_threshold,
	float nms_iou_threshold) {
	std::vector<std::vector<yolo_detector::Detection>> plates_per_vehicle(vehicle_crops.size());
	if (vehicle_crops.empty()) {
		return plates_per_vehicle;
	}

	// Nghiệp vụ: detect biển số trên TỪNG crop phương tiện, nhưng xử lý đa luồng
	// ở tầng ngoài để giảm tổng độ trễ khi có nhiều xe trong 1 frame.
	ParallelForEach(
		vehicle_crops.size(),
		app_config::kPlateDetectMaxWorkers,
		[&](size_t i) {
			// Mỗi crop xe chạy detect độc lập; output lưu theo index xe tương ứng.
			plates_per_vehicle[i] = yolo_detector::RunSingle(
				plate_sess,
				vehicle_crops[i],
				conf_threshold,
				nms_iou_threshold);
		});

	return plates_per_vehicle;
}

std::vector<PlateCandidate> BuildPlateCandidatesParallel(
	const std::vector<std::vector<yolo_detector::Detection>>& plates_per_vehicle,
	const std::vector<cv::Mat>& vehicle_crops,
	float min_plate_score) {
	std::vector<std::vector<PlateCandidate>> candidates_per_vehicle(plates_per_vehicle.size());

	// Công việc này CPU-bound, chỉ cần số worker vừa phải để tránh tạo quá nhiều thread.
	ParallelForEach(
		plates_per_vehicle.size(),
		app_config::kPlateDetectMaxWorkers,
		[&](size_t i) {
			const auto& dets = plates_per_vehicle[i];
			auto& out = candidates_per_vehicle[i];

			// Nghiệp vụ: mỗi xe tối đa 1 biển số.
			// Trong no-NMS mode, dets da được sort giảm dần theo score,
			// vì vậy chỉ cần lấy detection hợp lệ đầu tiên (top-1) rồi dùng.
			for (const auto& p : dets) {
				if (p.score < min_plate_score) {
					// Cắt ngưỡng score sớm để giảm công việc crop/preprocess OCR.
					continue;
				}
				cv::Rect pr_local = rect_utils::ToRectClamped(
					p.x1,
					p.y1,
					p.x2,
					p.y2,
					vehicle_crops[i].cols,
					vehicle_crops[i].rows);
				if (pr_local.width <= 2 || pr_local.height <= 2) {
					continue;
				}
				out.reserve(1);
				PlateCandidate c;
				c.vehicle_index = i;
				c.plate_in_vehicle = p;
				c.plate_rect_in_vehicle = pr_local;
				out.push_back(std::move(c));
				break;
			}
		});

	std::vector<PlateCandidate> candidates;
	size_t total = 0;
	for (const auto& per_vehicle : candidates_per_vehicle) {
		total += per_vehicle.size();
	}
	candidates.reserve(total);
	for (auto& per_vehicle : candidates_per_vehicle) {
		// Flatten kết quả tương ứng thứ tự vehicle_index ban đầu.
		for (auto& c : per_vehicle) {
			candidates.push_back(std::move(c));
		}
	}
	return candidates;
}

std::vector<cv::Mat> PreprocessPlatesParallel(
	const std::vector<PlateCandidate>& candidates,
	const std::vector<cv::Mat>& vehicle_crops,
	int out_w,
	int out_h) {
	std::vector<cv::Mat> plate_rgb_ocr(candidates.size());
	if (candidates.empty()) {
		return plate_rgb_ocr;
	}

	// Preprocess OCR là bước memory-bound, giới hạn worker để giảm context switch.
	ParallelForEach(
		candidates.size(),
		app_config::kPlateDetectMaxWorkers,
		[&](size_t i) {
			const auto& c = candidates[i];
			const cv::Mat& vehicle_bgr = vehicle_crops[c.vehicle_index];
			// Crop theo bbox biển số trong crop xe, sau đó chuyển thành RGB uint8 HWC cho OCR.
			cv::Mat plate_bgr = vehicle_bgr(c.plate_rect_in_vehicle);
			plate_rgb_ocr[i] = image_preprocess::PreprocessMatRgbU8Hwc(plate_bgr, out_w, out_h);
		});

	return plate_rgb_ocr;
}

} // namespace plate_parallel
