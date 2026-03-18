/*
 * Mo ta file: Trien khai song song hoa cac buoc lien quan den bien so.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
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

namespace {

// Chia viec theo kieu "work stealing" don gian: moi worker lay index tiep theo
// bang atomic de can bang tai khi moi crop co do phuc tap khac nhau.
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

cv::Rect ToRectClamped(float x1, float y1, float x2, float y2, int w, int h) {
	// Clamp toa do de dam bao crop khong vuot mien anh xe.
	int ix1 = std::max(0, std::min(static_cast<int>(std::floor(x1)), w - 1));
	int iy1 = std::max(0, std::min(static_cast<int>(std::floor(y1)), h - 1));
	int ix2 = std::max(0, std::min(static_cast<int>(std::ceil(x2)), w - 1));
	int iy2 = std::max(0, std::min(static_cast<int>(std::ceil(y2)), h - 1));
	int rw = std::max(0, ix2 - ix1);
	int rh = std::max(0, iy2 - iy1);
	return cv::Rect(ix1, iy1, rw, rh);
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

	// Nghiep vu: detect bien so tren TUNG crop phuong tien, nhung xu ly da luong
	// o tang ngoai de giam tong latency khi co nhieu xe trong 1 frame.
	ParallelForEach(
		vehicle_crops.size(),
		app_config::kPlateDetectMaxWorkers,
		[&](size_t i) {
			// Moi crop xe chay detect doc lap; output luu theo index xe tuong ung.
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

	// Cong viec nay CPU-bound, chi can so worker vua phai de tranh tao qua nhieu thread.
	ParallelForEach(
		plates_per_vehicle.size(),
		app_config::kPlateDetectMaxWorkers,
		[&](size_t i) {
			const auto& dets = plates_per_vehicle[i];
			auto& out = candidates_per_vehicle[i];

			// Nghiep vu: moi xe toi da 1 bien so.
			// Trong no-NMS mode, dets da duoc sort giam dan theo score,
			// vi vay chi can lay detection hop le dau tien (top-1) roi dung.
			for (const auto& p : dets) {
				if (p.score < min_plate_score) {
					// Cat nguong score som de giam cong viec crop/preprocess OCR.
					continue;
				}
				cv::Rect pr_local = ToRectClamped(
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
		// Flatten ket qua tuong ung thu tu vehicle_index ban dau.
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

	// Preprocess OCR la buoc memory-bound, gioi han worker de giam context switch.
	ParallelForEach(
		candidates.size(),
		app_config::kPlateDetectMaxWorkers,
		[&](size_t i) {
			const auto& c = candidates[i];
			const cv::Mat& vehicle_bgr = vehicle_crops[c.vehicle_index];
			// Crop theo bbox bien so trong crop xe, sau do chuyen thanh RGB uint8 HWC cho OCR.
			cv::Mat plate_bgr = vehicle_bgr(c.plate_rect_in_vehicle);
			plate_rgb_ocr[i] = image_preprocess::PreprocessMatRgbU8Hwc(plate_bgr, out_w, out_h);
		});

	return plate_rgb_ocr;
}

} // namespace plate_parallel
