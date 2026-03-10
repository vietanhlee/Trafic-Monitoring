#include "ocrplate/utils/plate_parallel.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <thread>
#include <utility>

#include "ocrplate/utils/image_preprocess.h"
#include "ocrplate/utils/parallel_utils.h"

namespace {

cv::Rect ToRectClamped(float x1, float y1, float x2, float y2, int w, int h) {
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

	const size_t worker_count = parallel_utils::ResolveWorkerCount(vehicle_crops.size());
	const size_t chunk = (vehicle_crops.size() + worker_count - 1) / worker_count;

	std::vector<std::future<void>> workers;
	workers.reserve(worker_count);
	for (size_t w = 0; w < worker_count; ++w) {
		const size_t begin = w * chunk;
		if (begin >= vehicle_crops.size()) {
			break;
		}
		const size_t end = std::min(vehicle_crops.size(), begin + chunk);
		workers.push_back(std::async(std::launch::async, [&, begin, end]() {
			for (size_t i = begin; i < end; ++i) {
				plates_per_vehicle[i] = yolo_detector::RunSingle(
					plate_sess,
					vehicle_crops[i],
					conf_threshold,
					nms_iou_threshold);
			}
		}));
	}
	for (auto& t : workers) {
		t.get();
	}
	return plates_per_vehicle;
}

std::vector<PlateCandidate> BuildPlateCandidatesParallel(
	const std::vector<std::vector<yolo_detector::Detection>>& plates_per_vehicle,
	const std::vector<cv::Mat>& vehicle_crops,
	float min_plate_score) {
	std::vector<std::vector<PlateCandidate>> candidates_per_vehicle(plates_per_vehicle.size());
	const size_t worker_count = parallel_utils::ResolveWorkerCount(plates_per_vehicle.size());
	const size_t chunk = (plates_per_vehicle.size() + worker_count - 1) / worker_count;

	std::vector<std::future<void>> workers;
	workers.reserve(worker_count);
	for (size_t w = 0; w < worker_count; ++w) {
		const size_t begin = w * chunk;
		if (begin >= plates_per_vehicle.size()) {
			break;
		}
		const size_t end = std::min(plates_per_vehicle.size(), begin + chunk);
		workers.push_back(std::async(std::launch::async, [&, begin, end]() {
			for (size_t i = begin; i < end; ++i) {
				const auto& dets = plates_per_vehicle[i];
				auto& out = candidates_per_vehicle[i];
				out.reserve(dets.size());
				for (const auto& p : dets) {
					if (p.score < min_plate_score) {
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
					PlateCandidate c;
					c.vehicle_index = i;
					c.plate_in_vehicle = p;
					c.plate_rect_in_vehicle = pr_local;
					out.push_back(std::move(c));
				}
			}
		}));
	}
	for (auto& t : workers) {
		t.get();
	}

	std::vector<PlateCandidate> candidates;
	size_t total = 0;
	for (const auto& per_vehicle : candidates_per_vehicle) {
		total += per_vehicle.size();
	}
	candidates.reserve(total);
	for (auto& per_vehicle : candidates_per_vehicle) {
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

	const size_t worker_count = parallel_utils::ResolveWorkerCount(candidates.size());
	const size_t chunk = (candidates.size() + worker_count - 1) / worker_count;

	std::vector<std::future<void>> workers;
	workers.reserve(worker_count);
	for (size_t w = 0; w < worker_count; ++w) {
		const size_t begin = w * chunk;
		if (begin >= candidates.size()) {
			break;
		}
		const size_t end = std::min(candidates.size(), begin + chunk);
		workers.push_back(std::async(std::launch::async, [&, begin, end]() {
			for (size_t i = begin; i < end; ++i) {
				const auto& c = candidates[i];
				const cv::Mat& vehicle_bgr = vehicle_crops[c.vehicle_index];
				cv::Mat plate_bgr = vehicle_bgr(c.plate_rect_in_vehicle);
				plate_rgb_ocr[i] = image_preprocess::PreprocessMatRgbU8Hwc(plate_bgr, out_w, out_h);
			}
		}));
	}
	for (auto& t : workers) {
		t.get();
	}
	return plate_rgb_ocr;
}

} // namespace plate_parallel
