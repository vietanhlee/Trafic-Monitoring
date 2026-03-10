#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgcodecs.hpp>

#include "ocrplate/core/app_config.h"
#include "ocrplate/services/brand_classifier.h"
#include "ocrplate/pipeline/frame_annotator.h"
#include "ocrplate/services/ocr_batch.h"
#include "ocrplate/utils/plate_parallel.h"
#include "ocrplate/services/yolo_detector.h"

namespace fs = std::filesystem;

namespace {

struct BenchmarkOptions {
	fs::path image_path = app_config::kDefaultImagePath;
	int warmup_runs = 3;
	int bench_runs = 10;
};

struct StageMetrics {
	double vehicle_detect_ms = 0.0;
	double vehicle_crop_ms = 0.0;
	double brand_branch_ms = 0.0;
	double brand_classify_ms = 0.0;
	double plate_branch_ms = 0.0;
	double plate_detect_ms = 0.0;
	double plate_map_ms = 0.0;
	double plate_crop_preprocess_ms = 0.0;
	double plate_ocr_ms = 0.0;
	double merge_ms = 0.0;
	double total_ms = 0.0;
	size_t vehicles_detected = 0;
	size_t vehicles_used = 0;
	size_t cars_used = 0;
	size_t plates_detected = 0;
};

struct PlateBranchResult {
	std::vector<ocr_batch::OcrText> texts;
	std::vector<yolo_detector::Detection> plate_boxes_in_image;
	std::vector<bool> vehicle_has_plate;
	bool has_any_plate = false;
	double branch_ms = 0.0;
	double plate_detect_ms = 0.0;
	double map_ms = 0.0;
	double crop_preprocess_ms = 0.0;
	double ocr_ms = 0.0;
};

struct BrandBranchResult {
	std::vector<brand_classifier::BrandResult> brand_results;
	double branch_ms = 0.0;
	double classify_ms = 0.0;
};

cv::Rect ToRectClamped(float x1, float y1, float x2, float y2, int w, int h);

void PrintUsage(const char* prog) {
	std::cout
		<< "Usage: " << prog << " [--image <path>] [--warmup <N>] [--runs <N>]\n"
		<< "  --image   : Duong dan 1 anh can benchmark (mac dinh tu app_config::kDefaultImagePath)\n"
		<< "  --warmup  : So lan warm-up truoc khi do (mac dinh 3)\n"
		<< "  --runs    : So lan do benchmark (mac dinh 10)\n";
}

BenchmarkOptions ParseArgs(int argc, char** argv) {
	BenchmarkOptions opt;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h") {
			PrintUsage(argv[0]);
			std::exit(0);
		}
		if (arg == "--image") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Thieu gia tri cho --image");
			}
			opt.image_path = argv[++i];
			continue;
		}
		if (arg == "--warmup") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Thieu gia tri cho --warmup");
			}
			opt.warmup_runs = std::stoi(argv[++i]);
			continue;
		}
		if (arg == "--runs") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Thieu gia tri cho --runs");
			}
			opt.bench_runs = std::stoi(argv[++i]);
			continue;
		}
		throw std::runtime_error("Tham so khong hop le: " + arg);
	}

	if (opt.warmup_runs < 0) {
		throw std::runtime_error("--warmup phai >= 0");
	}
	if (opt.bench_runs <= 0) {
		throw std::runtime_error("--runs phai > 0");
	}
	return opt;
}

cv::Rect ToRectClamped(float x1, float y1, float x2, float y2, int w, int h) {
	int ix1 = std::max(0, std::min(static_cast<int>(std::floor(x1)), w - 1));
	int iy1 = std::max(0, std::min(static_cast<int>(std::floor(y1)), h - 1));
	int ix2 = std::max(0, std::min(static_cast<int>(std::ceil(x2)), w - 1));
	int iy2 = std::max(0, std::min(static_cast<int>(std::ceil(y2)), h - 1));
	int rw = std::max(0, ix2 - ix1);
	int rh = std::max(0, iy2 - iy1);
	return cv::Rect(ix1, iy1, rw, rh);
}

double ElapsedMs(const std::chrono::steady_clock::time_point& t0, const std::chrono::steady_clock::time_point& t1) {
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

StageMetrics RunPipelineOnce(
	const cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess) {
	StageMetrics m;
	const auto total_t0 = std::chrono::steady_clock::now();

	const auto t_vehicle_0 = std::chrono::steady_clock::now();
	auto vehicles_batch = yolo_detector::RunBatch(vehicle_sess, {bgr}, app_config::kVehicleConfThresh, app_config::kNmsIouThresh);
	const auto t_vehicle_1 = std::chrono::steady_clock::now();
	m.vehicle_detect_ms = ElapsedMs(t_vehicle_0, t_vehicle_1);

	const auto& vehicles = vehicles_batch.at(0);
	m.vehicles_detected = vehicles.size();
	if (vehicles.empty()) {
		m.total_ms = ElapsedMs(total_t0, std::chrono::steady_clock::now());
		return m;
	}

	const auto t_crop_0 = std::chrono::steady_clock::now();
	std::vector<cv::Mat> vehicle_crops;
	std::vector<cv::Rect> vehicle_rects;
	std::vector<yolo_detector::Detection> vehicles_used;
	vehicle_crops.reserve(vehicles.size());
	vehicle_rects.reserve(vehicles.size());
	vehicles_used.reserve(vehicles.size());
	for (const auto& v : vehicles) {
		const float box_h = v.y2 - v.y1;
		const float expand_y2 = v.y2 + box_h * 0.05f;
		cv::Rect r = ToRectClamped(v.x1, v.y1, v.x2, expand_y2, bgr.cols, bgr.rows);
		if (r.width <= 2 || r.height <= 2) {
			continue;
		}
		vehicle_rects.push_back(r);
		vehicle_crops.push_back(bgr(r).clone());
		yolo_detector::Detection v_expanded = v;
		v_expanded.y2 = std::min(expand_y2, static_cast<float>(bgr.rows));
		vehicles_used.push_back(v_expanded);
	}
	const auto t_crop_1 = std::chrono::steady_clock::now();
	m.vehicle_crop_ms = ElapsedMs(t_crop_0, t_crop_1);
	m.vehicles_used = vehicle_crops.size();
	if (vehicle_crops.empty()) {
		m.total_ms = ElapsedMs(total_t0, std::chrono::steady_clock::now());
		return m;
	}

	std::vector<size_t> car_indices;
	std::vector<cv::Mat> car_crops;
	car_indices.reserve(vehicles_used.size());
	car_crops.reserve(vehicles_used.size());
	for (size_t i = 0; i < vehicles_used.size(); ++i) {
		if (vehicles_used[i].cls == 0) {
			car_indices.push_back(i);
			car_crops.push_back(vehicle_crops[i]);
		}
	}
	m.cars_used = car_crops.size();

	auto brand_future = std::async(std::launch::async, [&]() {
		BrandBranchResult br;
		const auto t_branch_0 = std::chrono::steady_clock::now();
		if (!car_crops.empty()) {
			const auto t_cls_0 = std::chrono::steady_clock::now();
			br.brand_results = brand_classifier::ClassifyBatch(
				brand_sess,
				car_crops,
				app_config::kBrandInputH,
				app_config::kBrandInputW);
			const auto t_cls_1 = std::chrono::steady_clock::now();
			br.classify_ms = ElapsedMs(t_cls_0, t_cls_1);
		}
		const auto t_branch_1 = std::chrono::steady_clock::now();
		br.branch_ms = ElapsedMs(t_branch_0, t_branch_1);
		return br;
	});

	auto plate_future = std::async(std::launch::async, [&]() {
		PlateBranchResult pr;
		pr.vehicle_has_plate.assign(vehicle_rects.size(), false);
		const auto t_branch_0 = std::chrono::steady_clock::now();

		const auto t_plate_0 = std::chrono::steady_clock::now();
		auto plates_per_vehicle = plate_parallel::DetectPlatesPerVehicleParallel(
			plate_sess,
			vehicle_crops,
			app_config::kPlateConfThresh,
			app_config::kNmsIouThresh);
		const auto t_plate_1 = std::chrono::steady_clock::now();
		pr.plate_detect_ms = ElapsedMs(t_plate_0, t_plate_1);

		std::vector<plate_parallel::PlateCandidate> candidates = plate_parallel::BuildPlateCandidatesParallel(
			plates_per_vehicle,
			vehicle_crops,
			app_config::kPlateConfThresh);

		auto map_future = std::async(std::launch::async, [&]() {
			const auto t_map_0 = std::chrono::steady_clock::now();
			std::vector<yolo_detector::Detection> mapped;
			mapped.reserve(candidates.size());
			std::vector<bool> has_plate(vehicle_rects.size(), false);
			for (const auto& c : candidates) {
				const cv::Rect& vr = vehicle_rects[c.vehicle_index];
				yolo_detector::Detection in_img = c.plate_in_vehicle;
				in_img.x1 += static_cast<float>(vr.x);
				in_img.y1 += static_cast<float>(vr.y);
				in_img.x2 += static_cast<float>(vr.x);
				in_img.y2 += static_cast<float>(vr.y);
				mapped.push_back(std::move(in_img));
				has_plate[c.vehicle_index] = true;
			}
			const auto t_map_1 = std::chrono::steady_clock::now();
			return std::make_tuple(std::move(mapped), std::move(has_plate), ElapsedMs(t_map_0, t_map_1));
		});

		auto crop_future = std::async(std::launch::async, [&]() {
			const auto t_crop_0 = std::chrono::steady_clock::now();
			std::vector<cv::Mat> plate_rgb_ocr = plate_parallel::PreprocessPlatesParallel(candidates, vehicle_crops, app_config::kInputW, app_config::kInputH);
			const auto t_crop_1 = std::chrono::steady_clock::now();
			return std::make_pair(std::move(plate_rgb_ocr), ElapsedMs(t_crop_0, t_crop_1));
		});

		auto mapped_out = map_future.get();
		pr.plate_boxes_in_image = std::move(std::get<0>(mapped_out));
		pr.vehicle_has_plate = std::move(std::get<1>(mapped_out));
		pr.map_ms = std::get<2>(mapped_out);

		auto crop_out = crop_future.get();
		std::vector<cv::Mat> plate_rgb_ocr = std::move(crop_out.first);
		pr.crop_preprocess_ms = crop_out.second;

		if (!plate_rgb_ocr.empty()) {
			const auto t_ocr_0 = std::chrono::steady_clock::now();
			pr.texts = ocr_batch::RunBatch(ocr_sess, plate_rgb_ocr, app_config::kAlphabet);
			const auto t_ocr_1 = std::chrono::steady_clock::now();
			pr.ocr_ms = ElapsedMs(t_ocr_0, t_ocr_1);
			if (pr.texts.size() != pr.plate_boxes_in_image.size()) {
				throw std::runtime_error("Loi noi bo: so OCR text != so plate boxes");
			}
			pr.has_any_plate = true;
		}

		const auto t_branch_1 = std::chrono::steady_clock::now();
		pr.branch_ms = ElapsedMs(t_branch_0, t_branch_1);
		return pr;
	});

	BrandBranchResult brand_result = brand_future.get();
	PlateBranchResult plate_result = plate_future.get();
	m.brand_branch_ms = brand_result.branch_ms;
	m.brand_classify_ms = brand_result.classify_ms;
	m.plate_branch_ms = plate_result.branch_ms;
	m.plate_detect_ms = plate_result.plate_detect_ms;
	m.plate_map_ms = plate_result.map_ms;
	m.plate_crop_preprocess_ms = plate_result.crop_preprocess_ms;
	m.plate_ocr_ms = plate_result.ocr_ms;
	m.plates_detected = plate_result.plate_boxes_in_image.size();

	const auto t_merge_0 = std::chrono::steady_clock::now();
	std::vector<int> brand_id_per_vehicle(vehicles_used.size(), -1);
	for (size_t k = 0; k < car_indices.size(); ++k) {
		brand_id_per_vehicle[car_indices[k]] = brand_result.brand_results[k].class_id;
	}
	std::vector<VehicleOverlayResult> vehicles_overlay;
	std::vector<PlateOverlayResult> plates_overlay;
	vehicles_overlay.reserve(vehicles_used.size());
	plates_overlay.reserve(plate_result.texts.size());

	if (!plate_result.has_any_plate) {
		for (size_t i = 0; i < vehicles_used.size(); ++i) {
			VehicleOverlayResult v;
			v.det = vehicles_used[i];
			v.has_plate = false;
			v.brand_id = brand_id_per_vehicle[i];
			vehicles_overlay.push_back(v);
		}
	} else {
		for (size_t i = 0; i < vehicles_used.size(); ++i) {
			VehicleOverlayResult v;
			v.det = vehicles_used[i];
			v.has_plate = plate_result.vehicle_has_plate[i];
			v.brand_id = brand_id_per_vehicle[i];
			vehicles_overlay.push_back(v);
		}
		for (size_t i = 0; i < plate_result.texts.size(); ++i) {
			PlateOverlayResult p;
			p.det = plate_result.plate_boxes_in_image[i];
			p.text = plate_result.texts[i].text;
			p.conf_avg = plate_result.texts[i].conf_avg;
			plates_overlay.push_back(std::move(p));
		}
	}
	const auto t_merge_1 = std::chrono::steady_clock::now();
	m.merge_ms = ElapsedMs(t_merge_0, t_merge_1);

	m.total_ms = ElapsedMs(total_t0, std::chrono::steady_clock::now());
	return m;
}

void ValidateInputPaths(const BenchmarkOptions& opt) {
	if (!fs::exists(opt.image_path) || !fs::is_regular_file(opt.image_path)) {
		throw std::runtime_error("Anh khong hop le: " + opt.image_path.string());
	}

	const fs::path vehicle_model_path = app_config::kVehicleModelPath;
	const fs::path plate_model_path = app_config::kPlateModelPath;
	const fs::path brand_model_path = app_config::kBrandCarModelPath;
	const fs::path ocr_model_path = app_config::kOcrModelPath;

	if (!fs::exists(vehicle_model_path)) {
		throw std::runtime_error("Khong tim thay model vehicle: " + vehicle_model_path.string());
	}
	if (!fs::exists(plate_model_path)) {
		throw std::runtime_error("Khong tim thay model plate: " + plate_model_path.string());
	}
	if (!fs::exists(ocr_model_path)) {
		throw std::runtime_error("Khong tim thay model ocr: " + ocr_model_path.string());
	}
	if (!fs::exists(brand_model_path)) {
		throw std::runtime_error("Khong tim thay model brand car classification: " + brand_model_path.string());
	}
}

} // namespace

int main(int argc, char** argv) {
	try {
		const BenchmarkOptions opt = ParseArgs(argc, argv);
		ValidateInputPaths(opt);

		cv::Mat bgr = cv::imread(opt.image_path.string(), cv::IMREAD_COLOR);
		if (bgr.empty()) {
			throw std::runtime_error("Khong doc duoc anh: " + opt.image_path.string());
		}

		std::cout << "=== Benchmark 1 anh ===\n";
		std::cout << "Image   : " << opt.image_path.string() << "\n";
		std::cout << "Warm-up : " << opt.warmup_runs << "\n";
		std::cout << "Runs    : " << opt.bench_runs << "\n";

		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark");
		Ort::SessionOptions sess_options;
		sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		sess_options.SetIntraOpNumThreads(4);
		sess_options.SetInterOpNumThreads(1);

		Ort::Session vehicle_sess(env, app_config::kVehicleModelPath, sess_options);
		Ort::Session plate_sess(env, app_config::kPlateModelPath, sess_options);
		Ort::Session ocr_sess(env, app_config::kOcrModelPath, sess_options);
		Ort::Session brand_sess(env, app_config::kBrandCarModelPath, sess_options);

		for (int i = 0; i < opt.warmup_runs; ++i) {
			(void)RunPipelineOnce(bgr, vehicle_sess, plate_sess, ocr_sess, brand_sess);
		}

		std::vector<StageMetrics> runs;
		runs.reserve(static_cast<size_t>(opt.bench_runs));
		for (int i = 0; i < opt.bench_runs; ++i) {
			runs.push_back(RunPipelineOnce(bgr, vehicle_sess, plate_sess, ocr_sess, brand_sess));
		}

		double sum_vehicle = 0.0;
		double sum_vehicle_crop = 0.0;
		double sum_brand_branch = 0.0;
		double sum_brand = 0.0;
		double sum_plate_branch = 0.0;
		double sum_plate = 0.0;
		double sum_plate_map = 0.0;
		double sum_plate_crop = 0.0;
		double sum_ocr = 0.0;
		double sum_merge = 0.0;
		double sum_total = 0.0;
		double sum_vehicles = 0.0;
		double sum_cars = 0.0;
		double sum_plates = 0.0;

		std::cout << "\n=== Per-run (ms) ===\n";
		for (size_t i = 0; i < runs.size(); ++i) {
			const auto& r = runs[i];
			sum_vehicle += r.vehicle_detect_ms;
			sum_vehicle_crop += r.vehicle_crop_ms;
			sum_brand_branch += r.brand_branch_ms;
			sum_brand += r.brand_classify_ms;
			sum_plate_branch += r.plate_branch_ms;
			sum_plate += r.plate_detect_ms;
			sum_plate_map += r.plate_map_ms;
			sum_plate_crop += r.plate_crop_preprocess_ms;
			sum_ocr += r.plate_ocr_ms;
			sum_merge += r.merge_ms;
			sum_total += r.total_ms;
			sum_vehicles += static_cast<double>(r.vehicles_used);
			sum_cars += static_cast<double>(r.cars_used);
			sum_plates += static_cast<double>(r.plates_detected);

			std::cout
				<< "run " << (i + 1)
				<< " | vehicle=" << r.vehicle_detect_ms
				<< " | v_crop=" << r.vehicle_crop_ms
				<< " | brand_branch=" << r.brand_branch_ms
				<< " | brand=" << r.brand_classify_ms
				<< " | plate_branch=" << r.plate_branch_ms
				<< " | plate=" << r.plate_detect_ms
				<< " | p_map=" << r.plate_map_ms
				<< " | p_crop=" << r.plate_crop_preprocess_ms
				<< " | ocr=" << r.plate_ocr_ms
				<< " | merge=" << r.merge_ms
				<< " | total=" << r.total_ms
				<< " | vehicles=" << r.vehicles_used
				<< " | cars=" << r.cars_used
				<< " | plates=" << r.plates_detected
				<< "\n";
		}

		const double denom = static_cast<double>(runs.size());
		const double avg_vehicle = sum_vehicle / denom;
		const double avg_vehicle_crop = sum_vehicle_crop / denom;
		const double avg_brand_branch = sum_brand_branch / denom;
		const double avg_brand = sum_brand / denom;
		const double avg_plate_branch = sum_plate_branch / denom;
		const double avg_plate = sum_plate / denom;
		const double avg_plate_map = sum_plate_map / denom;
		const double avg_plate_crop = sum_plate_crop / denom;
		const double avg_ocr = sum_ocr / denom;
		const double avg_merge = sum_merge / denom;
		const double avg_total = sum_total / denom;

		std::cout << "\n=== Average (ms) ===\n";
		std::cout << "vehicle detect : " << avg_vehicle << "\n";
		std::cout << "vehicle crop   : " << avg_vehicle_crop << "\n";
		std::cout << "brand branch   : " << avg_brand_branch << "\n";
		std::cout << "brand classify : " << avg_brand << "\n";
		std::cout << "plate branch   : " << avg_plate_branch << "\n";
		std::cout << "plate detect   : " << avg_plate << "\n";
		std::cout << "plate map      : " << avg_plate_map << "\n";
		std::cout << "plate crop/pre : " << avg_plate_crop << "\n";
		std::cout << "plate ocr      : " << avg_ocr << "\n";
		std::cout << "merge          : " << avg_merge << "\n";
		std::cout << "total pipeline : " << avg_total << "\n";
		if (avg_total > 0.0) {
			std::cout << "avg fps (infer only): " << (1000.0 / avg_total) << "\n";
		}
		std::cout << "avg vehicles used  : " << (sum_vehicles / denom) << "\n";
		std::cout << "avg cars used      : " << (sum_cars / denom) << "\n";
		std::cout << "avg plates detected: " << (sum_plates / denom) << "\n";

		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Loi: " << e.what() << "\n";
		return 1;
	}
}