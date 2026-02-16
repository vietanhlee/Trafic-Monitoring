#include <filesystem>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <future>
#include <algorithm>
#include <cctype>

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "app_config.h"
#include "brand_classifier.h"
#include "image_preprocess.h"
#include "ocr_batch.h"
#include "utils/cli_args.h"

#include "yolo26_nmsfree.h"

namespace fs = std::filesystem;
static const fs::path kOutputDir = "../out/build/img_out";

namespace {

const char* VehicleClassName(int cls) {
	switch (cls) {
	case 0:
		return "car";
	case 1:
		return "motor";
	default:
		return "unknown";
	}
}

const char* BrandClassName(int brand_id) {
	switch (brand_id) {
	case 0: return "BMW";
	case 1: return "ChengLong";
	case 2: return "DongFeng";
	case 3: return "Ford";
	case 4: return "Hino";
	case 5: return "Honda";
	case 6: return "Howo";
	case 7: return "Hyundai";
	case 8: return "Isuzu";
	case 9: return "Jac";
	case 10: return "Kia";
	case 11: return "KimLong";
	case 12: return "Lexus";
	case 13: return "Mazda";
	case 14: return "Mercedes-Benz";
	case 15: return "Mitsubishi";
	case 16: return "Other";
	case 17: return "Peugeot";
	case 18: return "Samco";
	case 19: return "Suzuki";
	case 20: return "Thaco";
	case 21: return "Toyota";
	case 22: return "VinFast";
	default: return "UnknownBrand";
	}
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

void DrawPlate(cv::Mat& bgr, const yolo26_nmsfree::Detection& det, const std::string& text, float conf_avg) {
	const bool is_low_ocr_conf = (conf_avg < app_config::kOcrConfAvgThresh);
	const cv::Scalar color = is_low_ocr_conf ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
	const cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
	const cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
	cv::rectangle(bgr, p1, p2, color, 2);

	char buf[256];
	std::snprintf(
		buf,
		sizeof(buf),
		"%s p%.2f o%.2f",
		text.empty() ? "?" : text.c_str(),
		det.score,
		conf_avg);
	std::string label = buf;
	if (label.empty()) {
		label = "?";
	}
	int baseline = 0;
	const double font_scale = 0.8;
	const int thickness = 2;
	const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const int x = std::max(0, p1.x);
	const int y = std::max(sz.height + 2, p1.y);
	cv::rectangle(bgr,
		cv::Rect(x, y - sz.height - 2, sz.width + 4, sz.height + baseline + 4),
		color,
		cv::FILLED);
	cv::putText(bgr, label, cv::Point(x + 2, y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), thickness);
}

void DrawVehicle(cv::Mat& bgr, const yolo26_nmsfree::Detection& det, bool has_plate, int brand_id) {
	const cv::Scalar color = has_plate ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
	const cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
	const cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
	cv::rectangle(bgr, p1, p2, color, 2);

	char buf[128];
	if (brand_id >= 0) {
		std::snprintf(buf, sizeof(buf), "%s %s %.2f", VehicleClassName(det.cls), BrandClassName(brand_id), det.score);
	} else {
		std::snprintf(buf, sizeof(buf), "%s %.2f", VehicleClassName(det.cls), det.score);
	}
	const std::string label = buf;
	int baseline = 0;
	const double font_scale = 0.7;
	const int thickness = 2;
	const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const int x = std::max(0, p1.x);
	const int y = std::max(sz.height + 2, p1.y);
	cv::rectangle(bgr,
		cv::Rect(x, y - sz.height - 2, sz.width + 4, sz.height + baseline + 4),
		color,
		cv::FILLED);
	cv::putText(bgr, label, cv::Point(x + 2, y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
}

void DrawFps(cv::Mat& bgr, double fps) {
	char fps_buf[64];
	std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.2f", fps);
	const std::string label = fps_buf;

	int baseline = 0;
	const double font_scale = 0.8;
	const int thickness = 2;
	const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const int margin = 8;
	const int x = margin;
	const int y = margin + sz.height;

	cv::rectangle(
		bgr,
		cv::Rect(x - 4, y - sz.height - 4, sz.width + 8, sz.height + baseline + 8),
		cv::Scalar(0, 0, 0),
		cv::FILLED);
	cv::putText(
		bgr,
		label,
		cv::Point(x, y + baseline / 2),
		cv::FONT_HERSHEY_SIMPLEX,
		font_scale,
		cv::Scalar(0, 255, 255),
		thickness);
}

bool AnnotateFrame(
	cv::Mat& bgr,
	Ort::Env& env,
	Ort::SessionOptions& sess_options,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	const fs::path& brand_model_path,
	bool verbose) {
	auto vehicles_batch = yolo26_nmsfree::RunBatch(vehicle_sess, {bgr}, app_config::kVehicleConfThresh);
	const auto& vehicles = vehicles_batch.at(0);
	if (vehicles.empty()) {
		if (verbose) {
			std::cout << "Khong phat hien phuong tien\n";
		}
		return false;
	}
	if (verbose) {
		for (size_t i = 0; i < vehicles.size(); ++i) {
			std::cout
				<< "vehicle " << i
				<< ": cls=" << vehicles[i].cls << "(" << VehicleClassName(vehicles[i].cls) << ")"
				<< " conf=" << vehicles[i].score
				<< " box=[" << vehicles[i].x1 << "," << vehicles[i].y1 << "," << vehicles[i].x2 << "," << vehicles[i].y2 << "]\n";
		}
	}

	std::vector<cv::Mat> vehicle_crops;
	std::vector<cv::Rect> vehicle_rects;
	std::vector<yolo26_nmsfree::Detection> vehicles_used;
	vehicle_crops.reserve(vehicles.size());
	vehicle_rects.reserve(vehicles.size());
	vehicles_used.reserve(vehicles.size());
	for (const auto& v : vehicles) {
		cv::Rect r = ToRectClamped(v.x1, v.y1, v.x2, v.y2, bgr.cols, bgr.rows);
		if (r.width <= 2 || r.height <= 2) {
			continue;
		}
		vehicle_rects.push_back(r);
		vehicle_crops.push_back(bgr(r).clone());
		vehicles_used.push_back(v);
	}
	if (vehicle_crops.empty()) {
		if (verbose) {
			std::cout << "Khong co crop phuong tien hop le\n";
		}
		return false;
	}

	std::vector<brand_classifier::BrandResult> brand_results(vehicle_crops.size());
	if (vehicle_crops.size() == 1) {
		Ort::Session brand_sess(env, brand_model_path.c_str(), sess_options);
		brand_results[0] = brand_classifier::ClassifySingle(
			brand_sess,
			vehicle_crops[0],
			app_config::kBrandInputH,
			app_config::kBrandInputW);
	} else {
		std::vector<std::future<brand_classifier::BrandResult>> brand_futures;
		brand_futures.reserve(vehicle_crops.size());
		for (size_t i = 0; i < vehicle_crops.size(); ++i) {
			brand_futures.push_back(std::async(
				std::launch::async,
				[&env, &sess_options, &brand_model_path, &vehicle_crops, i]() {
					Ort::Session local_brand_sess(env, brand_model_path.c_str(), sess_options);
					return brand_classifier::ClassifySingle(
						local_brand_sess,
						vehicle_crops[i],
						app_config::kBrandInputH,
						app_config::kBrandInputW);
				}));
		}
		for (size_t i = 0; i < brand_futures.size(); ++i) {
			brand_results[i] = brand_futures[i].get();
		}
	}
	if (verbose) {
		for (size_t i = 0; i < brand_results.size(); ++i) {
			std::cout
				<< "vehicle " << i
				<< ": brand=" << brand_results[i].class_id << "(" << BrandClassName(brand_results[i].class_id) << ")"
				<< " brand_conf=" << brand_results[i].conf << "\n";
		}
	}

	auto plates_per_vehicle = yolo26_nmsfree::RunBatch(plate_sess, vehicle_crops, app_config::kPlateConfThresh);

	std::vector<cv::Mat> plate_rgb_ocr;
	std::vector<yolo26_nmsfree::Detection> plate_boxes_in_image;
	std::vector<bool> vehicle_has_plate(vehicle_rects.size(), false);
	for (size_t i = 0; i < plates_per_vehicle.size(); ++i) {
		const auto& dets = plates_per_vehicle[i];
		const cv::Rect& vr = vehicle_rects[i];
		for (const auto& p : dets) {
			yolo26_nmsfree::Detection in_img = p;
			in_img.x1 += static_cast<float>(vr.x);
			in_img.y1 += static_cast<float>(vr.y);
			in_img.x2 += static_cast<float>(vr.x);
			in_img.y2 += static_cast<float>(vr.y);
			cv::Rect pr = ToRectClamped(in_img.x1, in_img.y1, in_img.x2, in_img.y2, bgr.cols, bgr.rows);
			if (pr.width <= 2 || pr.height <= 2) {
				continue;
			}
			cv::Mat plate_bgr = bgr(pr);
			cv::Mat plate_rgb = image_preprocess::PreprocessMatRgbU8Hwc(plate_bgr, app_config::kInputW, app_config::kInputH);
			plate_rgb_ocr.push_back(plate_rgb);
			plate_boxes_in_image.push_back(in_img);
			vehicle_has_plate[i] = true;
		}
	}
	if (plate_rgb_ocr.empty()) {
		if (verbose) {
			std::cout << "Khong phat hien bien so\n";
		}
		for (size_t i = 0; i < vehicles_used.size(); ++i) {
			DrawVehicle(bgr, vehicles_used[i], false, brand_results[i].class_id);
		}
		return true;
	}

	auto texts = ocr_batch::RunBatch(ocr_sess, plate_rgb_ocr, app_config::kAlphabet);
	if (texts.size() != plate_boxes_in_image.size()) {
		throw std::runtime_error("Loi noi bo: so OCR text != so plate boxes");
	}

	for (size_t i = 0; i < vehicles_used.size(); ++i) {
		DrawVehicle(bgr, vehicles_used[i], vehicle_has_plate[i], brand_results[i].class_id);
	}

	for (size_t i = 0; i < texts.size(); ++i) {
		DrawPlate(bgr, plate_boxes_in_image[i], texts[i].text, texts[i].conf_avg);
		if (verbose) {
			std::cout
				<< "plate " << i
				<< ": text=" << texts[i].text
				<< " plate_conf=" << plate_boxes_in_image[i].score
				<< " ocr_conf_avg=" << texts[i].conf_avg
				<< " box=[" << plate_boxes_in_image[i].x1 << "," << plate_boxes_in_image[i].y1 << "," << plate_boxes_in_image[i].x2 << "," << plate_boxes_in_image[i].y2 << "]\n";
		}
	}

	return true;
}

} // namespace

int ProcessOneImage(
	const fs::path& image_path,
	Ort::Env& env,
	Ort::SessionOptions& sess_options,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	const fs::path& brand_model_path,
	bool show_output) {
	if (!fs::exists(image_path)) {
		std::cerr << "Khong tim thay anh: " << image_path.string() << "\n";
		return 1;
	}

	cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
	if (bgr.empty()) {
		std::cerr << "Khong doc duoc anh: " << image_path.string() << "\n";
		return 1;
	}

	std::cout << "=== Xu ly anh: " << image_path.string() << " ===\n";
	fs::create_directories(kOutputDir);

	if (!AnnotateFrame(bgr, env, sess_options, vehicle_sess, plate_sess, ocr_sess, brand_model_path, true)) {
		return 0;
	}

	const fs::path out_path = kOutputDir / (image_path.stem().string() + "_annotated.jpg");
	if (!cv::imwrite(out_path.string(), bgr)) {
		throw std::runtime_error("Khong ghi duoc anh output: " + out_path.string());
	}
	std::cout << "Da ghi output: " << out_path.string() << "\n";

	if (show_output) {
		const std::string window_name = "OCR Plate - Image Output";
		cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(window_name, 1200, 800);
		cv::imshow(window_name, bgr);
		std::cout << "Nhan phim bat ky de dong cua so hien thi...\n";
		cv::waitKey(0);
		cv::destroyWindow(window_name);
	}
	return 0;
}

int ProcessOneVideo(
	const fs::path& video_path,
	Ort::Env& env,
	Ort::SessionOptions& sess_options,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	const fs::path& brand_model_path,
	bool show_output) {
	if (!fs::exists(video_path)) {
		std::cerr << "Khong tim thay video: " << video_path.string() << "\n";
		return 1;
	}

	cv::VideoCapture cap(video_path.string());
	if (!cap.isOpened()) {
		throw std::runtime_error("Khong mo duoc video: " + video_path.string());
	}

	std::cout << "=== Xu ly video: " << video_path.string() << " ===\n";
	fs::create_directories(kOutputDir);
	const std::string window_name = "OCR Plate - Video Output";
	if (show_output) {
		cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(window_name, 1200, 800);
	}

	cv::Mat frame;
	if (!cap.read(frame) || frame.empty()) {
		throw std::runtime_error("Video khong co frame hop le: " + video_path.string());
	}

	double input_fps = cap.get(cv::CAP_PROP_FPS);
	if (input_fps <= 0.0) {
		input_fps = 30.0;
	}

	const fs::path out_path = kOutputDir / (video_path.stem().string() + "_annotated.mp4");
	cv::VideoWriter writer(
		out_path.string(),
		cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
		input_fps,
		cv::Size(frame.cols, frame.rows));
	if (!writer.isOpened()) {
		throw std::runtime_error("Khong mo duoc video writer: " + out_path.string());
	}

	size_t frame_count = 0;
	while (true) {
		auto t0 = std::chrono::steady_clock::now();
		cv::Mat annotated = frame.clone();
		try {
			AnnotateFrame(annotated, env, sess_options, vehicle_sess, plate_sess, ocr_sess, brand_model_path, false);
		} catch (const std::exception& e) {
			std::cerr << "Canh bao frame " << frame_count << ": " << e.what() << "\n";
		}
		auto t1 = std::chrono::steady_clock::now();
		const double elapsed = std::chrono::duration<double>(t1 - t0).count();
		const double fps = (elapsed > 0.0) ? (1.0 / elapsed) : 0.0;

		DrawFps(annotated, fps);
		writer.write(annotated);
		if (show_output) {
			cv::imshow(window_name, annotated);
		}
		++frame_count;

		if (show_output) {
			const int key = cv::waitKey(1);
			if (key == 27 || key == 'q' || key == 'Q') {
				std::cout << "Dung som theo yeu cau nguoi dung (q/ESC)\n";
				break;
			}
		}

		if (frame_count % 30 == 0) {
			std::cout << "Da xu ly " << frame_count << " frame\n";
		}

		if (!cap.read(frame) || frame.empty()) {
			break;
		}
	}
	if (show_output) {
		cv::destroyWindow(window_name);
	}

	std::cout << "Da ghi output video: " << out_path.string() << " (" << frame_count << " frame)\n";
	return 0;
}

int main(int argc, char** argv) {
	try {
		const fs::path vehicle_model_path = app_config::kVehicleModelPath;
		const fs::path plate_model_path = app_config::kPlateModelPath;
		const fs::path brand_model_path = app_config::kBrandCarModelPath;
		const fs::path ocr_model_path = app_config::kOcrModelPath;
		fs::path image_path;
		fs::path folder_path;
		fs::path video_path;
		bool show_output = false;
		try {
			const auto opt = cli_args::Parse(argc, argv);
			if (opt.show_help) {
				cli_args::PrintUsage(argv[0], std::cout);
				return 0;
			}
			image_path = opt.image_path;
			folder_path = opt.folder_path;
			video_path = opt.video_path;
			show_output = opt.show;
		} catch (const std::exception& e) {
			std::cerr << e.what() << "\n";
			cli_args::PrintUsage(argv[0], std::cout);
			return 2;
		}

		if (image_path.empty() && folder_path.empty() && video_path.empty()) {
			image_path = app_config::kDefaultImagePath;
			std::cout << "(note) Khong truyen --image, dung anh mac dinh: " << image_path.string() << "\n";
		}

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
		if (!folder_path.empty()) {
			if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
				throw std::runtime_error("Thu muc khong hop le: " + folder_path.string());
			}
		}
		if (!video_path.empty()) {
			if (!fs::exists(video_path) || !fs::is_regular_file(video_path)) {
				throw std::runtime_error("Video khong hop le: " + video_path.string());
			}
		}

		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "main");
		Ort::SessionOptions sess_options;
		sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

		Ort::Session vehicle_sess(env, vehicle_model_path.c_str(), sess_options);
		Ort::Session plate_sess(env, plate_model_path.c_str(), sess_options);
		Ort::Session ocr_sess(env, ocr_model_path.c_str(), sess_options);

		if (!folder_path.empty()) {
			std::vector<fs::path> image_paths;
			for (const auto& entry : fs::directory_iterator(folder_path)) {
				if (!entry.is_regular_file()) {
					continue;
				}
				const fs::path p = entry.path();
				std::string ext = p.extension().string();
				std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
					return static_cast<char>(std::tolower(c));
				});
				if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
					image_paths.push_back(p);
				}
			}
			std::sort(image_paths.begin(), image_paths.end());
			if (image_paths.empty()) {
				throw std::runtime_error("Khong tim thay file anh hop le trong thu muc: " + folder_path.string());
			}

			int err_count = 0;
			for (const auto& p : image_paths) {
				try {
					const int rc = ProcessOneImage(p, env, sess_options, vehicle_sess, plate_sess, ocr_sess, brand_model_path, show_output);
					if (rc != 0) {
						++err_count;
					}
				} catch (const std::exception& e) {
					++err_count;
					std::cerr << "Loi khi xu ly anh " << p.string() << ": " << e.what() << "\n";
				}
			}
			std::cout << "Tong ket: da xu ly " << image_paths.size() << " anh, loi " << err_count << " anh\n";
			return (err_count == 0) ? 0 : 1;
		}

		if (!video_path.empty()) {
			return ProcessOneVideo(video_path, env, sess_options, vehicle_sess, plate_sess, ocr_sess, brand_model_path, show_output);
		}

		return ProcessOneImage(image_path, env, sess_options, vehicle_sess, plate_sess, ocr_sess, brand_model_path, show_output);
	} catch (const std::exception& e) {
		std::cerr << "Loi: " << e.what() << "\n";
		return 1;
	}
}