#include <filesystem>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "app_config.h"
#include "frame_annotator.h"
#include "utils/cli_args.h"

namespace fs = std::filesystem;
static const fs::path kOutputDir = "../out/build/img_out";

int ProcessOneImage(
	const fs::path& image_path,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool show_output,
	double* out_infer_ms = nullptr) {
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

	auto t0 = std::chrono::steady_clock::now();
	const bool detected = AnnotateFrame(bgr, vehicle_sess, plate_sess, ocr_sess, brand_sess, true);
	auto t1 = std::chrono::steady_clock::now();
	const double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	const double fps = (infer_ms > 0.0) ? (1000.0 / infer_ms) : 0.0;
	std::printf("Infer time: %.1f ms (%.2f FPS)\n", infer_ms, fps);
	if (out_infer_ms) *out_infer_ms = infer_ms;

	if (!detected) {
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
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
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
	double total_infer_sec = 0.0;
	while (true) {
		auto t0 = std::chrono::steady_clock::now();
		try {
			AnnotateFrame(frame, vehicle_sess, plate_sess, ocr_sess, brand_sess, false);
		} catch (const std::exception& e) {
			std::cerr << "Canh bao frame " << frame_count << ": " << e.what() << "\n";
		}
		auto t1 = std::chrono::steady_clock::now();
		const double elapsed = std::chrono::duration<double>(t1 - t0).count();
		total_infer_sec += elapsed;
		const double fps = (elapsed > 0.0) ? (1.0 / elapsed) : 0.0;

		DrawFps(frame, fps);
		writer.write(frame);
		if (show_output) {
			cv::imshow(window_name, frame);
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
	if (frame_count > 0 && total_infer_sec > 0.0) {
		const double avg_fps = static_cast<double>(frame_count) / total_infer_sec;
		const double avg_ms = (total_infer_sec / static_cast<double>(frame_count)) * 1000.0;
		std::printf("=== FPS trung binh (infer only): %.2f FPS (%.1f ms/frame) ===\n", avg_fps, avg_ms);
	}
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
		sess_options.SetIntraOpNumThreads(4);
		sess_options.SetInterOpNumThreads(1);

		Ort::Session vehicle_sess(env, vehicle_model_path.c_str(), sess_options);
		Ort::Session plate_sess(env, plate_model_path.c_str(), sess_options);
		Ort::Session ocr_sess(env, ocr_model_path.c_str(), sess_options);
		Ort::Session brand_sess(env, brand_model_path.c_str(), sess_options);

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
			double total_infer_ms = 0.0;
			size_t infer_count = 0;
			for (const auto& p : image_paths) {
				try {
					double img_infer_ms = 0.0;
					const int rc = ProcessOneImage(p, vehicle_sess, plate_sess, ocr_sess, brand_sess, show_output, &img_infer_ms);
					total_infer_ms += img_infer_ms;
					++infer_count;
					if (rc != 0) {
						++err_count;
					}
				} catch (const std::exception& e) {
					++err_count;
					std::cerr << "Loi khi xu ly anh " << p.string() << ": " << e.what() << "\n";
				}
			}
			std::cout << "Tong ket: da xu ly " << image_paths.size() << " anh, loi " << err_count << " anh\n";
			if (infer_count > 0 && total_infer_ms > 0.0) {
				const double avg_ms = total_infer_ms / static_cast<double>(infer_count);
				const double avg_fps = 1000.0 / avg_ms;
				std::printf("=== FPS trung binh (infer only): %.2f FPS (%.1f ms/frame, %zu anh) ===\n", avg_fps, avg_ms, infer_count);
			}
			return (err_count == 0) ? 0 : 1;
		}

		if (!video_path.empty()) {
			return ProcessOneVideo(video_path, vehicle_sess, plate_sess, ocr_sess, brand_sess, show_output);
		}

		return ProcessOneImage(image_path, vehicle_sess, plate_sess, ocr_sess, brand_sess, show_output);
	} catch (const std::exception& e) {
		std::cerr << "Loi: " << e.what() << "\n";
		return 1;
	}
}