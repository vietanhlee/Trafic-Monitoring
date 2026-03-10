#include <filesystem>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ocrplate/core/app_config.h"
#include "ocrplate/pipeline/frame_annotator.h"
#include "ocrplate/app/cli_args.h"

namespace fs = std::filesystem;
static const fs::path kOutputDir = "../out/build/img_out";

struct WorkingArea {
	cv::Rect bbox;
	std::vector<cv::Point> polygon_abs;
	std::vector<cv::Point> polygon_local;
	cv::Mat mask_local;
	bool enabled = false;
};

struct PolygonPickerState {
	std::vector<cv::Point> points;
	cv::Point hover{-1, -1};
};

void OnPolygonPickMouse(int event, int x, int y, int /*flags*/, void* userdata) {
	auto* st = static_cast<PolygonPickerState*>(userdata);
	if (st == nullptr) {
		return;
	}
	st->hover = cv::Point(x, y);
	if (event == cv::EVENT_LBUTTONDOWN) {
		st->points.emplace_back(x, y);
	} else if (event == cv::EVENT_RBUTTONDOWN) {
		if (!st->points.empty()) {
			st->points.pop_back();
		}
	}
}

WorkingArea BuildWorkingAreaFromPolygon(const cv::Size& frame_size, const std::vector<cv::Point>& polygon_abs) {
	WorkingArea area;
	if (polygon_abs.size() < 3) {
		area.bbox = cv::Rect(0, 0, frame_size.width, frame_size.height);
		area.polygon_abs = {
			cv::Point(0, 0),
			cv::Point(frame_size.width - 1, 0),
			cv::Point(frame_size.width - 1, frame_size.height - 1),
			cv::Point(0, frame_size.height - 1)};
		area.polygon_local = area.polygon_abs;
		area.mask_local = cv::Mat(frame_size, CV_8UC1, cv::Scalar(255));
		area.enabled = false;
		return area;
	}

	cv::Rect raw = cv::boundingRect(polygon_abs);
	const int x = std::max(0, std::min(raw.x, frame_size.width - 1));
	const int y = std::max(0, std::min(raw.y, frame_size.height - 1));
	const int w = std::max(1, std::min(raw.width, frame_size.width - x));
	const int h = std::max(1, std::min(raw.height, frame_size.height - y));
	area.bbox = cv::Rect(x, y, w, h);
	area.polygon_abs = polygon_abs;
	area.polygon_local.reserve(polygon_abs.size());
	for (const auto& p : polygon_abs) {
		area.polygon_local.emplace_back(p.x - area.bbox.x, p.y - area.bbox.y);
	}
	area.mask_local = cv::Mat(area.bbox.height, area.bbox.width, CV_8UC1, cv::Scalar(0));
	std::vector<std::vector<cv::Point>> polys{area.polygon_local};
	cv::fillPoly(area.mask_local, polys, cv::Scalar(255), cv::LINE_AA);
	area.enabled = true;
	return area;
}

WorkingArea SelectWorkingPolygon(const cv::Mat& first_frame) {
	const std::string win = "Chon vung da giac tracking";
	PolygonPickerState state;
	cv::namedWindow(win, cv::WINDOW_NORMAL);
	cv::resizeWindow(win, 1280, 800);
	cv::setMouseCallback(win, OnPolygonPickMouse, &state);

	while (true) {
		cv::Mat canvas = first_frame.clone();

		if (!state.points.empty()) {
			cv::Mat overlay = canvas.clone();
			std::vector<std::vector<cv::Point>> poly_fill{state.points};
			if (state.points.size() >= 3) {
				cv::fillPoly(overlay, poly_fill, cv::Scalar(40, 180, 70), cv::LINE_AA);
				cv::addWeighted(overlay, 0.22, canvas, 0.78, 0.0, canvas);
			}
			for (size_t i = 1; i < state.points.size(); ++i) {
				cv::line(canvas, state.points[i - 1], state.points[i], cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
			}
			if (state.hover.x >= 0 && state.hover.y >= 0) {
				cv::line(canvas, state.points.back(), state.hover, cv::Scalar(120, 220, 255), 1, cv::LINE_AA);
			}
			for (const auto& p : state.points) {
				cv::circle(canvas, p, 4, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);
				cv::circle(canvas, p, 3, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
			}
		}

		const std::string tip1 = "Click trai: them diem | Click phai/u: xoa diem cuoi | c: xoa het";
		const std::string tip2 = "Enter/Space: xac nhan da giac | Esc: bo qua (toan frame)";
		cv::rectangle(canvas, cv::Rect(12, 12, std::max(780, first_frame.cols / 2), 56), cv::Scalar(0, 0, 0), cv::FILLED);
		cv::putText(canvas, tip1, cv::Point(20, 34), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
		cv::putText(canvas, tip2, cv::Point(20, 58), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(160, 255, 190), 1, cv::LINE_AA);

		cv::imshow(win, canvas);
		const int key = cv::waitKey(16);
		if (key == 27) {
			cv::destroyWindow(win);
			return BuildWorkingAreaFromPolygon(first_frame.size(), {});
		}
		if (key == 'c' || key == 'C') {
			state.points.clear();
			continue;
		}
		if (key == 'u' || key == 'U') {
			if (!state.points.empty()) {
				state.points.pop_back();
			}
			continue;
		}
		if (key == 13 || key == 10 || key == 32) {
			if (state.points.size() >= 3) {
				cv::destroyWindow(win);
				return BuildWorkingAreaFromPolygon(first_frame.size(), state.points);
			}
		}
	}
}

void DrawWorkingAreaOverlay(cv::Mat& frame, const WorkingArea& area) {
	if (!area.enabled || area.polygon_abs.size() < 3) {
		return;
	}
	cv::Mat overlay = frame.clone();
	std::vector<std::vector<cv::Point>> poly{area.polygon_abs};
	cv::fillPoly(overlay, poly, cv::Scalar(50, 190, 80), cv::LINE_AA);
	cv::addWeighted(overlay, 0.10, frame, 0.90, 0.0, frame);

	// Vien ngoai dam + vien trong sang de de nhin tren nen phuc tap.
	cv::polylines(frame, poly, true, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
	cv::polylines(frame, poly, true, cv::Scalar(120, 255, 170), 2, cv::LINE_AA);

	const std::string lbl = "WORK AREA";
	const cv::Point anchor(area.bbox.x + 8, std::max(24, area.bbox.y + 22));
	cv::rectangle(frame, cv::Rect(anchor.x - 6, anchor.y - 18, 120, 24), cv::Scalar(0, 0, 0), cv::FILLED);
	cv::putText(frame, lbl, anchor, cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(160, 255, 190), 1, cv::LINE_AA);
}

int ProcessOneImage(
	const fs::path& image_path,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool save_output,
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
	if (save_output) {
		fs::create_directories(kOutputDir);
	}

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

	if (save_output) {
		const fs::path out_path = kOutputDir / (image_path.stem().string() + "_annotated.jpg");
		if (!cv::imwrite(out_path.string(), bgr)) {
			throw std::runtime_error("Khong ghi duoc anh output: " + out_path.string());
		}
		std::cout << "Da ghi output: " << out_path.string() << "\n";
	} else {
		std::cout << "--nosave: bo qua ghi file output cho anh nay\n";
	}

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
	bool save_output,
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
	if (save_output) {
		fs::create_directories(kOutputDir);
	}
	const std::string window_name = "OCR Plate - Video Output";
	std::atomic<bool> stop_requested{ false };

	std::mutex display_mutex;
	std::condition_variable display_cv;
	std::deque<cv::Mat> display_queue;
	std::thread display_thread;
	const size_t max_display_queue = static_cast<size_t>(std::max(1, app_config::kVideoDisplayQueueSize));

	auto MakeDisplayFrame = [](const cv::Mat& src) {
		const int max_w = std::max(1, app_config::kVideoPreviewMaxWidth);
		if (src.cols <= max_w) {
			return src.clone();
		}
		const double scale = static_cast<double>(max_w) / static_cast<double>(src.cols);
		cv::Mat dst;
		cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_LINEAR);
		return dst;
	};

	cv::Mat frame;
	if (!cap.read(frame) || frame.empty()) {
		throw std::runtime_error("Video khong co frame hop le: " + video_path.string());
	}

	// Chon vung da giac lam viec ngay tu dau, sau do chi infer/draw trong vung nay.
	const WorkingArea work_area = SelectWorkingPolygon(frame);
	std::cout << "Vung xu ly: x=" << work_area.bbox.x
		<< " y=" << work_area.bbox.y
		<< " w=" << work_area.bbox.width
		<< " h=" << work_area.bbox.height
		<< (work_area.enabled ? " (polygon)" : " (toan frame)")
		<< "\n";

	if (show_output) {
		display_thread = std::thread([&]() {
			cv::namedWindow(window_name, cv::WINDOW_NORMAL);
			cv::resizeWindow(window_name, 1200, 800);

			while (true) {
				cv::Mat frame_to_show;
				{
					std::unique_lock<std::mutex> lock(display_mutex);
					display_cv.wait(lock, [&]() {
						return stop_requested.load(std::memory_order_relaxed) || !display_queue.empty();
					});
					if (display_queue.empty()) {
						if (stop_requested.load(std::memory_order_relaxed)) {
							break;
						}
						continue;
					}
					frame_to_show = std::move(display_queue.back());
					display_queue.clear();
				}

				if (!frame_to_show.empty()) {
					cv::imshow(window_name, frame_to_show);
				}

				const int key = cv::waitKey(1);
				if (key == 27 || key == 'q' || key == 'Q') {
					stop_requested.store(true, std::memory_order_relaxed);
				}
			}

			cv::destroyWindow(window_name);
		});
	}

	double input_fps = cap.get(cv::CAP_PROP_FPS);
	if (input_fps <= 0.0) {
		input_fps = 30.0;
	}

	fs::path out_path;
	cv::VideoWriter writer;
	std::mutex writer_mutex;
	std::condition_variable writer_cv;
	std::deque<cv::Mat> writer_queue;
	std::thread writer_thread;
	std::atomic<bool> writer_stop{ false };
	constexpr size_t kMaxWriterQueue = 8;
	size_t dropped_writer_frames = 0;
	if (save_output) {
		out_path = kOutputDir / (video_path.stem().string() + "_annotated.mp4");
		writer.open(
			out_path.string(),
			cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
			input_fps,
			cv::Size(frame.cols, frame.rows));
		if (!writer.isOpened()) {
			throw std::runtime_error("Khong mo duoc video writer: " + out_path.string());
		}

		writer_thread = std::thread([&]() {
			while (true) {
				cv::Mat frame_to_write;
				bool popped = false;
				{
					std::unique_lock<std::mutex> lock(writer_mutex);
					writer_cv.wait(lock, [&]() {
						return writer_stop.load(std::memory_order_relaxed) || !writer_queue.empty();
					});
					if (writer_queue.empty()) {
						if (writer_stop.load(std::memory_order_relaxed)) {
							break;
						}
						continue;
					}
					frame_to_write = std::move(writer_queue.front());
					writer_queue.pop_front();
					popped = true;
				}
				if (popped) {
					writer_cv.notify_one();
				}
				writer.write(frame_to_write);
			}
		});
	}

	size_t frame_count = 0;
	double total_infer_sec = 0.0;
	size_t infer_count = 0;
	double fps_ema = 0.0;
	constexpr double kFpsEmaAlpha = 0.15;
	// Timestamp đầu vòng lặp của lần infer gần nhất (để tính interval giữa 2 infer)
	auto prev_infer_loop_start = std::chrono::steady_clock::now();
	bool has_prev_infer = false;
	FrameOverlayResult cached_overlay;
	bool has_cached_overlay = false;
	TrackingRuntimeContext tracking_ctx;
	const int infer_every_n = std::max(1, app_config::kVideoInferEveryNFrames);
	while (true) {
		if (stop_requested.load(std::memory_order_relaxed)) {
			std::cout << "Dung som theo yeu cau nguoi dung (q/ESC)\n";
			break;
		}

		// Advance tracking prediction every real frame.
		tracking_ctx.tracker.AdvanceFrame();

		auto loop_start = std::chrono::steady_clock::now();
		// FPS hiển thị dùng fps_ema đã tính tại lần infer gần nhất trước frame này
		const double display_fps = fps_ema;

		const bool run_infer = (frame_count % static_cast<size_t>(infer_every_n) == 0);
		if (run_infer) {
			// Cập nhật FPS từ interval giữa 2 lần infer liên tiếp:
			// infer_every_n frame mất bao lâu → fps = infer_every_n / interval
			if (has_prev_infer) {
				const double infer_interval = std::chrono::duration<double>(loop_start - prev_infer_loop_start).count();
				if (infer_interval > 0.0) {
					const double inst_fps = static_cast<double>(infer_every_n) / infer_interval;
					fps_ema = (fps_ema <= 0.0)
						? inst_fps
						: fps_ema * (1.0 - kFpsEmaAlpha) + inst_fps * kFpsEmaAlpha;
				}
			}
			prev_infer_loop_start = loop_start;
			has_prev_infer = true;

			auto infer_t0 = std::chrono::steady_clock::now();
			try {
				cv::Mat work_view = frame(work_area.bbox);
				cv::Mat infer_input;
				if (work_area.enabled) {
					work_view.copyTo(infer_input);
					cv::Mat masked;
					cv::bitwise_and(infer_input, infer_input, masked, work_area.mask_local);
					infer_input = std::move(masked);
				} else {
					infer_input = work_view;
				}
				has_cached_overlay = InferFrameOverlay(
					infer_input,
					vehicle_sess,
					plate_sess,
					ocr_sess,
					brand_sess,
					cached_overlay,
					false,
					&tracking_ctx);
				auto infer_t1 = std::chrono::steady_clock::now();
				total_infer_sec += std::chrono::duration<double>(infer_t1 - infer_t0).count();
				++infer_count;
			} catch (const std::exception& e) {
				std::cerr << "Canh bao frame " << frame_count << ": " << e.what() << "\n";
			}
		}

		if (has_cached_overlay) {
			cv::Mat draw_view = frame(work_area.bbox);
			DrawFrameOverlay(draw_view, cached_overlay);
		}
		DrawWorkingAreaOverlay(frame, work_area);

		DrawFps(frame, display_fps);
		if (save_output) {
			cv::Mat writer_frame = frame.clone();
			{
				std::unique_lock<std::mutex> lock(writer_mutex);
				if (show_output && writer_queue.size() >= kMaxWriterQueue) {
					writer_queue.pop_front();
					++dropped_writer_frames;
				} else {
					while (writer_queue.size() >= kMaxWriterQueue && !stop_requested.load(std::memory_order_relaxed)) {
						writer_cv.wait_for(lock, std::chrono::milliseconds(2));
					}
				}
				writer_queue.emplace_back(std::move(writer_frame));
			}
			writer_cv.notify_one();
		}
		if (show_output) {
			cv::Mat display_frame = MakeDisplayFrame(frame);
			{
				std::lock_guard<std::mutex> lock(display_mutex);
				display_queue.emplace_back(std::move(display_frame));
				while (display_queue.size() > max_display_queue) {
					display_queue.pop_front();
				}
			}
			display_cv.notify_one();
		}
		++frame_count;

		if (frame_count % 30 == 0) {
			std::cout << "Da xu ly " << frame_count << " frame\n";
		}

		if (!cap.read(frame) || frame.empty()) {
			break;
		}
	}
	if (show_output) {
		stop_requested.store(true, std::memory_order_relaxed);
		display_cv.notify_all();
		if (display_thread.joinable()) {
			display_thread.join();
		}
	}
	if (save_output) {
		writer_stop.store(true, std::memory_order_relaxed);
		writer_cv.notify_all();
		if (writer_thread.joinable()) {
			writer_thread.join();
		}
	}

	if (save_output) {
		std::cout << "Da ghi output video: " << out_path.string() << " (" << frame_count << " frame)\n";
		if (dropped_writer_frames > 0) {
			std::cout << "(note) Bo qua " << dropped_writer_frames << " frame khi ghi de giu realtime\n";
		}
	} else {
		std::cout << "--nosave: bo qua ghi file output video\n";
	}

	if (infer_count > 0 && total_infer_sec > 0.0) {
		const double infer_avg_ms = (total_infer_sec / static_cast<double>(infer_count)) * 1000.0;
		const double infer_fps = static_cast<double>(infer_count) / total_infer_sec;
		std::printf("=== Infer trung binh: %.2f FPS (%.1f ms/lan infer, moi %d frame infer 1 lan) ===\n", infer_fps, infer_avg_ms, infer_every_n);
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
		bool save_output = true;
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
			save_output = !opt.no_save;
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
					const int rc = ProcessOneImage(p, vehicle_sess, plate_sess, ocr_sess, brand_sess, true, show_output, &img_infer_ms);
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
			return ProcessOneVideo(video_path, vehicle_sess, plate_sess, ocr_sess, brand_sess, save_output, show_output);
		}

		return ProcessOneImage(image_path, vehicle_sess, plate_sess, ocr_sess, brand_sess, save_output, show_output);
	} catch (const std::exception& e) {
		std::cerr << "Loi: " << e.what() << "\n";
		return 1;
	}
}