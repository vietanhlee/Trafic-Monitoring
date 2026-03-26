/*
 * Mô tả file: Entrypoint chính: đọc nguồn vào, chạy pipeline và xuất kết quả.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
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
	// BBox bao ngoài polygon ROI, dùng để cắt view nhánh trên frame.
	cv::Rect bbox;
	// Polygon ROI trong hệ tọa độ frame gốc.
	std::vector<cv::Point> polygon_abs;
	// Polygon ROI đã đổi sang hệ tọa độ local của bbox.
	std::vector<cv::Point> polygon_local;
	// Mask nhị phân local (255 trong ROI, 0 ngoài ROI).
	cv::Mat mask_local;
	// true nếu user có chọn polygon hợp lệ, false nếu fallback toàn frame.
	bool enabled = false;
};

struct PolygonPickerState {
	// Danh sách điểm polygon user đã click.
	std::vector<cv::Point> points;
	// Vị trí chuột hiện tại để về preview segment cuối.
	cv::Point hover{-1, -1};
};

struct GateLineSelection {
	// Điểm đầu gate line trong hệ tọa độ local ROI.
	cv::Point p1_local{0, 0};
	// Điểm cuối gate line trong hệ tọa độ local ROI.
	cv::Point p2_local{0, 0};
	// true nếu gate line được chọn hợp lệ.
	bool enabled = false;
};

struct GateLinePickerState {
	// Danh sách điểm click khi user chọn gate line (tối đa 2 điểm).
	std::vector<cv::Point> points;
	// Vị trí hover của chuột để vẽ line preview.
	cv::Point hover{-1, -1};
};

// Xử lý sự kiện chuột khi người dùng vẽ polygon vùng làm việc.
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

// Xử lý sự kiện chuột khi người dùng chọn 2 điểm tạo đường ranh.
void OnGateLinePickMouse(int event, int x, int y, int /*flags*/, void* userdata) {
	auto* st = static_cast<GateLinePickerState*>(userdata);
	if (st == nullptr) {
		return;
	}
	st->hover = cv::Point(x, y);
	if (event != cv::EVENT_LBUTTONDOWN) {
		return;
	}
	if (st->points.size() >= 2) {
		st->points.clear();
	}
	st->points.emplace_back(x, y);
}

// Tạo WorkingArea từ đa giác tùy chọn; fallback toàn frame nếu polygon không hợp lệ.
WorkingArea BuildWorkingAreaFromPolygon(const cv::Size& frame_size, const std::vector<cv::Point>& polygon_abs) {
	WorkingArea area;
	if (polygon_abs.size() < 3) {
		// Polygon không hợp lệ -> coi như toàn bộ frame là vùng xử lý.
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
	// Clamp để đảm bảo bbox nằm trong kích thước frame, tránh truy cập out-of-bound.
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

// Hiện giao diện để người dùng chọn vùng infer theo đa giác.
WorkingArea SelectWorkingPolygon(const cv::Mat& first_frame) {
	const std::string win = "Chon vung da giac tracking";
	PolygonPickerState state;
	cv::namedWindow(win, cv::WINDOW_NORMAL);
	cv::resizeWindow(win, 1280, 800);
	cv::setMouseCallback(win, OnPolygonPickMouse, &state);

	while (true) {
		cv::Mat canvas = first_frame.clone();

		if (!state.points.empty()) {
			// Vẽ preview đa giác đang tạo (fill + cạnh + điểm neo) để người dùng quan sát.
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

		const std::string tip1 = "Click trai: them diem | Click phai/u: xóa diem cuoi | c: xóa het";
		const std::string tip2 = "Enter/Space: xac nhan da giac | Esc: bỏ qua (toan frame)";
		cv::rectangle(canvas, cv::Rect(12, 12, std::max(780, first_frame.cols / 2), 56), cv::Scalar(0, 0, 0), cv::FILLED);
		cv::putText(canvas, tip1, cv::Point(20, 34), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
		cv::putText(canvas, tip2, cv::Point(20, 58), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(160, 255, 190), 1, cv::LINE_AA);

		cv::imshow(win, canvas);
		const int key = cv::waitKey(16);
		// ESC: bỏ qua, C: clear, U: undo, Enter/Space: xác nhận polygon.
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

// Hiện giao diện để người dùng chọn 1 đường ranh (2 điểm) để gate predict.
GateLineSelection SelectGateLine(const cv::Mat& first_frame, const WorkingArea& area) {
	// Tên cửa sổ hiển thị để chọn line.
	const std::string win = "Chon duong ranh trigger predict";
	GateLinePickerState state;

	// Tạo cửa sổ và thiết lập callback chuột.
	cv::namedWindow(win, cv::WINDOW_NORMAL);
	cv::resizeWindow(win, 1280, 800);
	cv::setMouseCallback(win, OnGateLinePickMouse, &state);

	while (true) {
		// Sao chép frame đầu tiên để vẽ các đối tượng lên.
		cv::Mat canvas = first_frame.clone();
		if (area.polygon_abs.size() >= 3) {
			// Vẽ polygon ROI nếu có ít nhất 3 điểm.
			std::vector<std::vector<cv::Point>> poly{area.polygon_abs};
			cv::polylines(canvas, poly, true, cv::Scalar(120, 255, 170), 2, cv::LINE_AA);
		}

		// Vẽ các điểm mà người dùng đã click.
		for (const auto& p : state.points) {
			cv::circle(canvas, p, 4, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);
			cv::circle(canvas, p, 3, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
		}
		if (state.points.size() == 1 && state.hover.x >= 0 && state.hover.y >= 0) {
			// Vẽ đường từ điểm đầu tiên đến vị trí chuột hiện tại.
			cv::line(canvas, state.points[0], state.hover, cv::Scalar(90, 210, 255), 2, cv::LINE_AA);
		}
		if (state.points.size() == 2) {
			// Vẽ đường nối giữa hai điểm đã chọn.
			cv::line(canvas, state.points[0], state.points[1], cv::Scalar(90, 210, 255), 2, cv::LINE_AA);
		}

		// Hiển thị hướng dẫn sử dụng trên canvas.
		const std::string tip1 = "Click trai: chon 2 diem line | c: xóa line";
		const std::string tip2 = "Enter/Space: xac nhan | Esc: bỏ qua (predict nhu cu)";
		cv::rectangle(canvas, cv::Rect(12, 12, std::max(760, first_frame.cols / 2), 56), cv::Scalar(0, 0, 0), cv::FILLED);
		cv::putText(canvas, tip1, cv::Point(20, 34), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
		cv::putText(canvas, tip2, cv::Point(20, 58), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(170, 235, 255), 1, cv::LINE_AA);

		// Hiển thị canvas và xử lý các phím nhấn.
		cv::imshow(win, canvas);
		const int key = cv::waitKey(16);
		if (key == 27) {
			// Nhấn Esc để thoát và không chọn line.
			cv::destroyWindow(win);
			return GateLineSelection{};
		}
		if (key == 'c' || key == 'C') {
			// Nhấn 'c' để xóa các điểm đã chọn.
			state.points.clear();
			continue;
		}
		if (key == 13 || key == 10 || key == 32) {
			// Nhấn Enter hoặc Space để xác nhận line.
			if (state.points.size() == 2 && state.points[0] != state.points[1]) {
				GateLineSelection out;
				out.p1_local = cv::Point(state.points[0].x - area.bbox.x, state.points[0].y - area.bbox.y);
				out.p2_local = cv::Point(state.points[1].x - area.bbox.x, state.points[1].y - area.bbox.y);
				out.enabled = true;
				cv::destroyWindow(win);
				return out;
			}
		}
	}
}

// Vẽ lop phu (overlay) cho vung lam viec để người dùng nhin ro ROI dạng xu ly.
void DrawWorkingAreaOverlay(cv::Mat& frame, const WorkingArea& area) {
	if (!area.enabled || area.polygon_abs.size() < 3) {
		return;
	}
	cv::Mat overlay = frame.clone();
	std::vector<std::vector<cv::Point>> poly{area.polygon_abs};
	cv::fillPoly(overlay, poly, cv::Scalar(50, 190, 80), cv::LINE_AA);
	cv::addWeighted(overlay, 0.10, frame, 0.90, 0.0, frame);

	// Viền ngoài đậm + viền trong sáng để nhìn trên nền phức tạp.
	cv::polylines(frame, poly, true, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
	cv::polylines(frame, poly, true, cv::Scalar(120, 255, 170), 2, cv::LINE_AA);

	const std::string lbl = "WORK AREA";
	const cv::Point anchor(area.bbox.x + 8, std::max(24, area.bbox.y + 22));
	cv::rectangle(frame, cv::Rect(anchor.x - 6, anchor.y - 18, 120, 24), cv::Scalar(0, 0, 0), cv::FILLED);
	cv::putText(frame, lbl, anchor, cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(160, 255, 190), 1, cv::LINE_AA);
}

// Hàm DrawGateLineOverlay: Vẽ overlay cho đường ranh (gate line) trên frame.
//
// Tham số:
// - frame: Frame hiện tại để vẽ overlay.
// - area: Vùng làm việc (WorkingArea) chứa thông tin về ROI.
// - gate_line: Thông tin về đường ranh được chọn (nếu có).
void DrawGateLineOverlay(cv::Mat& frame, const WorkingArea& area, const GateLineSelection& gate_line) {
	if (!gate_line.enabled) {
		// Nếu đường ranh không được kích hoạt, thoát hàm.
		return;
	}

	// Tính toán tọa độ tuyệt đối của hai điểm trên đường ranh.
	const cv::Point p1(gate_line.p1_local.x + area.bbox.x, gate_line.p1_local.y + area.bbox.y);
	const cv::Point p2(gate_line.p2_local.x + area.bbox.x, gate_line.p2_local.y + area.bbox.y);

	// Vẽ đường ranh với hai lớp: lớp ngoài đậm và lớp trong sáng.
	cv::line(frame, p1, p2, cv::Scalar(90, 210, 255), 3, cv::LINE_AA);
	cv::line(frame, p1, p2, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

	// Vẽ nhãn "PREDICT GATE LINE" tại vị trí gần đường ranh.
	const cv::Point anchor(std::max(8, std::min(p1.x, p2.x)), std::max(20, std::min(p1.y, p2.y) - 8));
	cv::rectangle(frame, cv::Rect(anchor.x, anchor.y - 16, 146, 22), cv::Scalar(0, 0, 0), cv::FILLED);
	cv::putText(frame, "PREDICT GATE LINE", cv::Point(anchor.x + 4, anchor.y), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(170, 235, 255), 1, cv::LINE_AA);
}

// Xử lý 1 ảnh đơn: infer, annotate, lưu/hiển thị kết quả và trả về mã trạng thái.
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
		std::cerr << "Không tim thay anh: " << image_path.string() << "\n";
		return 1;
	}

	cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
	if (bgr.empty()) {
		std::cerr << "Không đọc được anh: " << image_path.string() << "\n";
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
			throw std::runtime_error("Không ghi được anh output: " + out_path.string());
		}
		std::cout << "Da ghi output: " << out_path.string() << "\n";
	} else {
		std::cout << "--nosave: bỏ qua ghi file output cho anh nay\n";
	}

	if (show_output) {
		const std::string window_name = "OCR Plate - Image Output";
		cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(window_name, 1200, 800);
		cv::imshow(window_name, bgr);
		std::cout << "Nhan phim bật ky để dong cua so hiện thi...\n";
		cv::waitKey(0);
		cv::destroyWindow(window_name);
	}
	return 0;
}

// Xử lý 1 video: infer theo chu kỳ frame, tracking liên tục và xuất video annotate.
int ProcessOneVideo(
	const fs::path& video_path,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool save_output,
	bool show_output) {
	if (!fs::exists(video_path)) {
		std::cerr << "Không tim thay video: " << video_path.string() << "\n";
		return 1;
	}

	cv::VideoCapture cap(video_path.string());
	if (!cap.isOpened()) {
		throw std::runtime_error("Không mở được video: " + video_path.string());
	}

	std::cout << "=== Xu ly video: " << video_path.string() << " ===\n";
	if (save_output) {
		fs::create_directories(kOutputDir);
	}
	const std::string window_name = "OCR Plate - Video Output";
	std::atomic<bool> stop_requested{ false };

	// Các biến đồng bộ cho luồng hiển thị preview realtime.
	std::mutex display_mutex;
	std::condition_variable display_cv;
	std::deque<cv::Mat> display_queue;
	std::thread display_thread;
	// Giới hạn số frame cho queue preview để tránh tăng độ trễ hiển thị.
	const size_t max_display_queue = static_cast<size_t>(std::max(1, app_config::kVideoDisplayQueueSize));

	auto MakeDisplayFrame = [](const cv::Mat& src) {
		const int max_w = std::max(1, app_config::kVideoPreviewMaxWidth);
		// Chỉ resize khi cần để giảm chi phí copy/scale trên frame nhỏ.
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
		throw std::runtime_error("Video không co frame hop le: " + video_path.string());
	}

	// Chọn vùng đa giác làm việc ngay từ đầu, sau đó chỉ infer/draw trong vùng này.
	const WorkingArea work_area = SelectWorkingPolygon(frame);
	std::cout << "Vung xu ly: x=" << work_area.bbox.x
		<< " y=" << work_area.bbox.y
		<< " w=" << work_area.bbox.width
		<< " h=" << work_area.bbox.height
		<< (work_area.enabled ? " (polygon)" : " (toan frame)")
		<< "\n";
	const GateLineSelection gate_line = SelectGateLine(frame, work_area);
	if (gate_line.enabled) {
		std::cout << "Predict gate line: p1=(" << gate_line.p1_local.x << "," << gate_line.p1_local.y
			<< ") p2=(" << gate_line.p2_local.x << "," << gate_line.p2_local.y << ") [tọa độ trong ROI]\n";
	} else {
		std::cout << "Predict gate line: tắt (xu ly nhu cu)\n";
	}

	if (show_output) {
		display_thread = std::thread([&]() {
			cv::namedWindow(window_name, cv::WINDOW_NORMAL);
			cv::resizeWindow(window_name, 1200, 800);

			while (true) {
				cv::Mat frame_to_show;
				{
					std::unique_lock<std::mutex> lock(display_mutex);
					// Display thread ngủ đến khi có frame mới hoặc có lệnh dừng.
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
					// Chỉ giữ frame mới nhất để ưu tiên realtime thay vì hiện frame cũ.
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
		// Fallback cho video không có metadata FPS hợp lệ.
		input_fps = 30.0;
	}

	// Các biến đồng bộ cho luồng ghi video output.
	fs::path out_path;
	cv::VideoWriter writer;
	std::mutex writer_mutex;
	std::condition_variable writer_cv;
	std::deque<cv::Mat> writer_queue;
	std::thread writer_thread;
	std::atomic<bool> writer_stop{ false };
	// Hàng đợi writer có giới hạn để tránh phình bộ nhớ khi infer chậm.
	constexpr size_t kMaxWriterQueue = 8;
	// Đếm số frame bị bỏ qua khi ưu tiên realtime.
	size_t dropped_writer_frames = 0;
	if (save_output) {
		out_path = kOutputDir / (video_path.stem().string() + "_annotated.mp4");
		writer.open(
			out_path.string(),
			cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
			input_fps,
			cv::Size(frame.cols, frame.rows));
		if (!writer.isOpened()) {
			throw std::runtime_error("Không mở được video writer: " + out_path.string());
		}

		writer_thread = std::thread([&]() {
			while (true) {
				cv::Mat frame_to_write;
				bool popped = false;
				{
					std::unique_lock<std::mutex> lock(writer_mutex);
					// Writer thread ngủ đến khi có frame cần ghi hoặc nhận tín hiệu kết thúc.
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
				// Ghi I/O tách riêng thread để không chặn vòng infer.
				writer.write(frame_to_write);
			}
		});
	}

	size_t frame_count = 0;
	// Tổng thời gian infer thuần (không tính render/IO) để thống kê cuối video.
	double total_infer_sec = 0.0;
	// Số lần infer thực tế đã chạy.
	size_t infer_count = 0;
	// FPS EMA để hiển thị ổn định hơn FPS tức thời.
	double fps_ema = 0.0;
	constexpr double kFpsEmaAlpha = 0.15;
	// Timestamp đầu vòng lặp của lần infer gần nhất (để tính interval giữa 2 infer)
	auto prev_infer_loop_start = std::chrono::steady_clock::now();
	bool has_prev_infer = false;
	FrameOverlayResult cached_overlay;
	// Có overlay hợp lệ để tái sử dụng cho các frame skip infer hay không.
	bool has_cached_overlay = false;
	TrackingRuntimeContext tracking_ctx;
	tracking_ctx.enable_predict_on_line_cross = gate_line.enabled;
	tracking_ctx.gate_line_p1 = gate_line.p1_local;
	tracking_ctx.gate_line_p2 = gate_line.p2_local;
	const int infer_every_n = std::max(1, app_config::kVideoInferEveryNFrames);
	while (true) {
		if (stop_requested.load(std::memory_order_relaxed)) {
			std::cout << "Dùng som theo yeu cau người dùng (q/ESC)\n";
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

			// Mốc bắt đầu infer của frame này (chỉ tính phần infer).
			auto infer_t0 = std::chrono::steady_clock::now();
			try {
				// work_view là view ROI theo bbox, không copy dữ liệu gốc.
				cv::Mat work_view = frame(work_area.bbox);
				// infer_input là ảnh đầu vào thực tế của pipeline (có/không áp mask polygon).
				cv::Mat infer_input;
				if (work_area.enabled) {
					// Nếu có polygon: mask ROI để model chỉ nhìn vùng quan tâm.
					work_view.copyTo(infer_input);
					cv::Mat masked;
					cv::bitwise_and(infer_input, infer_input, masked, work_area.mask_local);
					infer_input = std::move(masked);
				} else {
					// Không có polygon: infer trực tiếp trên ROI toàn frame/bbox.
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
				// Mốc kết thúc infer để cộng dồn thống kê thời gian.
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
		DrawGateLineOverlay(frame, work_area, gate_line);

		DrawFps(frame, display_fps);
		if (save_output) {
			// Clone để writer thread sở hữu bản riêng, tránh data race với frame hiện tại.
			cv::Mat writer_frame = frame.clone();
			{
				std::unique_lock<std::mutex> lock(writer_mutex);
				if (show_output && writer_queue.size() >= kMaxWriterQueue) {
					// Ưu tiên realtime: bỏ frame cũ nếu queue đầy khi đang show.
					writer_queue.pop_front();
					++dropped_writer_frames;
				} else {
					// Chế độ không show: chờ queue rỗng bớt để hạn chế mất frame.
					while (writer_queue.size() >= kMaxWriterQueue && !stop_requested.load(std::memory_order_relaxed)) {
						writer_cv.wait_for(lock, std::chrono::milliseconds(2));
					}
				}
				writer_queue.emplace_back(std::move(writer_frame));
			}
			writer_cv.notify_one();
		}
		if (show_output) {
			// Tạo frame preview đã scale để giảm tải GUI thread.
			cv::Mat display_frame = MakeDisplayFrame(frame);
			{
				std::lock_guard<std::mutex> lock(display_mutex);
				display_queue.emplace_back(std::move(display_frame));
				// Chặn queue display phình to: luôn cắt frame cũ nhất khi vượt ngưỡng.
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
			std::cout << "(note) Bỏ qua " << dropped_writer_frames << " frame khi ghi để giu realtime\n";
		}
	} else {
		std::cout << "--nosave: bỏ qua ghi file output video\n";
	}

	if (infer_count > 0 && total_infer_sec > 0.0) {
		const double infer_avg_ms = (total_infer_sec / static_cast<double>(infer_count)) * 1000.0;
		const double infer_fps = static_cast<double>(infer_count) / total_infer_sec;
		std::printf("=== Infer trung binh: %.2f FPS (%.1f ms/lan infer, moi %d frame infer 1 lan) ===\n", infer_fps, infer_avg_ms, infer_every_n);
	}
	return 0;
}

// Điểm vào chương trình: parse CLI, nạp model, chọn mode image/folder/video và thực thi.
int main(int argc, char** argv) {
	try {
		// Đường dẫn model được lấy tập trung từ app_config.
		const fs::path vehicle_model_path = app_config::kVehicleModelPath;
		const fs::path plate_model_path = app_config::kPlateModelPath;
		const fs::path brand_model_path = app_config::kBrandCarModelPath;
		const fs::path ocr_model_path = app_config::kOcrModelPath;
		// Các biến mode input sẽ được parse từ CLI.
		fs::path image_path;
		fs::path folder_path;
		fs::path video_path;
		// Có lưu output và có hiện preview hay không.
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
			std::cout << "(note) Không truyen --image, dùng anh mac định: " << image_path.string() << "\n";
		}

		if (!fs::exists(vehicle_model_path)) {
			throw std::runtime_error("Không tim thay model vehicle: " + vehicle_model_path.string());
		}
		if (!fs::exists(plate_model_path)) {
			throw std::runtime_error("Không tim thay model plate: " + plate_model_path.string());
		}
		if (!fs::exists(ocr_model_path)) {
			throw std::runtime_error("Không tim thay model ocr: " + ocr_model_path.string());
		}
		if (!fs::exists(brand_model_path)) {
			throw std::runtime_error("Không tim thay model brand car classification: " + brand_model_path.string());
		}
		if (!folder_path.empty()) {
			if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
				throw std::runtime_error("Thư mục không hop le: " + folder_path.string());
			}
		}
		if (!video_path.empty()) {
			if (!fs::exists(video_path) || !fs::is_regular_file(video_path)) {
				throw std::runtime_error("Video không hop le: " + video_path.string());
			}
		}

		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "main");
		// Common options cho vehicle/ocr/brand model: cho phep intra-op cao hon.
		Ort::SessionOptions common_options;
		common_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		common_options.SetIntraOpNumThreads(4);
		common_options.SetInterOpNumThreads(1);

		// Plate model chạy nhiều luong o tang ngoai, nen để intra-op = 1 tranh oversubscription.
		Ort::SessionOptions plate_options;
		plate_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		plate_options.SetIntraOpNumThreads(1);
		plate_options.SetInterOpNumThreads(1);

		Ort::Session vehicle_sess(env, vehicle_model_path.c_str(), common_options);
		Ort::Session plate_sess(env, plate_model_path.c_str(), plate_options);
		Ort::Session ocr_sess(env, ocr_model_path.c_str(), common_options);
		Ort::Session brand_sess(env, brand_model_path.c_str(), common_options);

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
				throw std::runtime_error("Không tim thay file anh hop le trong thư mục: " + folder_path.string());
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