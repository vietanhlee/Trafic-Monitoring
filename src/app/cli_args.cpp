/*
 * Mô tả file: Triển khai parse tham số dòng lệnh và kiểm tra hợp lệ đầu vào.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */

#include "ocrplate/app/cli_args.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "ocrplate/core/app_config.h"

namespace cli_args {

// Hàm PrintUsage: In chi tiết cách dùng cho các mode: image, folder, video.
//
// Tham số:
// - argv0: Tên chương trình (thường là argv[0]).
// - os: Output stream để in thông tin.
void PrintUsage(const char* argv0, std::ostream& os) {
	os
		<< "Cách dùng:\n"
		<< "  " << argv0 << " --image <đường_dẫn_ảnh.jpg> [--show] [--nosave]\n\n"
		<< "  " << argv0 << " --folder <đường_dẫn_thư_mục_ảnh>\n\n"
		<< "  " << argv0 << " --video <đường_dẫn_video.mp4> [--show] [--nosave]\n\n"
		<< "Ghi chú:\n"
		<< "  - Pipeline: vehicle_detection (YOLO26 NMS-free) -> crop vehicle -> plate_detection (multi-thread per vehicle, no NMS, lấy top-1 plate mỗi xe) -> OCR plate (batch)\n"
		<< "  - Models (fix cứng trong include/ocrplate/core/app_config.h):\n"
		<< "      vehicle: " << app_config::kVehicleModelPath << "\n"
		<< "      plate  : " << app_config::kPlateModelPath << "\n"
		<< "      ocr    : " << app_config::kOcrModelPath << "\n"
		<< "  - OCR preprocess: crop biển số -> RGB -> resize (" << app_config::kInputW << "x" << app_config::kInputH << ") -> uint8 NHWC\n"
		<< "  - Output image/folder: ghi ảnh <tên>_annotated.jpg và in các bbox + text ra stdout\n"
		<< "  - Output video: ghi video <tên>_annotated.mp4, có overlay FPS góc trên bên trái\n"
		<< "  - --show: hiển thị cửa sổ output trong lúc xử lý (video: bấm q/ESC để thoát sớm)\n"
		<< "  - --nosave: không lưu file output (chỉ hợp lệ khi dùng với --image hoặc --video)\n";
}

// Parse và validate tham số CLI; ném exception nếu tham số không hợp lệ.
Options Parse(int argc, char** argv) {
	Options opt;
	for (int i = 1; i < argc; ++i) {
		// a là token CLI hiện tại đang được phân tích.
		std::string a = argv[i];
		if ((a == "--image" || a == "-i") && i + 1 < argc) {
			opt.image_path = argv[++i];
		} else if ((a == "--folder" || a == "-f") && i + 1 < argc) {
			opt.folder_path = argv[++i];
		} else if ((a == "--video" || a == "-v") && i + 1 < argc) {
			opt.video_path = argv[++i];
		} else if (a == "--show") {
			opt.show = true;
		} else if (a == "--no-show") {
			opt.show = false;
		} else if (a == "--nosave") {
			opt.no_save = true;
		} else if (a == "--help" || a == "-h") {
			opt.show_help = true;
			return opt;
		} else {
			throw std::runtime_error("Tham số không hop le: " + a);
		}
	}

	// mode_count dùng để đảm bảo user chỉ chọn đúng 1 mode chạy.
	int mode_count = 0;
	mode_count += opt.image_path.empty() ? 0 : 1;
	mode_count += opt.folder_path.empty() ? 0 : 1;
	mode_count += opt.video_path.empty() ? 0 : 1;
	if (mode_count > 1) {
		throw std::runtime_error("Chi được dùng mot trong ba tham số: --image hoặc --folder hoặc --video");
	}
	if (opt.no_save) {
		if (!opt.folder_path.empty()) {
			throw std::runtime_error("--nosave không ap dùng cho --folder");
		}
		if (opt.image_path.empty() && opt.video_path.empty()) {
			throw std::runtime_error("--nosave chi được dùng kem --image hoặc --video");
		}
	}
	if (mode_count == 0) {
		// Giữ hành vi cũ: cho phép fallback ảnh mặc định trong main.
	}
	return opt;
}

} // namespace cli_args
