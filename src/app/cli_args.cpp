/*
 * Mo ta file: Trien khai parse tham so dong lenh va kiem tra hop le dau vao.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/app/cli_args.h"

#include <iostream>
#include <string>

#include "ocrplate/core/app_config.h"

namespace cli_args {

// In chi tiet cach dung cho cac mode: image, folder, video.
void PrintUsage(const char* argv0, std::ostream& os) {
	os
		<< "Cach dung:\n"
		<< "  " << argv0 << " --image <duong_dan_anh.jpg> [--show] [--nosave]\n\n"
		<< "  " << argv0 << " --folder <duong_dan_thu_muc_anh>\n\n"
		<< "  " << argv0 << " --video <duong_dan_video.mp4> [--show] [--nosave]\n\n"
		<< "Ghi chu:\n"
		<< "  - Pipeline: vehicle_detection (YOLO26 NMS-free) -> crop vehicle -> plate_detection (multi-thread per vehicle, no NMS, lay top-1 plate moi xe) -> OCR plate (batch)\n"
		<< "  - Models (fix cung trong include/ocrplate/core/app_config.h):\n"
		<< "      vehicle: " << app_config::kVehicleModelPath << "\n"
		<< "      plate  : " << app_config::kPlateModelPath << "\n"
		<< "      ocr    : " << app_config::kOcrModelPath << "\n"
		<< "  - OCR preprocess: crop bien so -> RGB -> resize (" << app_config::kInputW << "x" << app_config::kInputH << ") -> uint8 NHWC\n"
		<< "  - Output image/folder: ghi anh <ten>_annotated.jpg va in cac bbox + text ra stdout\n"
		<< "  - Output video: ghi video <ten>_annotated.mp4, co overlay FPS goc tren ben trai\n"
		<< "  - --show: hien thi cua so output trong luc xu ly (video: bam q/ESC de thoat som)\n"
		<< "  - --nosave: khong luu file output (chi hop le khi dung voi --image hoac --video)\n";
}

// Parse va validate tham so CLI; nem exception neu tham so khong hop le.
Options Parse(int argc, char** argv) {
	Options opt;
	for (int i = 1; i < argc; ++i) {
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
			throw std::runtime_error("Tham so khong hop le: " + a);
		}
	}

	int mode_count = 0;
	mode_count += opt.image_path.empty() ? 0 : 1;
	mode_count += opt.folder_path.empty() ? 0 : 1;
	mode_count += opt.video_path.empty() ? 0 : 1;
	if (mode_count > 1) {
		throw std::runtime_error("Chi duoc dung mot trong ba tham so: --image hoac --folder hoac --video");
	}
	if (opt.no_save) {
		if (!opt.folder_path.empty()) {
			throw std::runtime_error("--nosave khong ap dung cho --folder");
		}
		if (opt.image_path.empty() && opt.video_path.empty()) {
			throw std::runtime_error("--nosave chi duoc dung kem --image hoac --video");
		}
	}
	if (mode_count == 0) {
		// giu hanh vi cu: cho phep fallback anh mac dinh trong main
	}
	return opt;
}

} // namespace cli_args
