# ============================================================================
# CMake tối thiểu để dùng các tính năng hiện tại của dự án.
# Chọn 3.16 để tương thích rộng, nhưng vẫn đủ cho workflow CMake hiện đại.
# ============================================================================
cmake_minimum_required(VERSION 3.16)

# Định danh dự án và ngôn ngữ biên dịch.
# - CXX: mã nguồn C++ chính của pipeline OCR.
# - C  : một số dependency/toolchain có thể cần ngôn ngữ C.
project(ocr_plate LANGUAGES C CXX)

# Chuẩn C++ bắt buộc: không fallback xuống phiên bản thấp hơn.
# Việc này giúp đảm bảo hành vi đồng nhất giữa các máy build.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Xuất compile_commands.json để IDE/LSP tra cứu include path, macro và flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Đường dẫn thư viện ONNX Runtime được vendored trong third_party.
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

# Cấu trúc output thống nhất cho toàn bộ executable của dự án.
# Ví dụ: out/build/bin/main, out/build/bin/benchmark
set(OUT_DIR "${CMAKE_SOURCE_DIR}/out/build")
set(BIN_DIR "${OUT_DIR}/bin")

# OpenCV là dependency bắt buộc cho xử lý ảnh/video và render overlay.
find_package(OpenCV REQUIRED)

# Tạo trước thư mục nhị phân để tránh lỗi khi toolchain ghi output.
file(MAKE_DIRECTORY "${BIN_DIR}")

# Thiết lập output runtime mặc định cho mọi target executable.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}")

# Tìm thư viện onnxruntime theo đường dẫn cố định của dự án.
# NO_DEFAULT_PATH giúp tránh link nhầm bản onnxruntime ở hệ thống.
find_library(ONNXRUNTIME_LIB
	onnxruntime
	PATHS "${ONNXRUNTIME_DIR}/lib"
	NO_DEFAULT_PATH
)

if (NOT ONNXRUNTIME_LIB)
	message(FATAL_ERROR "Không tìm thấy libonnxruntime trong: ${ONNXRUNTIME_DIR}/lib")
endif()

# ---------------------------------------------------------------------------
# Library: ocrplate_utils
# Vai trò: nhóm hàm tiện ích (tiền xử lý ảnh, decode ONNX, song song hóa, báo cáo OCR).
# ---------------------------------------------------------------------------
add_library(ocrplate_utils STATIC
	src/utils/image_preprocess.cpp
	src/utils/onnx_decode_utils.cpp
	src/utils/parallel_utils.cpp
	src/utils/plate_parallel.cpp
	src/utils/ocr_report.cpp
)

target_include_directories(ocrplate_utils PUBLIC
	"${CMAKE_SOURCE_DIR}/include"
)

# SYSTEM PUBLIC:
# - SYSTEM: giảm cảnh báo từ header bên thứ ba.
# - PUBLIC: target link tới library này cũng kế thừa include path.
target_include_directories(ocrplate_utils SYSTEM PUBLIC
	"${ONNXRUNTIME_DIR}/include"
	${OpenCV_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Library: ocrplate_tracking
# Vai trò: theo dõi đối tượng qua nhiều frame + lưu trạng thái định danh.
# ---------------------------------------------------------------------------
add_library(ocrplate_tracking STATIC
	src/tracking/byte_track_tracker.cpp
	src/tracking/vehicle_identity_store.cpp
)

target_include_directories(ocrplate_tracking PUBLIC
	"${CMAKE_SOURCE_DIR}/include"
)

# ---------------------------------------------------------------------------
# Library: ocrplate_services
# Vai trò: lớp service suy luận model (YOLO/OCR/Brand) và hậu xử lý.
# ---------------------------------------------------------------------------
add_library(ocrplate_services STATIC
	src/services/onnx_runner.cpp
	src/services/yolo_detector.cpp
	src/services/yolo_preprocess.cpp
	src/services/yolo_nms.cpp
	src/services/yolo_postprocess.cpp
	src/services/ocr_batch.cpp
	src/services/post_process_out_string.cpp
	src/services/brand_classifier.cpp
)

target_include_directories(ocrplate_services PUBLIC
	"${CMAKE_SOURCE_DIR}/include"
)

target_include_directories(ocrplate_services SYSTEM PUBLIC
	"${ONNXRUNTIME_DIR}/include"
	${OpenCV_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Library: ocrplate_pipeline
# Vai trò: điều phối service + tracking để tạo overlay cuối cùng cho frame.
# ---------------------------------------------------------------------------
add_library(ocrplate_pipeline STATIC
	src/pipeline/frame_annotator.cpp
	src/pipeline/track_trace.cpp
)

target_include_directories(ocrplate_pipeline PUBLIC
	"${CMAKE_SOURCE_DIR}/include"
)

target_include_directories(ocrplate_pipeline SYSTEM PUBLIC
	"${ONNXRUNTIME_DIR}/include"
	${OpenCV_INCLUDE_DIRS}
)

# Pipeline phụ thuộc trực tiếp vào service/utils/tracking.
# Dùng PUBLIC để executable phía trên không cần tự link lại các thành phần này.
target_link_libraries(ocrplate_pipeline PUBLIC
	ocrplate_services
	ocrplate_utils
	ocrplate_tracking
)

# ---------------------------------------------------------------------------
# Executable: main
# Vai trò: ứng dụng CLI chạy pipeline chính.
# ---------------------------------------------------------------------------
add_executable(main
	src/app/main.cpp
	src/app/cli_args.cpp
)

# PRIVATE vì dependency chỉ phục vụ nội bộ target main.
target_link_libraries(main PRIVATE
	ocrplate_pipeline
	"${ONNXRUNTIME_LIB}"
	${OpenCV_LIBS}
	pthread
	dl
)

# ---------------------------------------------------------------------------
# Executable: benchmark
# Vai trò: đo hiệu năng pipeline (FPS/độ trễ).
# ---------------------------------------------------------------------------
add_executable(benchmark
	src/app/benchmark.cpp
)

target_link_libraries(benchmark PRIVATE
	ocrplate_pipeline
	"${ONNXRUNTIME_LIB}"
	${OpenCV_LIBS}
	pthread
	dl
)

# RPATH giúp binary tìm đúng thư viện động onnxruntime tại thời điểm chạy.
# - BUILD_RPATH : khi chạy trực tiếp từ thư mục build.
# - INSTALL_RPATH: khi cài đặt/đóng gói (nếu có bước install).
set_target_properties(main PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
	BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)

set_target_properties(benchmark PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
	BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)
