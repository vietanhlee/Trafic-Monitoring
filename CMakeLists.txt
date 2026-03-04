cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
set(BIN_DIR ${CMAKE_SOURCE_DIR}/out/build/bin)

find_package(OpenCV REQUIRED)


file(MAKE_DIRECTORY "${BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}")

add_executable(main
	src/main.cpp
	src/frame_annotator.cpp
	src/image_preprocess.cpp
	src/post_process_out_string.cpp
	src/onnx_runner.cpp
	src/yolo_detector.cpp
	src/yolo_preprocess.cpp
	src/yolo_nms.cpp
	src/yolo_postprocess.cpp
	src/ocr_batch.cpp
	src/brand_classifier.cpp
	src/utils/plate_parallel.cpp
	src/utils/parallel_utils.cpp
	src/utils/cli_args.cpp
	src/utils/ocr_report.cpp
	src/utils/onnx_decode_utils.cpp
)

add_executable(benchmark
	src/benchmark.cpp
	src/image_preprocess.cpp
	src/post_process_out_string.cpp
	src/onnx_runner.cpp
	src/yolo_detector.cpp
	src/yolo_preprocess.cpp
	src/yolo_nms.cpp
	src/yolo_postprocess.cpp
	src/ocr_batch.cpp
	src/brand_classifier.cpp
	src/utils/plate_parallel.cpp
	src/utils/parallel_utils.cpp
	src/utils/ocr_report.cpp
	src/utils/onnx_decode_utils.cpp
)

set_target_properties(main PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)

set_target_properties(benchmark PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)

target_include_directories(main PRIVATE
	"${CMAKE_SOURCE_DIR}/include"
	"${ONNXRUNTIME_DIR}/include"
	"${OpenCV_INCLUDE_DIRS}"
)

target_include_directories(benchmark PRIVATE
	"${CMAKE_SOURCE_DIR}/include"
	"${ONNXRUNTIME_DIR}/include"
	"${OpenCV_INCLUDE_DIRS}"
)

find_library(ONNXRUNTIME_LIB
	onnxruntime
	PATHS "${ONNXRUNTIME_DIR}/lib"
	NO_DEFAULT_PATH
)

if (NOT ONNXRUNTIME_LIB)
	message(FATAL_ERROR "Khong tim thay libonnxruntime trong: ${ONNXRUNTIME_DIR}/lib")
endif()

target_link_libraries(main PRIVATE
	"${ONNXRUNTIME_LIB}"
	${OpenCV_LIBS}
	pthread
	dl
)

target_link_libraries(benchmark PRIVATE
	"${ONNXRUNTIME_LIB}"
	${OpenCV_LIBS}
	pthread
	dl
)

set_target_properties(main PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)

set_target_properties(benchmark PROPERTIES
	BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
	INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)
