cmake_minimum_required(VERSION 3.16)
project(ocr_plate LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
set(OUT_DIR "${CMAKE_SOURCE_DIR}/out/build")
set(BIN_DIR "${OUT_DIR}/bin")

find_package(OpenCV REQUIRED)

file(MAKE_DIRECTORY "${BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}")

find_library(ONNXRUNTIME_LIB
	onnxruntime
	PATHS "${ONNXRUNTIME_DIR}/lib"
	NO_DEFAULT_PATH
)

if (NOT ONNXRUNTIME_LIB)
	message(FATAL_ERROR "Khong tim thay libonnxruntime trong: ${ONNXRUNTIME_DIR}/lib")
endif()

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

target_include_directories(ocrplate_utils SYSTEM PUBLIC
	"${ONNXRUNTIME_DIR}/include"
	${OpenCV_INCLUDE_DIRS}
)

add_library(ocrplate_tracking STATIC
	src/tracking/byte_track_tracker.cpp
	src/tracking/vehicle_identity_store.cpp
)

target_include_directories(ocrplate_tracking PUBLIC
	"${CMAKE_SOURCE_DIR}/include"
)

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

target_link_libraries(ocrplate_pipeline PUBLIC
	ocrplate_services
	ocrplate_utils
	ocrplate_tracking
)

add_executable(main
	src/app/main.cpp
	src/app/cli_args.cpp
)

target_link_libraries(main PRIVATE
	ocrplate_pipeline
	"${ONNXRUNTIME_LIB}"
	${OpenCV_LIBS}
	pthread
	dl
)

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
