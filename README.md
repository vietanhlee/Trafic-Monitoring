# OCR Plate (C++ / ONNX Runtime / OpenCV)

Pipeline nhận diện biển số xe với C++:

1. Detect phương tiện (YOLO)
2. Crop phương tiện (mở rộng nhẹ bbox theo trục Y)
3. Chạy 2 nhánh song song:
	- Nhánh A: phân loại brand (API batch, nội bộ infer đa luồng)
	- Nhánh B: detect plate theo batch -> crop plate -> OCR theo batch
4. Vẽ bbox + nhãn lên ảnh/video output

## 1. Yêu cầu

- Linux (khuyến nghị Ubuntu 22.04)
- CMake >= 3.10
- Compiler hỗ trợ C++23
- OpenCV dev

Cài dependency:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

ONNX Runtime đã được vendor sẵn tại `third_party/onnxruntime`.

## 2. Build

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j"$(nproc)" --target main benchmark
```

Binary:

- `out/build/bin/main`
- `out/build/bin/benchmark`

## 3. Chạy `main`

CLI hỗ trợ 3 mode (chọn đúng 1 mode):

- `--image <path_anh>`
- `--folder <path_thu_muc_anh>`
- `--video <path_video>`

Option hiển thị:

- `--show`
- `--no-show` (mặc định)

Ví dụ:

```bash
cd build

# 1 ảnh
../out/build/bin/main --image ../img/1.jpeg

# cả thư mục ảnh
../out/build/bin/main --folder ../img

# video
../out/build/bin/main --video ../video2.mp4 --show
```

Nếu không truyền mode nào, app dùng ảnh mặc định trong `app_config::kDefaultImagePath`.

## 4. Chạy benchmark (1 ảnh)

Benchmark có warm-up trước khi đo và in thời gian theo stage:

- vehicle detect
- brand classify
- plate detect
- plate OCR
- total pipeline

Ví dụ:

```bash
cd build
../out/build/bin/benchmark --image ../img/1.jpeg --warmup 3 --runs 10
```

## 5. Cấu hình model và ngưỡng

Khai báo tại `include/app_config.h`:

- `kVehicleModelPath = ../model/vehicle_int8.onnx`
- `kPlateModelPath = ../model/plate_int8.onnx`
- `kBrandCarModelPath = ../model/brand_car_classification.onnx`
- `kOcrModelPath = ../model/model_ocr_plate.onnx`

Thông số chính hiện tại:

- OCR input: `64x128`, RGB, uint8, NHWC
- Brand input: `224x224`, float32 NCHW
- `kVehicleConfThresh = 0.55`
- `kPlateConfThresh = 0.7`
- `kOcrConfAvgThresh = 0.75`
- `kNmsIouThresh = 0.45`

## 6. Docker

Build image:

```bash
docker build -t ocr-plate .
```

Chạy `main`:

```bash
docker run --rm -v "$PWD/img:/app/img" ocr-plate --image /app/img/1.jpeg
```

Chạy `benchmark`:

```bash
docker run --rm -v "$PWD/img:/app/img" --entrypoint /app/benchmark ocr-plate --image /app/img/1.jpeg --warmup 3 --runs 10
```

## 7. Cấu trúc mã nguồn chính

- `src/main.cpp`: CLI + luồng xử lý image/folder/video
- `src/frame_annotator.cpp`: pipeline annotate frame + overlay
- `src/brand_classifier.cpp`: classify brand (single + batch/multi-thread)
- `src/yolo_detector.cpp`: infer YOLO core
- `src/yolo_preprocess.cpp`: letterbox + tensor packing
- `src/yolo_nms.cpp`: NMS
- `src/yolo_postprocess.cpp`: parse output YOLO
- `src/ocr_batch.cpp`: OCR batch
- `src/benchmark.cpp`: benchmark từng stage

## 8. Lưu ý quan trọng về batch brand

Một số model ONNX có input khai báo dynamic (ví dụ `float32[s77,3,224,224]`) nhưng graph nội bộ vẫn có `Reshape/View` ràng buộc batch=1.

Vì vậy, code hiện tại dùng cơ chế batch API + infer đa luồng theo từng mẫu (`batch=1`) để đảm bảo ổn định runtime, tránh lỗi reshape khi số xe > 1.