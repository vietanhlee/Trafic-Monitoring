# OCR Plate Pipeline (ONNX Runtime C++ + OpenCV)

Pipeline nhận diện biển số theo chuỗi:

1. Detect phương tiện (YOLO26 NMS-free)
2. Crop phương tiện
3. Phân loại hãng xe trên crop phương tiện
4. Detect biển số trên crop phương tiện (batch)
5. Crop biển số và OCR (batch)
6. Vẽ bbox + nhãn + FPS lên ảnh/video output

## 1) Yêu cầu môi trường

- Linux (khuyến nghị Ubuntu/Debian)
- CMake >= 3.10
- GCC/G++ có hỗ trợ C++23
- OpenCV dev

Cài gói cần thiết:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

ONNX Runtime đã được vendor sẵn trong `third_party/onnxruntime`.

## 2) Build

```bash
rm -rf build
mkdir -p build
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Binary sau build:

- `out/build/bin/main`

## 3) Cách chạy

Chương trình hỗ trợ 3 mode (chỉ dùng 1 mode mỗi lần):

- `--image <path_anh>`
- `--folder <path_thu_muc_anh>`
- `--video <path_video>`

Tuỳ chọn hiển thị:

- `--show`: bật cửa sổ hiển thị output realtime
- `--no-show`: tắt hiển thị (mặc định)

### Ví dụ chạy ảnh

```bash
cd build
../out/build/bin/main --image ../img/test1.jpg
../out/build/bin/main --image ../img/test1.jpg --show
```

### Ví dụ chạy thư mục ảnh

```bash
cd build
../out/build/bin/main --folder ../img
../out/build/bin/main --folder ../img --show
```

### Ví dụ chạy video

```bash
cd build
../out/build/bin/main --video ../video2.mp4
../out/build/bin/main --video ../video2.mp4 --show
```

Ghi chú:

- Nếu không truyền mode nào, chương trình dùng ảnh mặc định `kDefaultImagePath` trong `include/app_config.h`.
- Với `--show`, cửa sổ OpenCV được đặt kích thước **800x600**.
- Khi chạy video với `--show`, bấm `q` hoặc `ESC` để dừng sớm.

## 4) Model và cấu hình chính

Khai báo trong `include/app_config.h`:

- OCR model: `../model/model.onnx`
- Vehicle model: `../model/vehicle_detection.onnx`
- Plate model: `../model/plate_detection.onnx`
- Brand model: `../model/brand_car_classification.onnx`

Thông số chính:

- OCR input: `64x128` (RGB, uint8, NHWC)
- Brand input: `224x224`
- `kVehicleConfThresh = 0.4`
- `kPlateConfThresh = 0.4`
- `kOcrConfAvgThresh = 0.60`

## 5) Output

Output được ghi tại:

- `out/build/img_out`

Tên file output:

- Ảnh: `<ten_anh>_annotated.jpg`
- Video: `<ten_video>_annotated.mp4`

Ngoài file output, chương trình in log xử lý (vehicle/brand/plate/OCR) ra stdout.

## 6) Troubleshooting nhanh

- Lỗi `Tham so khong hop le`: kiểm tra đúng cờ CLI, ví dụ `--show` không đi kèm giá trị.
- Không mở được model/video/image: kiểm tra đường dẫn tương đối, nên chạy từ thư mục `build` như ví dụ.
- Warning ONNX Runtime (optimizer): thường là cảnh báo tối ưu đồ thị, không nhất thiết là lỗi suy luận.