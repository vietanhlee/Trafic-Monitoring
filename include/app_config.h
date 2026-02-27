#pragma once

#include <string>

namespace app_config {

// Đường dẫn model
inline constexpr const char* kOcrModelPath = "../model/model_ocr_plate.onnx";
inline constexpr const char* kVehicleModelPath = "../model/vehicle_int8.onnx";
inline constexpr const char* kPlateModelPath = "../model/plate_int8.onnx";
inline constexpr const char* kBrandCarModelPath = "../model/brand_car_classification.onnx";

// Alias để tương thích ngược (code OCR cũ)
inline constexpr const char* kModelPath = kOcrModelPath;

// Input shape : (1, 64, 128, 3) kiểu uint8, layout NHWC
inline constexpr int kInputH = 64;
inline constexpr int kInputW = 128;
inline constexpr int kInputC = 3;

// Input model phân loại hãng xe
inline constexpr int kBrandInputH = 224;
inline constexpr int kBrandInputW = 224;

// Ngưỡng confidence
inline constexpr float kVehicleConfThresh = 0.55f;
inline constexpr float kPlateConfThresh = 0.7f;
inline constexpr float kOcrConfAvgThresh = 0.75f;

// NMS IoU threshold (dùng cho standard YOLO output)
inline constexpr float kNmsIouThresh = 0.45f;

// Ký tự cuối '_' là blank cho CTC.
inline const std::string kAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

// Ảnh mặc định để chạy nếu không truyền --image
inline constexpr const char* kDefaultImagePath = "../img/1.jpeg";

// Video: cứ mỗi N frame mới chạy inference 1 lần, các frame còn lại tái dùng overlay gần nhất
inline constexpr int kVideoInferEveryNFrames = 5;

} // namespace app_config
