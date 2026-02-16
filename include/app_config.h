#pragma once

#include <string>

namespace app_config {

// Đường dẫn model
inline constexpr const char* kOcrModelPath = "../model/model.onnx";
inline constexpr const char* kVehicleModelPath = "../model/vehicle_detection.onnx";
inline constexpr const char* kPlateModelPath = "../model/plate_detection.onnx";
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

// Ngưỡng confidence (YOLO26 là NMS-free end-to-end, chỉ cần lọc theo score)
inline constexpr float kVehicleConfThresh = 0.4f;
inline constexpr float kPlateConfThresh = 0.4f;
inline constexpr float kOcrConfAvgThresh = 0.60f;

// Ký tự cuối '_' là blank cho CTC.
inline const std::string kAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

// Ảnh mặc định để chạy nếu không truyền --image
inline constexpr const char* kDefaultImagePath = "../img/51V4579.jpg";

} // namespace app_config
