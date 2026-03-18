/*
 * Mo ta file: Khai bao cac hang so cau hinh trung tam cho model, nguong va duong dan mac dinh.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <cstddef>
#include <string>

namespace app_config {

// Đường dẫn model
inline constexpr const char* kOcrModelPath = "../model/model_ocr_plate.onnx";
inline constexpr const char* kVehicleModelPath = "../model/vehicle_int8.onnx";
inline constexpr const char* kPlateModelPath = "../model/plate_int8.onnx";
inline constexpr const char* kBrandCarModelPath = "../model/brand_car_classification.onnx";

// Alias để tương thích ngược (code OCR cũ)
inline constexpr const char* kModelPath = kOcrModelPath;

// Kích thước input: (1, 64, 128, 3) kiểu uint8, layout NHWC
inline constexpr int kInputH = 64;
inline constexpr int kInputW = 128;
inline constexpr int kInputC = 3;

// Kích thước input model phân loại hãng xe
inline constexpr int kBrandInputH = 224;
inline constexpr int kBrandInputW = 224;

// Ngưỡng confidence
inline constexpr float kVehicleConfThresh = 0.6f;
inline constexpr float kPlateConfThresh = 0.6f;
inline constexpr float kOcrConfAvgThresh = 0.8f;

// Ngưỡng IoU cho NMS (dùng cho output YOLO dạng chuẩn)
inline constexpr float kNmsIouThresh = 0.45f;

// Plate branch: moi phuong tien chi co 1 bien so,
// nen tat NMS va chon top-1 detection theo score.
inline constexpr float kPlateNmsIouThresh = 0.0f;

// Plate detect da co chia theo xe bang da luong o tang ngoai,
// nen gioi han worker de tranh oversubscription thread.
inline constexpr std::size_t kPlateDetectMaxWorkers = 6;

// Ký tự cuối '_' là blank cho CTC.
inline const std::string kAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

// Ảnh mặc định để chạy nếu không truyền --image
inline constexpr const char* kDefaultImagePath = "../img/1.jpeg";

// Video: cứ mỗi N frame mới chạy inference 1 lần, các frame còn lại tái dùng overlay gần nhất.
// Profile can bang "on dinh + FPS": 5.
inline constexpr int kVideoInferEveryNFrames = 3;

// Tracking-by-detection (vehicle id)
inline constexpr float kTrackerIouThreshold = 0.35f;
inline constexpr int kTrackerMaxMissedFrames = 20;
inline constexpr int kTrackerMinConfirmedHits = 1;

// Tham số ghép nối theo kiểu ByteTrack.
// - Bước 1: ghép track với detection có score >= kTrackerHighScoreThreshold.
// - Bước 2: ghép các track còn lại với detection có score >= kTrackerLowScoreThreshold,
//          dùng ngưỡng IoU lỏng hơn.
inline constexpr float kTrackerHighScoreThreshold = 0.65f;
inline constexpr float kTrackerLowScoreThreshold = 0.45f;
inline constexpr float kTrackerIouThresholdLow = 0.12f;

// Chấp nhận kết quả nhận diện cho từng track khi conf vượt ngưỡng.
inline constexpr float kTrackBrandAcceptConf = 0.6f;
inline constexpr float kTrackPlateOcrAcceptConf = 0.8f;
// Neu predict brand that bai lien tiep den nguong nay thi khoa brand (khong predict nua).
inline constexpr int kTrackBrandMaxAttempts = 5;
// Neu OCR plate that bai lien tiep den nguong nay thi khoa plate = unknown.
inline constexpr int kTrackPlateMaxOcrAttempts = 5;
inline constexpr const char* kTrackPlateUnknownText = "unknown";
// Neu detect plate that bai lien tiep den nguong nay thi khoa plate = no_plate,
// cac frame sau bo qua detect+ocr plate cho track do.
inline constexpr int kTrackPlateMaxDetectAttempts = 9;
inline constexpr const char* kTrackPlateNoPlateText = "no_plate";
inline constexpr int kPlateTextMinLen = 6;
inline constexpr int kPlateTextMaxLen = 9;

// Ve duong trace cho tung track_id (lich su tam bbox).
inline constexpr int kTrackTraceMaxPoints = 20;
inline constexpr int kTrackTraceThickness = 2;

// Hiển thị video: giới hạn chiều rộng preview để giảm copy/render khi dùng --show
inline constexpr int kVideoPreviewMaxWidth = 960;

// Hàng đợi preview frame cho thread hiển thị
inline constexpr int kVideoDisplayQueueSize = 10;

} // namespace app_config
