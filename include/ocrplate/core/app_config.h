/*
 * Mô tả file: Khai báo các hằng số cấu hình trung tâm cho model, ngưỡng và đường dẫn mặc định.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <cstddef>
#include <string>

namespace app_config {

// ===== Đường dẫn model =====
// Model OCR ký tự biển số (nhận crop biển số và trả về chuỗi ký tự).
// Kiểu: đường dẫn tương đối tính từ thư mục chạy binary.
// Tác động: sai đường dẫn -> tạo session ONNX thất bại ngay từ đầu.
inline constexpr const char* kOcrModelPath = "../model/model_ocr.onnx";
// Model detect phương tiện (xe máy/ô tô) cho nhánh tracking.
// Đây là detector đầu vào của pipeline, chất lượng của nó ảnh hưởng trực tiếp đến ID switch.
inline constexpr const char* kVehicleModelPath = "../model/vehicle_int8.onnx";
// Model detect vùng biển số trong từng bbox phương tiện.
// Được gọi sau vehicle detect và trước OCR.
inline constexpr const char* kPlateModelPath = "../model/plate_int8.onnx";
// Model phân loại hãng xe theo crop phương tiện.
// Có thể bỏ qua trong một số profile benchmark, nhưng mặc định vẫn được khởi tạo.
inline constexpr const char* kBrandCarModelPath = "../model/brand_car_classification.onnx";

// Alias tương thích ngược cho code cũ vẫn dùng tên kModelPath.
// Lưu ý: alias này trỏ đến model OCR.
inline constexpr const char* kModelPath = kOcrModelPath;

// ===== Cấu hình input model OCR =====
// Chiều cao input OCR.
// Đơn vị: pixel.
// Giá trị hiện tại khớp với shape model OCR đang sử dụng.
inline constexpr int kInputH = 64;
// Chiều rộng input OCR.
// Đơn vị: pixel.
inline constexpr int kInputW = 128;
// Số kênh màu input OCR (RGB/BGR 3 kênh).
// Lưu ý: tiền xử lý sẽ đảm bảo thứ tự kênh đúng với model.
inline constexpr int kInputC = 3;

// ===== Cấu hình input model brand classifier =====
// Chiều cao input model phân loại hãng xe.
inline constexpr int kBrandInputH = 224;
// Chiều rộng input model phân loại hãng xe.
inline constexpr int kBrandInputW = 224;

// ===== Ngưỡng confidence cơ bản =====
// Ngưỡng score tối thiểu để giữ detection phương tiện.
// Tăng giá trị này -> giảm false positive nhưng dễ bỏ sót xe khó/nhỏ.
// Giảm giá trị này -> nhạy hơn, nhưng có thể tăng churn trong tracking.
inline constexpr float kVehicleConfThresh = 0.6f;
// Ngưỡng score tối thiểu cho detection biển số.
// Thường đặt thấp hơn OCR conf vì plate detector cần ưu tiên recall.
inline constexpr float kPlateConfThresh = 0.5f;
// Ngưỡng trung bình confidence OCR để chấp nhận text cuối cùng.
// Nếu quá thấp dễ giữ nhầm ký tự; nếu quá cao dễ mất biển số mờ/ngược sáng.
inline constexpr float kOcrConfAvgThresh = 0.8f;

// ===== Non-Maximum Suppression =====
// Ngưỡng IoU NMS cho detector YOLO tổng quát.
// IoU > ngưỡng thì box điểm thấp hơn bị loại để tránh trùng lặp.
inline constexpr float kNmsIouThresh = 0.45f;

// Ngưỡng NMS cho nhánh plate detect trong từng xe.
// Đặt = 0 để bỏ qua NMS và lấy top-1 score cao nhất, phù hợp bài toán
// "mỗi phương tiện có tối đa 1 biển số" trong dữ liệu deployment hiện tại.
inline constexpr float kPlateNmsIouThresh = 0.0f;

// Giới hạn số worker cho detect biển số song song.
// Lý do: bên ngoài đã chia task theo xe, nếu mở quá nhiều thread sẽ bị oversubscription
// và giảm hiệu năng thực tế do context switch.
inline constexpr std::size_t kPlateDetectMaxWorkers = 6;

// Bảng ký tự cho CTC decoder OCR.
// Ký tự cuối cùng '_' là token blank CTC, không phải ký tự output thực tế.
inline const std::string kAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

// Đường dẫn ảnh mặc định khi chạy chế độ image mà không truyền --image.
inline constexpr const char* kDefaultImagePath = "../img/1.jpeg";

// ===== Điều độ infer video =====
// Cứ mỗi N frame mới chạy infer 1 lần.
// Các frame ở giữa sẽ dùng tracker prediction + overlay cache.
// N lớn -> FPS cao hơn nhưng dễ mất độ bám tracking khi bị che khuất.
inline constexpr int kVideoInferEveryNFrames = 5;

// ===== Tracking-by-detection (ByteTrack) =====
// Ngưỡng IoU stage 1 (match track với high-score detection).
// Giá trị vừa phải để cân bằng: không quá chặt (bỏ mất match), không quá lỏng (dễ match nhầm).
inline constexpr float kTrackerIouThreshold = 0.25f;
// Số frame tối đa một track confirmed được phép "mất detection" trước khi xoá.
// Tăng giá trị -> giữ ID qua che khuất lâu hơn; quá cao có thể giữ nhầm ghost track.
inline constexpr int kTrackerMaxMissedFrames = 18;
// Số lần hit tối thiểu để track được đánh dấu confirmed.
// Đặt cao hơn để tránh tạo track ảo từ detection nhiễu.
inline constexpr int kTrackerMinConfirmedHits = 3;

// Ngưỡng confidence cho nhóm detection "high" (stage 1).
// Detection >= ngưỡng này có quyền tạo track mới nếu chưa match vào track cũ.
inline constexpr float kTrackerHighScoreThreshold = 0.60f;
// Ngưỡng confidence cho nhóm detection "low" (stage 2 cứu track).
// Detection low-score chỉ để cứu track cũ, KHÔNG tạo track mới.
inline constexpr float kTrackerLowScoreThreshold = 0.20f;
// Ngưỡng IoU cho stage 2 (match với low-score detections).
// Đặt quá thấp dễ gây nhầm ID; đặt quá cao khó cứu track khi box rung.
inline constexpr float kTrackerIouThresholdLow = 0.20f;

// ===== Ngưỡng chấp nhận kết quả theo track =====
// Chỉ chấp nhận brand prediction khi score >= ngưỡng này.
inline constexpr float kTrackBrandAcceptConf = 0.6f;
// Chỉ chấp nhận OCR plate text khi conf trung bình >= ngưỡng này.
inline constexpr float kTrackPlateOcrAcceptConf = 0.8f;
// Số lần dự đoán brand thất bại liên tiếp tối đa trước khi khoá brand cho track.
// Mục đích: tránh tốn compute lặp lại vô ích trên track chất lượng kém.
inline constexpr int kTrackBrandMaxAttempts = 5;
// Số lần OCR biển số thất bại liên tiếp tối đa trước khi gán "unknown".
inline constexpr int kTrackPlateMaxOcrAttempts = 5;
// Nhãn fallback khi OCR thất bại quá ngưỡng attempt.
inline constexpr const char* kTrackPlateUnknownText = "unknown";
// Số lần detect biển số thất bại liên tiếp tối đa trước khi gán "no_plate".
// Sau khi đạt ngưỡng này, hệ thống bỏ qua detect+OCR plate cho track để tiết kiệm tài nguyên.
inline constexpr int kTrackPlateMaxDetectAttempts = 9;
// Nhãn fallback khi xác định xe không có biển số hợp lệ theo logic hiện tại.
inline constexpr const char* kTrackPlateNoPlateText = "no_plate";
// Độ dài tối thiểu biển số hợp lệ sau OCR + hậu xử lý.
inline constexpr int kPlateTextMinLen = 6;
// Độ dài tối đa biển số hợp lệ sau OCR + hậu xử lý.
inline constexpr int kPlateTextMaxLen = 9;

// ===== Hiển thị trace đường đi =====
// Số điểm lịch sử tối đa lưu cho mỗi track để vẽ đường đi.
inline constexpr int kTrackTraceMaxPoints = 20;
// Độ dày nét vẽ trace (pixel).
inline constexpr int kTrackTraceThickness = 2;

// ===== Hiển thị realtime =====
// Giới hạn chiều rộng khung preview khi --show để giảm tải copy/render.
inline constexpr int kVideoPreviewMaxWidth = 960;

// Số frame tối đa trong hàng đợi preview (thread display).
// Giá trị quá nhỏ dễ giật, quá lớn dễ tăng độ trễ hiển thị.
inline constexpr int kVideoDisplayQueueSize = 10;

} // namespace app_config
