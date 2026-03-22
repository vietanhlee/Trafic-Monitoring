/*
 * Mô tả file: Định nghĩa kiểu dữ liệu detection dùng chung cho pipeline YOLO.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

namespace yolo_detector {

// Cấu trúc detection dùng chung cho toàn bộ pipeline.
// Mục tiêu: giữ kiểu dữ liệu gọn nhẹ để trao đổi nhanh giữa detect/tracking/overlay.
struct Detection {
	// Tọa độ trái (left) của bounding box trong hệ pixel ảnh gốc.
	// Đơn vị: pixel.
	// Ràng buộc mong đợi: x1 <= x2 sau khi decode + clamp.
	float x1 = 0.0f;
	// Tọa độ trên (top) của bounding box.
	// Đơn vị: pixel.
	float y1 = 0.0f;
	// Tọa độ phải (right) của bounding box.
	// Đơn vị: pixel.
	float x2 = 0.0f;
	// Tọa độ dưới (bottom) của bounding box.
	// Đơn vị: pixel.
	float y2 = 0.0f;
	// Độ tin cậy của detection sau khi decode head (và có thể đã qua NMS).
	// Miền giá trị thông thường: [0, 1].
	float score = 0.0f;
	// Nhãn lớp đối tượng (class index).
	// Ví dụ: 0 = car, 1 = motorbike (phụ thuộc model và mapping trong code).
	int cls = -1;
};

} // namespace yolo_detector
