/*
 * Mô tả file: Tiện ích ghi báo cáo OCR và thống kê kết quả nhận diện.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace ocr_report {

struct ConfidenceSummary {
	// Số timestep được đưa vào thống kê (thường bỏ qua blank/pad tùy logic gọi).
	size_t used_timesteps = 0;
	// Giá trị confidence trung bình của các timestep được sử dụng.
	double avg = 0.0;
};

// In báo cáo OCR dạng text cho một kết quả decode.
// Nội dung báo cáo gồm:
// - Chuỗi biển số đã decode.
// - Thống kê confidence trung bình.
// - Confidence theo từng timestep để debug ký tự khó.
// expected_timesteps được dùng để căn báo/trình bày theo độ dài mong đợi.
void PrintResult(
	std::ostream& os,
	const std::string& decoded,
	const std::vector<int64_t>& indices,
	const std::vector<float>& conf,
	const std::string& alphabet,
	size_t expected_timesteps);

// Tổng hợp confidence theo expected_timesteps để phục vụ ngưỡng chấp nhận OCR.
// Trả về ConfidenceSummary gồm số timestep sử dụng và giá trị trung bình.
ConfidenceSummary SummarizeTimesteps(
	const std::vector<float>& conf,
	size_t expected_timesteps);

} // namespace ocr_report
