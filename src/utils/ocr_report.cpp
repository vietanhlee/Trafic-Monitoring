/*
 * Mô tả file: Triển khai xuất báo cáo OCR theo frame/track để debug và thống kê.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#include "ocrplate/utils/ocr_report.h"

#include <algorithm>
#include <iomanip>
#include <ostream>

namespace ocr_report {

ConfidenceSummary SummarizeTimesteps(const std::vector<float>& conf, size_t expected_timesteps) {
	ConfidenceSummary s;
	// T: số timestep sẽ được đưa vào thống kê (tối đa bằng expected_timesteps).
	const size_t T = std::min(expected_timesteps, conf.size());
	s.used_timesteps = T;
	if (T == 0) {
		s.avg = 0.0;
		return s;
	}
	double sum = 0.0;
	// Cộng confidence trên T timestep đầu để tính trung bình.
	for (size_t t = 0; t < T; ++t) sum += conf[t];
	s.avg = sum / static_cast<double>(T);
	return s;
}

void PrintResult(
	std::ostream& os,
	const std::string& decoded,
	const std::vector<int64_t>& indices,
	const std::vector<float>& conf,
	const std::string& alphabet,
	size_t expected_timesteps) {

	os << "Bịển số: " << decoded << "\n";
	os << std::fixed << std::setprecision(4);

	// s gồm thông tin tổng hợp confidence để in một dòng report ngắn gọn.
	const auto s = SummarizeTimesteps(conf, expected_timesteps);
	os << "Conf(avg_T=" << s.used_timesteps << ", softmax_top1): " << s.avg << "\n";

	// T là số timestep sẽ in chi tiết (có giới hạn để tránh vượt dữ liệu).
	const size_t T = s.used_timesteps;
	if (T > 0 && indices.size() >= T && conf.size() >= T) {
		os << "Timestep conf (9 ký tự): ";
		for (size_t t = 0; t < T; ++t) {
			// idx: class dự đoán tại timestep t, dùng để map ra ký tự.
			const int64_t idx = indices[t];
			char ch = '?';
			if (idx >= 0 && idx < static_cast<int64_t>(alphabet.size())) {
				ch = alphabet[static_cast<size_t>(idx)];
			}
			os << ch << ":" << conf[t];
			if (t + 1 < T) os << " ";
		}
		os << "\n";
	}
}

} // namespace ocr_report
