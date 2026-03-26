/**
 * @file post_process_out_string.cpp
 * @brief Triển khai hàm hậu xử lý index OCR thành chuỗi text biển số.
 */
#include "ocrplate/services/post_process_out_string.h"

namespace post_process_out_string {

std::string PostprocessIndicesToString(const std::vector<int64_t>& indices,
                                const std::string& alphabet,
                                int64_t blank_index) {
	// Bảo vệ đầu vào: không có dữ liệu thì trả chuỗi rỗng.
    if (indices.empty() || alphabet.empty()) {
        return std::string();
    }

    std::string out;
	// Reserve để tránh cấp phát lại nhiều lần trong vòng lặp append.
    out.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
		// idx là class-id tại timestep i.
        const int64_t idx = indices[i];
        if (idx < 0 || idx >= static_cast<int64_t>(alphabet.size())) {
			// Bỏ qua index lỗi/ngoài miền để tránh crash và giữ output ổn định.
            continue;
        }
		// Map class-id -> ký tự theo bảng alphabet.
        out.push_back(alphabet[static_cast<size_t>(idx)]);
    }

	// Đến đây, out vẫn có thể còn trailing blank/pad do CTC decoder argmax.

    // Chỉ strip các ký tự blank ở CUỐI chuỗi (blank/pad thường nằm cuối và có thể lặp).
    if (blank_index >= 0 && blank_index < static_cast<int64_t>(alphabet.size())) {
		// blank_char là ký tự đại diện token blank CTC trong alphabet.
        const char blank_char = alphabet[static_cast<size_t>(blank_index)];
        while (!out.empty() && out.back() == blank_char) {
			// Cắt trailing blanks để lấy nội dung text thực tế.
            out.pop_back();
        }
    }

    return out;
}

} // namespace post_process_out_string
