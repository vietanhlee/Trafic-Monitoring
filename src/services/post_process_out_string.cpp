/**
 * @file post_process_out_string.cpp
 * @brief Trien khai ham hau xu ly index OCR thanh chuoi text bịển số.
 */
#include "ocrplate/services/post_process_out_string.h"

namespace post_process_out_string {

/**
 * @brief Map day index OCR thanh chuoi ký tự va cat blank o cuoi.
 *
 * @param indices Day index theo timestep tu model OCR.
 * @param alphabet Bang ký tự map index -> char.
 * @param blank_index Vi tri token blank trong alphabet.
 * @return std::string Chuoi text da hau xu ly.
 */
std::string PostprocessIndicesToString(const std::vector<int64_t>& indices,
                                const std::string& alphabet,
                                int64_t blank_index) {
	// Bao về đầu vào: không co dữ liệu thi tra chuoi rong.
    if (indices.empty() || alphabet.empty()) {
        return std::string();
    }

    std::string out;
	// Reserve để tranh cap phat lai nhiều lan trong vong lap append.
    out.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
		// idx la class-id tai timestep i.
        const int64_t idx = indices[i];
        if (idx < 0 || idx >= static_cast<int64_t>(alphabet.size())) {
			// Bỏ qua index loi/ngoai mien để tranh crash va giu output on định.
            continue;
        }
		// Map class-id -> ký tự theo bang alphabet.
        out.push_back(alphabet[static_cast<size_t>(idx)]);
    }

	// Den day, out van có thể chưa trailing blank/pad do CTC decoder argmax.

    // Chỉ strip các ký tự blank ở CUỐI chuỗi (blank/pad thường nằm cuối và có thể lặp).
    if (blank_index >= 0 && blank_index < static_cast<int64_t>(alphabet.size())) {
		// blank_char la ký tự dai dien token blank CTC trong alphabet.
        const char blank_char = alphabet[static_cast<size_t>(blank_index)];
        while (!out.empty() && out.back() == blank_char) {
			// Cat trailing blanks để lấy noi dùng text thuc te.
            out.pop_back();
        }
    }

    return out;
}

} // namespace post_process_out_string
