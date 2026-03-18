/*
 * Mo ta file: Trien khai chuan hoa chuoi bien so sau OCR (loc ky tu, dinh dang).
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/services/post_process_out_string.h"

namespace post_process_out_string {

std::string PostprocessIndicesToString(const std::vector<int64_t>& indices,
                                const std::string& alphabet,
                                int64_t blank_index) {
    if (indices.empty() || alphabet.empty()) {
        return std::string();
    }

    std::string out;
    out.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        const int64_t idx = indices[i];
        if (idx < 0 || idx >= static_cast<int64_t>(alphabet.size())) {
            continue;
        }
        out.push_back(alphabet[static_cast<size_t>(idx)]);
    }

    // Chỉ strip các ký tự blank ở CUỐI chuỗi (blank/pad thường nằm cuối và có thể lặp).
    if (blank_index >= 0 && blank_index < static_cast<int64_t>(alphabet.size())) {
        const char blank_char = alphabet[static_cast<size_t>(blank_index)];
        while (!out.empty() && out.back() == blank_char) {
            out.pop_back();
        }
    }

    return out;
}

} // namespace post_process_out_string
