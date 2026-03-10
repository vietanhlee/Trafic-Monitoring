#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace post_process_out_string {

// Hậu xử lý đơn giản:
// - Map indices -> ký tự theo `alphabet`
// - Chỉ xóa các token blank/pad ở CUỐI chuỗi (thường là '_' lặp lại)
std::string PostprocessIndicesToString(const std::vector<int64_t>& indices,
								const std::string& alphabet,
								int64_t blank_index);

} // namespace post_process_out_string
