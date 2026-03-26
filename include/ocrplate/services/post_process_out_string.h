/**
 * @file post_process_out_string.h
 * @brief Khai báo hàm hậu xử lý kết quả OCR index thành chuỗi text.
 *
 * Module này map index -> ký tự theo alphabet và cắt blank token ở cuối chuỗi.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace post_process_out_string {

/**
 * @brief Hậu xử lý chuỗi index OCR thành chuỗi text.
 *
 * Quy tắc mặc định:
 * - Map mỗi index sang ký tự từ alphabet.
 * - Cắt các token blank/pad ở cuối chuỗi.
 * - Giữ nguyên các ký tự hợp lệ trong thân chuỗi.
 *
 * @param indices Kết quả argmax theo timestep.
 * @param alphabet Bảng ký tự map index -> char.
 * @param blank_index Vị trí token blank CTC trong alphabet.
 * @return std::string Chuỗi text đã hậu xử lý.
 */
std::string PostprocessIndicesToString(const std::vector<int64_t>& indices,
								const std::string& alphabet,
								int64_t blank_index);

} // namespace post_process_out_string
