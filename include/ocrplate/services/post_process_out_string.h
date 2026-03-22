/**
 * @file post_process_out_string.h
 * @brief Khai bao ham hau xu ly kết quả OCR index thanh chuoi text.
 *
 * Module nay map index -> ký tự theo alphabet va cat blank token o cuoi chuoi.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace post_process_out_string {

/**
 * @brief Hau xu ly chuoi index OCR thanh chuoi text.
 *
 * Quy tac mac định:
 * - Map moi index sang ký tự tu alphabet.
 * - Cat cac token blank/pad o cuoi chuoi.
 * - Giu nguyen cac ký tự hop le trong than chuoi.
 *
 * @param indices Kết quả argmax theo timestep.
 * @param alphabet Bang ký tự map index -> char.
 * @param blank_index Vi tri token blank CTC trong alphabet.
 * @return std::string Chuoi text da hau xu ly.
 */
std::string PostprocessIndicesToString(const std::vector<int64_t>& indices,
								const std::string& alphabet,
								int64_t blank_index);

} // namespace post_process_out_string
