/*
 * Mô tả file: Helper đa luồng dùng chung cho các công đoạn batch.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <cstddef>

namespace parallel_utils {

// Xác định số worker hợp lý cho bài toán item_count.
// Mục tiêu:
// - item ít  -> tránh tạo quá nhiều thread.
// - item nhiều -> tận dụng CPU nhưng không oversubscription.
// Hàm này được dùng bởi các module song song (plate detect, preprocess, ...).
size_t ResolveWorkerCount(size_t item_count);

} // namespace parallel_utils
