/*
 * Mô tả file: Triển khai helper chia việc và gom kết quả đa luồng.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */

#include "ocrplate/utils/parallel_utils.h"

#include <algorithm>
#include <thread>

namespace parallel_utils {

// Hàm ResolveWorkerCount: Xác định số lượng worker tối ưu để xử lý.
//
// Tham số:
// - item_count: Số lượng công việc cần xử lý.
//
// Trả về:
// - Số lượng worker tối ưu (không vượt quá số lượng công việc).
size_t ResolveWorkerCount(size_t item_count) {
	if (item_count == 0) {
		// Trả về 1 để caller không cần xử lý trường hợp 0 worker.
		return 1;
	}
	const unsigned int hw = std::thread::hardware_concurrency();
	// Nếu hệ thống không trả về được số core, fallback 4 worker.
	const size_t preferred = (hw == 0) ? 4 : static_cast<size_t>(hw);
	// Số worker không vượt quá số item, tránh tạo thread rỗng.
	return std::max<size_t>(1, std::min(item_count, preferred));
}

} // namespace parallel_utils
