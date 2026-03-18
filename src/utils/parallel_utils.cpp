/*
 * Mo ta file: Trien khai helper chia viec va gom ket qua da luong.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/utils/parallel_utils.h"

#include <algorithm>
#include <thread>

namespace parallel_utils {

size_t ResolveWorkerCount(size_t item_count) {
	if (item_count == 0) {
		// Tra ve 1 de caller khong can xu ly truong hop 0 worker.
		return 1;
	}
	const unsigned int hw = std::thread::hardware_concurrency();
	// Neu he thong khong tra ve duoc so core, fallback 4 worker.
	const size_t preferred = (hw == 0) ? 4 : static_cast<size_t>(hw);
	// So worker khong vuot qua so item, tranh tao thread rong.
	return std::max<size_t>(1, std::min(item_count, preferred));
}

} // namespace parallel_utils
