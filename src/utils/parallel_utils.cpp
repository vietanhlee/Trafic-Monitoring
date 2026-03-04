#include "utils/parallel_utils.h"

#include <algorithm>
#include <thread>

namespace parallel_utils {

size_t ResolveWorkerCount(size_t item_count) {
	if (item_count == 0) {
		return 1;
	}
	const unsigned int hw = std::thread::hardware_concurrency();
	const size_t preferred = (hw == 0) ? 4 : static_cast<size_t>(hw);
	return std::max<size_t>(1, std::min(item_count, preferred));
}

} // namespace parallel_utils
