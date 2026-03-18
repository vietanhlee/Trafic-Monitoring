/*
 * Mo ta file: Helper da luong dung chung cho cac cong doan batch.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <cstddef>

namespace parallel_utils {

size_t ResolveWorkerCount(size_t item_count);

} // namespace parallel_utils
