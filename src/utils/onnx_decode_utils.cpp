/*
 * Mo ta file: Trien khai helper giai ma tensor output ONNX thanh du lieu co nghia.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/utils/onnx_decode_utils.h"

#include <cmath>

namespace {

template <typename T>
onnx_runner::ArgMaxWithConfResult ArgMaxWithConfImpl(const T* data, int64_t time_dim, int64_t class_dim) {
	onnx_runner::ArgMaxWithConfResult r;
	r.indices.reserve(static_cast<size_t>(time_dim));
	r.conf.reserve(static_cast<size_t>(time_dim));

	for (int64_t t = 0; t < time_dim; ++t) {
		const T* row = data + t * class_dim;
		int64_t best_i = 0;
		T best_v = row[0];
		bool all_in_01 = (row[0] >= static_cast<T>(0)) && (row[0] <= static_cast<T>(1));
		double sum_row = static_cast<double>(row[0]);
		for (int64_t c = 1; c < class_dim; ++c) {
			const T v = row[c];
			if (v > best_v) {
				best_v = v;
				best_i = c;
			}
			all_in_01 = all_in_01 && (v >= static_cast<T>(0)) && (v <= static_cast<T>(1));
			sum_row += static_cast<double>(v);
		}

		float p_best = 0.0f;
		const bool looks_like_probs = all_in_01 && (std::abs(sum_row - 1.0) <= 1e-2);
		if (looks_like_probs) {
			p_best = static_cast<float>(best_v);
		} else {
			const T max_v = best_v;
			double sum_exp = 0.0;
			for (int64_t c = 0; c < class_dim; ++c) {
				sum_exp += std::exp(static_cast<double>(row[c] - max_v));
			}
			if (sum_exp > 0.0) {
				p_best = static_cast<float>(1.0 / sum_exp);
			}
		}

		r.indices.push_back(best_i);
		r.conf.push_back(p_best);
	}

	return r;
}

} // namespace

namespace onnx_decode_utils {

onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const float* data, int64_t time_dim, int64_t class_dim) {
	return ArgMaxWithConfImpl(data, time_dim, class_dim);
}

onnx_runner::ArgMaxWithConfResult ArgMaxWithConf(const double* data, int64_t time_dim, int64_t class_dim) {
	return ArgMaxWithConfImpl(data, time_dim, class_dim);
}

} // namespace onnx_decode_utils
