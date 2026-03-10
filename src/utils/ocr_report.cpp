#include "ocrplate/utils/ocr_report.h"

#include <algorithm>
#include <iomanip>
#include <ostream>

namespace ocr_report {

ConfidenceSummary SummarizeTimesteps(const std::vector<float>& conf, size_t expected_timesteps) {
	ConfidenceSummary s;
	const size_t T = std::min(expected_timesteps, conf.size());
	s.used_timesteps = T;
	if (T == 0) {
		s.avg = 0.0;
		return s;
	}
	double sum = 0.0;
	for (size_t t = 0; t < T; ++t) sum += conf[t];
	s.avg = sum / static_cast<double>(T);
	return s;
}

void PrintResult(
	std::ostream& os,
	const std::string& decoded,
	const std::vector<int64_t>& indices,
	const std::vector<float>& conf,
	const std::string& alphabet,
	size_t expected_timesteps) {

	os << "Bien so: " << decoded << "\n";
	os << std::fixed << std::setprecision(4);

	const auto s = SummarizeTimesteps(conf, expected_timesteps);
	os << "Conf(avg_T=" << s.used_timesteps << ", softmax_top1): " << s.avg << "\n";

	const size_t T = s.used_timesteps;
	if (T > 0 && indices.size() >= T && conf.size() >= T) {
		os << "Timestep conf (9 ky tu): ";
		for (size_t t = 0; t < T; ++t) {
			const int64_t idx = indices[t];
			char ch = '?';
			if (idx >= 0 && idx < static_cast<int64_t>(alphabet.size())) {
				ch = alphabet[static_cast<size_t>(idx)];
			}
			os << ch << ":" << conf[t];
			if (t + 1 < T) os << " ";
		}
		os << "\n";
	}
}

} // namespace ocr_report
