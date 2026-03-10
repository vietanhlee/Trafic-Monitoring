#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace ocr_report {

struct ConfidenceSummary {
	size_t used_timesteps = 0;
	double avg = 0.0;
};

// In ra:
// - Bien so: <decoded>
// - Conf(avg_T=..., softmax_top1): ...
// - Timestep conf (9 ky tu): <ch:conf ...>
void PrintResult(
	std::ostream& os,
	const std::string& decoded,
	const std::vector<int64_t>& indices,
	const std::vector<float>& conf,
	const std::string& alphabet,
	size_t expected_timesteps);

ConfidenceSummary SummarizeTimesteps(
	const std::vector<float>& conf,
	size_t expected_timesteps);

} // namespace ocr_report
