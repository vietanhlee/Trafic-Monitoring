#include "yolo_detector_internal.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace yolo_detector {

using detail::FillTensorFromRGB_NCHW;
using detail::FillTensorFromRGB_NHWC;
using detail::GetInputSpec;
using detail::LetterboxToSizeRGB;
using detail::ParseOutput;

namespace {

// ── Inference (single batch, no split) ────────────────────────────────

std::vector<std::vector<Detection>> RunBatchNoSplit(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold,
	float nms_iou_threshold) {
	if (bgr_images.empty()) {
		return {};
	}

	const InputSpec spec = GetInputSpec(session);
	const int in_h = static_cast<int>(spec.h);
	const int in_w = static_cast<int>(spec.w);

	std::vector<cv::Mat> rgbs;
	std::vector<LetterboxInfo> infos;
	rgbs.resize(bgr_images.size());
	infos.resize(bgr_images.size());

	const size_t count = bgr_images.size();
	const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
	const size_t worker_count = std::min(count, static_cast<size_t>(hw));
	if (worker_count <= 1) {
		for (size_t i = 0; i < count; ++i) {
			rgbs[i] = LetterboxToSizeRGB(bgr_images[i], in_w, in_h, infos[i]);
		}
	} else {
		std::atomic<size_t> next_index{0};
		std::vector<std::thread> workers;
		workers.reserve(worker_count);
		for (size_t w = 0; w < worker_count; ++w) {
			workers.emplace_back([&]() {
				while (true) {
					const size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
					if (i >= count) {
						break;
					}
					rgbs[i] = LetterboxToSizeRGB(bgr_images[i], in_w, in_h, infos[i]);
				}
			});
		}
		for (auto& t : workers) {
			t.join();
		}
	}

	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	const char* input_name = input_name_alloc.get();

	const size_t output_count = session.GetOutputCount();
	if (output_count == 0) {
		throw std::runtime_error("YOLO model khong co output");
	}
	std::vector<Ort::AllocatedStringPtr> output_name_alloc;
	std::vector<const char*> output_names;
	output_name_alloc.reserve(output_count);
	output_names.reserve(output_count);
	for (size_t i = 0; i < output_count; ++i) {
		output_name_alloc.push_back(session.GetOutputNameAllocated(i, allocator));
		output_names.push_back(output_name_alloc.back().get());
	}

	Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const int64_t batch = static_cast<int64_t>(rgbs.size());

	std::vector<int64_t> input_shape;
	if (spec.nchw) {
		input_shape = {batch, 3, spec.h, spec.w};
	} else {
		input_shape = {batch, spec.h, spec.w, 3};
	}

	std::vector<float> input_f32;
	std::vector<uint8_t> input_u8;
	Ort::Value input_tensor{nullptr};
	if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		if (spec.nchw) {
			FillTensorFromRGB_NCHW(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		} else {
			FillTensorFromRGB_NHWC(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		}
		input_tensor = Ort::Value::CreateTensor<float>(
			mem_info,
			input_f32.data(),
			input_f32.size(),
			input_shape.data(),
			input_shape.size());
	} else if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
		if (spec.nchw) {
			FillTensorFromRGB_NCHW(rgbs, in_h, in_w, input_u8, /*scale_01=*/false);
		} else {
			const size_t per = static_cast<size_t>(in_h) * static_cast<size_t>(in_w) * 3ull;
			input_u8.resize(static_cast<size_t>(batch) * per);
			for (size_t i = 0; i < rgbs.size(); ++i) {
				std::memcpy(input_u8.data() + i * per, rgbs[i].data, per);
			}
		}
		input_tensor = Ort::Value::CreateTensor<uint8_t>(
			mem_info,
			input_u8.data(),
			input_u8.size(),
			input_shape.data(),
			input_shape.size());
	} else {
		throw std::runtime_error("YOLO input type chua ho tro (chi ho tro float32/uint8)");
	}

	const std::vector<const char*> input_names = {input_name};
	auto outputs = session.Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		&input_tensor,
		1,
		output_names.data(),
		output_names.size());

	Ort::Value& out0 = outputs.at(0);
	return ParseOutput(out0, infos, conf_threshold, nms_iou_threshold);
}

} // namespace

// ── Public API ────────────────────────────────────────────────────────

std::vector<std::vector<Detection>> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold,
	float nms_iou_threshold) {
	if (bgr_images.empty()) {
		return {};
	}

	const InputSpec spec = GetInputSpec(session);
	const int64_t fixed_batch = (spec.n > 0) ? spec.n : -1;
	if (fixed_batch > 0 && static_cast<int64_t>(bgr_images.size()) != fixed_batch) {
		std::vector<std::vector<Detection>> all;
		all.reserve(bgr_images.size());
		for (size_t i = 0; i < bgr_images.size(); ) {
			const size_t take = static_cast<size_t>(fixed_batch);
			const size_t end = std::min(bgr_images.size(), i + take);
			if (end - i != take) {
				for (; i < bgr_images.size(); ++i) {
					auto one = RunBatchNoSplit(session, {bgr_images[i]}, conf_threshold, nms_iou_threshold);
					all.push_back(std::move(one.at(0)));
				}
				break;
			}
			std::vector<cv::Mat> chunk;
			chunk.reserve(take);
			for (size_t j = i; j < end; ++j) {
				chunk.push_back(bgr_images[j]);
			}
			auto out = RunBatchNoSplit(session, chunk, conf_threshold, nms_iou_threshold);
			for (auto& v : out) {
				all.push_back(std::move(v));
			}
			i = end;
		}
		return all;
	}

	return RunBatchNoSplit(session, bgr_images, conf_threshold, nms_iou_threshold);
}

} // namespace yolo_detector
