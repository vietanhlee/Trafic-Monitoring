/*
 * Mo ta file: Trien khai infer YOLO cho batch, xu ly fixed-batch va dynamic-batch.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/services/yolo_detector_internal.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace yolo_detector {

using detail::FillTensorFromRGB_NCHW;
using detail::FillTensorFromRGB_NHWC;
using detail::GetInputSpec;
using detail::LetterboxToSizeRGB;
using detail::ParseOutput;

namespace {

// Metadata I/O cua session duoc cache de tranh goi
// GetInputNameAllocated/GetOutputNameAllocated lap lai o moi lan infer.
struct SessionIoNames {
	std::string input_name;
	std::vector<std::string> output_name_storage;
	std::vector<const char*> output_names;
};

// Lay/cached ten input-output cua session de giam chi phi truy van metadata.
std::shared_ptr<const SessionIoNames> GetSessionIoNames(Ort::Session& session) {
	static std::mutex cache_mu;
	static std::unordered_map<std::uintptr_t, std::shared_ptr<const SessionIoNames>> cache;

	const auto key = reinterpret_cast<std::uintptr_t>(&session);
	{
		std::lock_guard<std::mutex> lk(cache_mu);
		auto it = cache.find(key);
		if (it != cache.end()) {
			return it->second;
		}
	}

	// Build metadata khi cache miss.
	Ort::AllocatorWithDefaultOptions allocator;
	auto built = std::make_shared<SessionIoNames>();
	auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	built->input_name = input_name_alloc.get();

	const size_t output_count = session.GetOutputCount();
	if (output_count == 0) {
		throw std::runtime_error("YOLO model khong co output");
	}
	built->output_name_storage.reserve(output_count);
	built->output_names.reserve(output_count);
	for (size_t i = 0; i < output_count; ++i) {
		auto out_name = session.GetOutputNameAllocated(i, allocator);
		built->output_name_storage.emplace_back(out_name.get());
	}
	for (const auto& s : built->output_name_storage) {
		built->output_names.push_back(s.c_str());
	}

	{
		std::lock_guard<std::mutex> lk(cache_mu);
		auto [it, inserted] = cache.emplace(key, built);
		if (!inserted) {
			return it->second;
		}
	}
	return built;
}

// ── Inference (single batch, no split) ────────────────────────────────

// Chay infer 1 batch lien tuc (khong tach chunk), gom preprocess + postprocess.
std::vector<std::vector<Detection>> RunBatchNoSplit(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold,
	float nms_iou_threshold) {
	if (bgr_images.empty()) {
		return {};
	}

	const InputSpec spec = GetInputSpec(session);
	const auto io_names = GetSessionIoNames(session);
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
		// Batch nho: chay tuan tu de tranh overhead tao thread.
		for (size_t i = 0; i < count; ++i) {
			rgbs[i] = LetterboxToSizeRGB(bgr_images[i], in_w, in_h, infos[i]);
		}
	} else {
		// Batch lon: chia preprocessing theo chi so atomics cho nhieu worker.
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
		// Model float32: co the scale [0,1] truoc khi tao tensor input.
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
		// Model uint8: giu nguyen dynamic range pixel, copy theo layout model can.
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

	const std::vector<const char*> input_names = {io_names->input_name.c_str()};
	// Chay ONNX session va parse output dau tien theo format YOLO.
	auto outputs = session.Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		&input_tensor,
		1,
		io_names->output_names.data(),
		io_names->output_names.size());

	Ort::Value& out0 = outputs.at(0);
	return ParseOutput(out0, infos, conf_threshold, nms_iou_threshold);
}

} // namespace

// ── Public API ────────────────────────────────────────────────────────

// API detect batch: tu xu ly truong hop model fixed-batch bang cach tach chunk.
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
	// Neu model bi fixed batch (vd n=1), can chia nho input theo dung batch model.
	// Muc tieu: giu hanh vi dung cho ca dynamic-batch va static-batch model.
	if (fixed_batch > 0 && static_cast<int64_t>(bgr_images.size()) != fixed_batch) {
		std::vector<std::vector<Detection>> all;
		all.reserve(bgr_images.size());
		for (size_t i = 0; i < bgr_images.size(); ) {
			const size_t take = static_cast<size_t>(fixed_batch);
			const size_t end = std::min(bgr_images.size(), i + take);
			if (end - i != take) {
				// Du khong du 1 chunk fixed-batch: fallback infer tung anh.
				for (; i < bgr_images.size(); ++i) {
					auto one = RunBatchNoSplit(session, {bgr_images[i]}, conf_threshold, nms_iou_threshold);
					all.push_back(std::move(one.at(0)));
				}
				break;
			}
			std::vector<cv::Mat> chunk;
			// Tao chunk dung kich thuoc fixed batch de giu dung shape cua model.
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

// API detect anh don, tai su dung chung duong xu ly voi batch.
std::vector<Detection> RunSingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	float conf_threshold,
	float nms_iou_threshold) {
	// Tai su dung pipeline RunBatchNoSplit de giu mot duong preprocess/postprocess duy nhat.
	auto out = RunBatchNoSplit(session, {bgr_image}, conf_threshold, nms_iou_threshold);
	return std::move(out[0]);
}

} // namespace yolo_detector
