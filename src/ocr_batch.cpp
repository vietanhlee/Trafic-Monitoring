#include "ocr_batch.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <stdexcept>

#include "post_process_out_string.h"

#include "utils/onnx_decode_utils.h"
#include "utils/parallel_utils.h"

namespace ocr_batch {
namespace {

void EnsureRgbU8(const cv::Mat& m, int h, int w) {
	if (m.empty()) {
		throw std::runtime_error("OCR input rong");
	}
	if (m.type() != CV_8UC3) {
		throw std::runtime_error("OCR input can CV_8UC3");
	}
	if (m.rows != h || m.cols != w) {
		throw std::runtime_error("OCR input sai kich thuoc (can " + std::to_string(w) + "x" + std::to_string(h) + ")");
	}
	if (!m.isContinuous()) {
		throw std::runtime_error("OCR input phai continuous");
	}
}

std::vector<onnx_runner::ArgMaxWithConfResult> DecodeBatchOutput(Ort::Value& out0) {
	if (!out0.IsTensor()) {
		throw std::runtime_error("OCR output[0] khong phai tensor");
	}
	auto info = out0.GetTensorTypeAndShapeInfo();
	const auto shape = info.GetShape();
	const auto elem_type = info.GetElementType();
	if (shape.size() != 3) {
		throw std::runtime_error("OCR output rank khong hop le (can 3), rank=" + std::to_string(shape.size()));
	}
	const int64_t batch = shape[0];
	const int64_t time_dim = shape[1];
	const int64_t class_dim = shape[2];
	if (batch <= 0 || time_dim <= 0 || class_dim <= 0) {
		throw std::runtime_error("OCR output shape khong hop le");
	}

	std::vector<onnx_runner::ArgMaxWithConfResult> out;
	out.reserve(static_cast<size_t>(batch));
	if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		const float* data = out0.GetTensorData<float>();
		for (int64_t n = 0; n < batch; ++n) {
			const float* base = data + n * time_dim * class_dim;
			out.push_back(onnx_decode_utils::ArgMaxWithConf(base, time_dim, class_dim));
		}
		return out;
	}
	if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
		const double* data = out0.GetTensorData<double>();
		for (int64_t n = 0; n < batch; ++n) {
			const double* base = data + n * time_dim * class_dim;
			out.push_back(onnx_decode_utils::ArgMaxWithConf(base, time_dim, class_dim));
		}
		return out;
	}
	throw std::runtime_error("OCR output type chua ho tro (chi float/double)");
}

float ComputeAvgConfNonBlank(const std::vector<int64_t>& indices, const std::vector<float>& conf, int64_t blank_index) {
	if (indices.empty() || conf.empty()) {
		return 0.0f;
	}
	const size_t T = std::min(indices.size(), conf.size());
	int last_nonblank = -1;
	for (size_t t = 0; t < T; ++t) {
		if (indices[t] != blank_index) {
			last_nonblank = static_cast<int>(t);
		}
	}
	if (last_nonblank < 0) {
		return 0.0f;
	}
	double sum = 0.0;
	size_t cnt = 0;
	for (int t = 0; t <= last_nonblank; ++t) {
		if (indices[static_cast<size_t>(t)] == blank_index) {
			continue;
		}
		sum += static_cast<double>(conf[static_cast<size_t>(t)]);
		++cnt;
	}
	return (cnt == 0) ? 0.0f : static_cast<float>(sum / static_cast<double>(cnt));
}

std::vector<OcrText> RunFixedBatchOneParallel(
	Ort::Session& session,
	const std::vector<cv::Mat>& rgb_u8_hwc,
	const std::string& alphabet) {
	std::vector<OcrText> all(rgb_u8_hwc.size());
	const size_t worker_count = parallel_utils::ResolveWorkerCount(rgb_u8_hwc.size());
	const size_t chunk = (rgb_u8_hwc.size() + worker_count - 1) / worker_count;

	std::vector<std::future<void>> workers;
	workers.reserve(worker_count);
	for (size_t w = 0; w < worker_count; ++w) {
		const size_t begin = w * chunk;
		if (begin >= rgb_u8_hwc.size()) {
			break;
		}
		const size_t end = std::min(rgb_u8_hwc.size(), begin + chunk);
		workers.push_back(std::async(std::launch::async, [&, begin, end]() {
			for (size_t i = begin; i < end; ++i) {
				auto out = RunBatch(session, {rgb_u8_hwc[i]}, alphabet);
				all[i] = std::move(out[0]);
			}
		}));
	}
	for (auto& t : workers) {
		t.get();
	}
	return all;
}

} // namespace

std::vector<OcrText> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& rgb_u8_hwc,
	const std::string& alphabet) {
	if (rgb_u8_hwc.empty()) {
		return {};
	}

	// Xử lý model fix batch (thường gặp: batch=1)
	auto input_type_info = session.GetInputTypeInfo(0);
	auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	const auto input_shape = input_tensor_info.GetShape();
	if (input_shape.size() == 4 && input_shape[0] > 0) {
		const int64_t fixed_batch = input_shape[0];
		if (fixed_batch == 1 && rgb_u8_hwc.size() > 1) {
			return RunFixedBatchOneParallel(session, rgb_u8_hwc, alphabet);
		}
		if (fixed_batch > 1 && static_cast<int64_t>(rgb_u8_hwc.size()) != fixed_batch) {
			throw std::runtime_error("OCR model fix batch=" + std::to_string(fixed_batch) + ", nhung input batch=" + std::to_string(rgb_u8_hwc.size()));
		}
	}

	const int h = rgb_u8_hwc[0].rows;
	const int w = rgb_u8_hwc[0].cols;
	for (const auto& m : rgb_u8_hwc) {
		EnsureRgbU8(m, h, w);
	}

	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	const char* input_name = input_name_alloc.get();

	const size_t output_count = session.GetOutputCount();
	if (output_count == 0) {
		throw std::runtime_error("OCR model khong co output");
	}
	std::vector<Ort::AllocatedStringPtr> output_name_alloc;
	std::vector<const char*> output_names;
	output_name_alloc.reserve(output_count);
	output_names.reserve(output_count);
	for (size_t i = 0; i < output_count; ++i) {
		output_name_alloc.push_back(session.GetOutputNameAllocated(i, allocator));
		output_names.push_back(output_name_alloc.back().get());
	}

	const auto elem_type = input_tensor_info.GetElementType();
	if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
		throw std::runtime_error("OCR input type khong dung (can uint8)");
	}

	const int64_t batch = static_cast<int64_t>(rgb_u8_hwc.size());
	const std::vector<int64_t> run_input_shape = {batch, h, w, 3};
	const size_t per = static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull;
	std::vector<uint8_t> input;
	input.resize(static_cast<size_t>(batch) * per);
	for (size_t i = 0; i < rgb_u8_hwc.size(); ++i) {
		std::memcpy(input.data() + i * per, rgb_u8_hwc[i].data, per);
	}

	Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(
		mem_info,
		input.data(),
		input.size(),
		run_input_shape.data(),
		run_input_shape.size());

	const std::vector<const char*> input_names = {input_name};
	auto outputs = session.Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		&input_tensor,
		1,
		output_names.data(),
		output_names.size());

	Ort::Value& out0 = outputs.at(0);
	auto decoded = DecodeBatchOutput(out0);
	const int64_t blank_index = static_cast<int64_t>(alphabet.size()) - 1;

	std::vector<OcrText> texts;
	texts.reserve(decoded.size());
	for (auto& one : decoded) {
		OcrText t;
		t.text = post_process_out_string::PostprocessIndicesToString(one.indices, alphabet, blank_index);
		t.conf_avg = ComputeAvgConfNonBlank(one.indices, one.conf, blank_index);
		texts.push_back(std::move(t));
	}
	return texts;
}

} // namespace ocr_batch
