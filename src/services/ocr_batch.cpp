/**
 * @file ocr_batch.cpp
 * @brief Triển khai OCR batch: validate input, infer ONNX, decode CTC và tính confidence.
 */
#include "ocrplate/services/ocr_batch.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "ocrplate/services/post_process_out_string.h"

#include "ocrplate/utils/onnx_decode_utils.h"
#include "ocrplate/utils/parallel_utils.h"

namespace ocr_batch {
namespace {

/**
 * @brief Metadata session OCR được cache để giảm chi phí truy vấn mỗi lần infer.
 */
struct OcrSessionMeta {
	// Kiểu dữ liệu input tensor model OCR (kỳ vọng uint8 trong project hiện tại).
	ONNXTensorElementDataType input_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	// Batch cố định của model nếu shape input xác định; -1 nếu dynamic.
	int64_t fixed_batch = -1;
	// Tên node input trong graph ONNX.
	std::string input_name;
	// Vùng lưu tên output để đảm bảo const char* bên dưới luôn hợp lệ.
	std::vector<std::string> output_name_storage;
	// Danh sách con trỏ tên output dùng trực tiếp cho session.Run().
	std::vector<const char*> output_names;
};

/**
 * @brief Lấy metadata input/output cua session OCR va cache theo session pointer.
 *
 * @param session Session ONNX Runtime cua model OCR.
 * @return std::shared_ptr<const OcrSessionMeta> Metadata da san sang cho infer.
 */
std::shared_ptr<const OcrSessionMeta> GetSessionMeta(Ort::Session& session) {
	static std::mutex cache_mu;
	static std::unordered_map<std::uintptr_t, std::shared_ptr<const OcrSessionMeta>> cache;

	const auto key = reinterpret_cast<std::uintptr_t>(&session);
	// Fast-path: nếu metadata da co trong cache thi tra về ngay.
	{
		std::lock_guard<std::mutex> lk(cache_mu);
		auto it = cache.find(key);
		if (it != cache.end()) {
			return it->second;
		}
	}

	auto meta = std::make_shared<OcrSessionMeta>();
	// Đọc shape/type input một lần để quyết định chế độ chạy fixed-batch hay dynamic.
	auto input_type_info = session.GetInputTypeInfo(0);
	auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	// input_shape thường là [N,H,W,C] với OCR model đang dùng.
	const auto input_shape = input_tensor_info.GetShape();
	meta->input_elem_type = input_tensor_info.GetElementType();
	if (input_shape.size() == 4 && input_shape[0] > 0) {
		// Nếu batch là static, lưu lại để validate input ở runtime.
		meta->fixed_batch = input_shape[0];
	}

	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	meta->input_name = input_name_alloc.get();

	const size_t output_count = session.GetOutputCount();
	if (output_count == 0) {
		throw std::runtime_error("OCR model không có output");
	}
	meta->output_name_storage.reserve(output_count);
	meta->output_names.reserve(output_count);
	for (size_t i = 0; i < output_count; ++i) {
		// Lấy tên từng output để có thể gọi Run() với nhiều output khi cần.
		auto out_name = session.GetOutputNameAllocated(i, allocator);
		meta->output_name_storage.emplace_back(out_name.get());
	}
	for (const auto& s : meta->output_name_storage) {
		meta->output_names.push_back(s.c_str());
	}

	{
		// Ghi metadata vào cache. Nếu race insert xảy ra, dùng bản đã có.
		std::lock_guard<std::mutex> lk(cache_mu);
		auto [it, inserted] = cache.emplace(key, meta);
		if (!inserted) {
			return it->second;
		}
	}
	return meta;
}

/**
 * @brief Kiểm tra mot anh OCR đầu vào co dùng type/shape/memory layout.
 *
 * @param m Anh đầu vào cần kiểm tra.
 * @param h Chiều cao kỳ vọng.
 * @param w Chiều rộng kỳ vọng.
 * @throws std::runtime_error Nếu input không hợp le.
 */
void EnsureRgbU8(const cv::Mat& m, int h, int w) {
	if (m.empty()) {
		throw std::runtime_error("OCR input rong");
	}
	if (m.type() != CV_8UC3) {
		throw std::runtime_error("OCR input cần CV_8UC3");
	}
	if (m.rows != h || m.cols != w) {
		// Tắt ca anh trong batch phai dong nhất shape để pack 1 tensor liên tục.
		throw std::runtime_error("OCR input sai kích thước (cần " + std::to_string(w) + "x" + std::to_string(h) + ")");
	}
	if (!m.isContinuous()) {
		// Tensor tao bang memcpy yeu cau dữ liệu liên tục trong bộ nhớ.
		throw std::runtime_error("OCR input phai continuous");
	}
}

/**
 * @brief Decode output tensor OCR theo batch thanh index + confidence.
 *
 * @param out0 Output tensor dau tien cua model OCR.
 * @return std::vector<onnx_runner::ArgMaxWithConfResult> Kết quả decode theo batch.
 */
std::vector<onnx_runner::ArgMaxWithConfResult> DecodeBatchOutput(Ort::Value& out0) {
	if (!out0.IsTensor()) {
		throw std::runtime_error("OCR output[0] không phai tensor");
	}
	auto info = out0.GetTensorTypeAndShapeInfo();
	const auto shape = info.GetShape();
	const auto elem_type = info.GetElementType();
	if (shape.size() != 3) {
		throw std::runtime_error("OCR output rank không hop le (cần 3), rank=" + std::to_string(shape.size()));
	}
	const int64_t batch = shape[0];
	// time_dim: so timestep theo truc chuoi OCR.
	const int64_t time_dim = shape[1];
	// class_dim: so lop ký tự (alphabet + blank).
	const int64_t class_dim = shape[2];
	if (batch <= 0 || time_dim <= 0 || class_dim <= 0) {
		throw std::runtime_error("OCR output shape không hop le");
	}

	std::vector<onnx_runner::ArgMaxWithConfResult> out;
	out.reserve(static_cast<size_t>(batch));
	if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		// Model xuat float32: decode tung sample theo [T, C].
		const float* data = out0.GetTensorData<float>();
		for (int64_t n = 0; n < batch; ++n) {
			// base la con tro dau ma tran logits cua sample n.
			const float* base = data + n * time_dim * class_dim;
			out.push_back(onnx_decode_utils::ArgMaxWithConf(base, time_dim, class_dim));
		}
		return out;
	}
	if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
		// Hỗ trợ them output double để tương thích model export khac nhau.
		const double* data = out0.GetTensorData<double>();
		for (int64_t n = 0; n < batch; ++n) {
			// base la con tro dau ma tran logits cua sample n (kieu f64).
			const double* base = data + n * time_dim * class_dim;
			out.push_back(onnx_decode_utils::ArgMaxWithConf(base, time_dim, class_dim));
		}
		return out;
	}
	throw std::runtime_error("OCR output type chưa hỗ trợ (chi float/double)");
}

/**
 * @brief Tinh confidence trung binh tren cac timestep non-blank.
 *
 * @param indices Day index top-1 theo timestep.
 * @param conf Day confidence top-1 theo timestep.
 * @param blank_index Chỉ số token blank CTC.
 * @return float Gia tri confidence trung binh cho phan text thuc.
 */
float ComputeAvgConfNonBlank(const std::vector<int64_t>& indices, const std::vector<float>& conf, int64_t blank_index) {
	if (indices.empty() || conf.empty()) {
		return 0.0f;
	}
	// T la so timestep hop le co du ca index va conf.
	const size_t T = std::min(indices.size(), conf.size());
	int last_nonblank = -1;
	for (size_t t = 0; t < T; ++t) {
		// Tim vi tri ký tự non-blank cuoi để bỏ qua phan duoi chuoi (padding/time-step du).
		if (indices[t] != blank_index) {
			last_nonblank = static_cast<int>(t);
		}
	}
	if (last_nonblank < 0) {
		// Toàn bộ la blank -> coi nhu không co text hop le.
		return 0.0f;
	}
	double sum = 0.0;
	// cnt la so timestep non-blank được dùng để tinh trung binh.
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

/**
 * @brief Xu ly truong hop model fix batch=1 nhung input co nhiều anh.
 *
 * Ham fan-out thanh nhiều infer don va chạy song song theo worker threads.
 *
 * @param session Session ONNX Runtime OCR.
 * @param rgb_u8_hwc Danh sach anh OCR input.
 * @param alphabet Bang ký tự CTC.
 * @return std::vector<OcrText> Kết quả OCR theo thứ tự đầu vào.
 */
std::vector<OcrText> RunFixedBatchOneParallel(
	Ort::Session& session,
	const std::vector<cv::Mat>& rgb_u8_hwc,
	const std::string& alphabet) {
	std::vector<OcrText> all(rgb_u8_hwc.size());
	// worker_count được tinh theo item_count va so core may.
	const size_t worker_count = parallel_utils::ResolveWorkerCount(rgb_u8_hwc.size());
	if (worker_count <= 1 || rgb_u8_hwc.size() <= 1) {
		for (size_t i = 0; i < rgb_u8_hwc.size(); ++i) {
			auto out = RunBatch(session, {rgb_u8_hwc[i]}, alphabet);
			all[i] = std::move(out[0]);
		}
		return all;
	}

	// Atomic scheduling giup cần bang tai va giảm overhead tao future.
	std::atomic<size_t> next_index{0};
	std::vector<std::thread> workers;
	workers.reserve(worker_count);
	for (size_t w = 0; w < worker_count; ++w) {
		workers.emplace_back([&]() {
			while (true) {
				// i la index sample tiep theo se được worker hiện tại xu ly.
				const size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
				if (i >= rgb_u8_hwc.size()) {
					break;
				}
				// Gọi RunBatch voi batch=1 để tương thích model fixed-batch=1.
				auto out = RunBatch(session, {rgb_u8_hwc[i]}, alphabet);
				all[i] = std::move(out[0]);
			}
		});
	}
	for (auto& t : workers) {
		t.join();
	}
	return all;
}

} // namespace

/**
 * @brief Infer OCR theo batch va tra về text + confidence tung anh.
 *
 * @param session Session ONNX Runtime OCR.
 * @param rgb_u8_hwc Danh sach anh RGB uint8 HWC.
 * @param alphabet Bang ký tự CTC.
 * @return std::vector<OcrText> Kết quả OCR theo thứ tự đầu vào.
 */
std::vector<OcrText> RunBatch(
	Ort::Session& session,
	const std::vector<cv::Mat>& rgb_u8_hwc,
	const std::string& alphabet) {
	if (rgb_u8_hwc.empty()) {
		// Không co input -> không co output.
		return {};
	}
	const auto meta = GetSessionMeta(session);

	// Xử lý model fix batch (thường gặp: batch=1)
	if (meta->fixed_batch > 0) {
		// fixed_batch: rang buoc shape input tu model (nếu model static batch).
		const int64_t fixed_batch = meta->fixed_batch;
		if (fixed_batch == 1 && rgb_u8_hwc.size() > 1) {
			// Model batch=1: fan-out thanh nhiều infer don để tan dùng CPU.
			return RunFixedBatchOneParallel(session, rgb_u8_hwc, alphabet);
		}
		if (fixed_batch > 1 && static_cast<int64_t>(rgb_u8_hwc.size()) != fixed_batch) {
			throw std::runtime_error("OCR model fix batch=" + std::to_string(fixed_batch) + ", nhung input batch=" + std::to_string(rgb_u8_hwc.size()));
		}
	}

	const int h = rgb_u8_hwc[0].rows;
	const int w = rgb_u8_hwc[0].cols;
	// Validate dong nhất shape/type toàn bộ batch trước khi tao tensor.
	for (const auto& m : rgb_u8_hwc) {
		EnsureRgbU8(m, h, w);
	}

	if (meta->input_elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
		throw std::runtime_error("OCR input type không dùng (cần uint8)");
	}

	const int64_t batch = static_cast<int64_t>(rgb_u8_hwc.size());
	// Shape input OCR dạng su dùng: [N,H,W,3] (NHWC).
	const std::vector<int64_t> run_input_shape = {batch, h, w, 3};
	// per: so byte/pixel elements cua 1 anh RGB HWC.
	const size_t per = static_cast<size_t>(h) * static_cast<size_t>(w) * 3ull;
	std::vector<uint8_t> input;
	input.resize(static_cast<size_t>(batch) * per);
	for (size_t i = 0; i < rgb_u8_hwc.size(); ++i) {
		// Layout NHWC uint8: copy lien tiep tung anh vào vung batch.
		std::memcpy(input.data() + i * per, rgb_u8_hwc[i].data, per);
	}

	Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(
		mem_info,
		input.data(),
		input.size(),
		run_input_shape.data(),
		run_input_shape.size());

	const std::vector<const char*> input_names = {meta->input_name.c_str()};
	// outputs gom toàn bộ output nodes, nhung decode chu yeu output[0].
	auto outputs = session.Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		&input_tensor,
		1,
		meta->output_names.data(),
		meta->output_names.size());

	Ort::Value& out0 = outputs.at(0);
	// decoded la kết quả argmax + conf theo tung sample trong batch.
	auto decoded = DecodeBatchOutput(out0);
	// blank_index la vi tri token blank CTC trong alphabet.
	const int64_t blank_index = static_cast<int64_t>(alphabet.size()) - 1;

	std::vector<OcrText> texts;
	texts.reserve(decoded.size());
	for (auto& one : decoded) {
		OcrText t;
		// Postprocess CTC: collapse + loai blank -> chuoi bịển số.
		t.text = post_process_out_string::PostprocessIndicesToString(one.indices, alphabet, blank_index);
		// conf_avg được tinh tren ký tự non-blank để phan anh độ tin cậy text thuc.
		t.conf_avg = ComputeAvgConfNonBlank(one.indices, one.conf, blank_index);
		texts.push_back(std::move(t));
	}
	return texts;
}

} // namespace ocr_batch
