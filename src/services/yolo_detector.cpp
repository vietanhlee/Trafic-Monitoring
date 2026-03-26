/**
 * @file yolo_detector.cpp
 * @brief Triển khai suy luận YOLO theo batch/ảnh đơn, hỗ trợ dynamic và fixed batch.
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

/**
 * @brief Metadata I/O của session được cache theo session pointer.
 */
struct SessionIoNames {
	// Tên input node của detector.
	std::string input_name;
	// Vùng lưu tên output để đảm bảo vòng đời cho các const char* bên dưới.
	std::vector<std::string> output_name_storage;
	// Danh sách con trỏ output name truyền trực tiếp vào session.Run().
	std::vector<const char*> output_names;
};

/**
 * @brief Lấy metadata tên input/output của session và cache theo session pointer.
 *
 * @param session Session ONNX Runtime của model YOLO.
 * @return std::shared_ptr<const SessionIoNames> Metadata tên node I/O.
 */
std::shared_ptr<const SessionIoNames> GetSessionIoNames(Ort::Session& session) {
	static std::mutex cache_mu;
	static std::unordered_map<std::uintptr_t, std::shared_ptr<const SessionIoNames>> cache;

	const auto key = reinterpret_cast<std::uintptr_t>(&session);
	// Cache hit: dùng lại metadata, bỏ qua truy vấn graph.
	{
		std::lock_guard<std::mutex> lk(cache_mu);
		auto it = cache.find(key);
		if (it != cache.end()) {
			return it->second;
		}
	}

	// Build metadata khi cache miss.
	Ort::AllocatorWithDefaultOptions allocator;
	// built là object tạm chứa metadata I/O trước khi đưa vào cache dùng chung.
	auto built = std::make_shared<SessionIoNames>();
	// Lấy tên input node thứ 0 từ graph ONNX.
	// ONNX Runtime trả về kiểu AllocatedStringPtr, phải copy sang std::string
	// để tên còn hợp lệ sau khi biến tạm bị hủy.
	auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	built->input_name = input_name_alloc.get();

	// Đếm số output node của model để chuẩn bị mảng tên output truyền vào Run().
	const size_t output_count = session.GetOutputCount();
	if (output_count == 0) {
		throw std::runtime_error("YOLO model không có output");
	}
	// Reserve trước để tránh cấp phát lại nhiều lần khi push tên output.
	built->output_name_storage.reserve(output_count);
	built->output_names.reserve(output_count);
	for (size_t i = 0; i < output_count; ++i) {
		// Đọc tên output thứ i và copy vào vùng lưu string bền vững.
		auto out_name = session.GetOutputNameAllocated(i, allocator);
		built->output_name_storage.emplace_back(out_name.get());
	}
	for (const auto& s : built->output_name_storage) {
		// Tạo mảng const char* trỏ vào string đã lưu để dùng trực tiếp cho session.Run().
		built->output_names.push_back(s.c_str());
	}

	{
		// Ghi metadata vào cache dùng chung.
		// Nếu có race (thread khác vừa ghi trước), dùng bản đã có để đảm bảo nhất quán.
		std::lock_guard<std::mutex> lk(cache_mu);
		auto [it, inserted] = cache.emplace(key, built);
		if (!inserted) {
			return it->second;
		}
	}
	return built;
}

// ── Inference (single batch, no split) ────────────────────────────────

/**
 * @brief Chạy suy luận cho một batch liên tục (không tách chunk).
 *
 * @param session Session ONNX Runtime của model YOLO.
 * @param bgr_images Danh sách ảnh BGR đầu vào.
 * @param conf_threshold Ngưỡng confidence lọc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<std::vector<Detection>> Detection theo từng ảnh trong batch.
 */
std::vector<std::vector<Detection>> RunBatchNoSplit(
	Ort::Session& session,
	const std::vector<cv::Mat>& bgr_images,
	float conf_threshold,
	float nms_iou_threshold) {
	if (bgr_images.empty()) {
		return {};
	}

	const InputSpec spec = GetInputSpec(session);
	// io_names gồm input_name + output_names đã cache.
	const auto io_names = GetSessionIoNames(session);
	const int in_h = static_cast<int>(spec.h);
	const int in_w = static_cast<int>(spec.w);

	std::vector<cv::Mat> rgbs;
	std::vector<LetterboxInfo> infos;
	// rgbs và infos cùng index với bgr_images.
	rgbs.resize(bgr_images.size());
	infos.resize(bgr_images.size());

	const size_t count = bgr_images.size();
	const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
	const size_t worker_count = std::min(count, static_cast<size_t>(hw));
	if (worker_count <= 1) {
		// Batch nhỏ: chạy tuần tự để tránh overhead tạo thread.
		for (size_t i = 0; i < count; ++i) {
			rgbs[i] = LetterboxToSizeRGB(bgr_images[i], in_w, in_h, infos[i]);
		}
	} else {
		// Batch lớn: chia preprocessing theo chỉ số atomic cho nhiều worker.
		// Ý tưởng: mỗi worker lấy 1 index ảnh chưa xử lý, chạy letterbox+RGB,
		// rồi tiếp tục lấy index kế tiếp cho đến khi hết việc.
		std::atomic<size_t> next_index{0};
		// Danh sách thread worker dùng để xử lý song song các ảnh trong batch.
		std::vector<std::thread> workers;
		// Reserve trước để tránh cấp phát lại khi emplace_back nhiều thread.
		workers.reserve(worker_count);
		for (size_t w = 0; w < worker_count; ++w) {
			// Mỗi worker chạy cùng một hàm lambda với cơ chế lấy việc động.
			// Capture [&] để dùng chung các biến: next_index, rgbs, infos, bgr_images...
			workers.emplace_back([&]() {
				while (true) {
					// fetch_add trả về index hiện tại rồi tăng lên 1 cho lần kế tiếp.
					// memory_order_relaxed đủ dùng vì ta chỉ cần phát index duy nhất,
					// không cần ràng buộc đồng bộ thứ tự dữ liệu giữa các thread ở đây.
					const size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
					if (i >= count) {
						// Hết ảnh cần xử lý -> worker thoát vòng lặp.
						break;
					}
					// Tiền xử lý ảnh thứ i: resize+letterbox+BGR->RGB và lưu metadata map ngược.
					// Mỗi thread ghi vào index i riêng nên không đụng ghi chéo dữ liệu.
					rgbs[i] = LetterboxToSizeRGB(bgr_images[i], in_w, in_h, infos[i]);
				}
			});
		}
		// Chờ toàn bộ worker hoàn thành để đảm bảo rgbs/infos đã đầy đủ trước khi infer.
		for (auto& t : workers) {
			t.join();
		}
	}

	// Mô tả bộ nhớ CPU cho tensor input mà ONNX Runtime sẽ đọc.
	Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	// Số phần tử batch thực tế sau preprocess.
	const int64_t batch = static_cast<int64_t>(rgbs.size());

	// input_shape là shape tensor đầu vào theo đúng layout model (NCHW hoặc NHWC).
	std::vector<int64_t> input_shape;
	if (spec.nchw) {
		// Input tensor layout [N,C,H,W].
		input_shape = {batch, 3, spec.h, spec.w};
	} else {
		// Input tensor layout [N,H,W,C].
		input_shape = {batch, spec.h, spec.w, 3};
	}

	// Các buffer dữ liệu input theo từng kiểu model có thể yêu cầu.
	// Chỉ một trong ba buffer sẽ được dùng để tạo input_tensor.
	std::vector<float> input_f32;
	std::vector<Ort::Float16_t> input_f16;
	std::vector<uint8_t> input_u8;
	// input_tensor là đối tượng ONNX Runtime đại diện cho tensor đầu vào cuối cùng.
	Ort::Value input_tensor{nullptr};
	if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		// Model float32: có thể scale [0,1] trước khi tạo tensor input.
		// scale_01=true vì đa số model float kỳ vọng giá trị đã chuẩn hóa.
		if (spec.nchw) {
			FillTensorFromRGB_NCHW(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		} else {
			FillTensorFromRGB_NHWC(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		}
		// Tạo tensor float32 trỏ trực tiếp vào buffer input_f32.
		input_tensor = Ort::Value::CreateTensor<float>(
			mem_info,
			input_f32.data(),
			input_f32.size(),
			input_shape.data(),
			input_shape.size());
	} else if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
		// Model float16: preprocess theo float32, sau đó convert sang fp16.
		if (spec.nchw) {
			FillTensorFromRGB_NCHW(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		} else {
			FillTensorFromRGB_NHWC(rgbs, in_h, in_w, input_f32, /*scale_01=*/true);
		}
		input_f16.resize(input_f32.size());
		for (size_t i = 0; i < input_f32.size(); ++i) {
			// Ép từng phần tử float32 sang float16 để đúng kiểu input model.
			input_f16[i] = Ort::Float16_t(input_f32[i]);
		}
		// Tạo tensor float16 từ buffer input_f16 đã convert.
		input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
			mem_info,
			input_f16.data(),
			input_f16.size(),
			input_shape.data(),
			input_shape.size());
	} else if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
		// Model uint8: giữ nguyên dynamic range pixel, copy theo layout model cần.
		if (spec.nchw) {
			FillTensorFromRGB_NCHW(rgbs, in_h, in_w, input_u8, /*scale_01=*/false);
		} else {
			FillTensorFromRGB_NHWC(rgbs, in_h, in_w, input_u8, /*scale_01=*/false);
		}
		// Tạo tensor uint8 khi model kỳ vọng byte image thô.
		input_tensor = Ort::Value::CreateTensor<uint8_t>(
			mem_info,
			input_u8.data(),
			input_u8.size(),
			input_shape.data(),
			input_shape.size());
	} else {
		// Chặn sớm kiểu input chưa hỗ trợ để tránh infer sai dữ liệu ngầm.
		throw std::runtime_error("YOLO input type chưa hỗ trợ (chi hỗ trợ float32/float16/uint8)");
	}

	// Danh sách tên input; ở đây model detector chỉ dùng 1 input chính.
	const std::vector<const char*> input_names = {io_names->input_name.c_str()};
	// Chạy ONNX session và parse output đầu tiên theo format YOLO.
	// output names lấy từ io_names (đã cache) để giảm chi phí truy vấn metadata lặp lại.
	auto outputs = session.Run(
		Ort::RunOptions{nullptr},
		input_names.data(),
		&input_tensor,
		1,
		io_names->output_names.data(),
		io_names->output_names.size());

	// Theo thiết kế hiện tại, parser đọc output đầu tiên (head chính của detector).
	Ort::Value& out0 = outputs.at(0);
	// ParseOutput sẽ decode box/class/score, map ngược về ảnh gốc, rồi NMS theo cấu hình.
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
	// Nếu model bị fixed batch (ví dụ n=1), cần chia nhỏ input theo đúng batch model.
	// Mục tiêu: giữ hành vi đúng cho cả dynamic-batch và static-batch model.
	if (fixed_batch > 0 && static_cast<int64_t>(bgr_images.size()) != fixed_batch) {
		std::vector<std::vector<Detection>> all;
		all.reserve(bgr_images.size());
		for (size_t i = 0; i < bgr_images.size(); ) {
			const size_t take = static_cast<size_t>(fixed_batch);
			const size_t end = std::min(bgr_images.size(), i + take);
			if (end - i != take) {
				// Nếu không đủ 1 chunk fixed-batch: fallback infer từng ảnh.
				for (; i < bgr_images.size(); ++i) {
					auto one = RunBatchNoSplit(session, {bgr_images[i]}, conf_threshold, nms_iou_threshold);
					all.push_back(std::move(one.at(0)));
				}
				break;
			}
			std::vector<cv::Mat> chunk;
			// Tạo chunk đúng kích thước fixed batch để giữ đúng shape của model.
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

std::vector<Detection> RunSingle(
	Ort::Session& session,
	const cv::Mat& bgr_image,
	float conf_threshold,
	float nms_iou_threshold) {
	// Tái sử dụng pipeline RunBatchNoSplit để giữ một đường preprocess/postprocess duy nhất.
	auto out = RunBatchNoSplit(session, {bgr_image}, conf_threshold, nms_iou_threshold);
	return std::move(out[0]);
}

} // namespace yolo_detector
