/**
 * @file onnx_runner.cpp
 * @brief Triển khai helper infer ONNX OCR và decode argmax/confidence.
 */
#include "ocrplate/services/onnx_runner.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "ocrplate/utils/onnx_decode_utils.h"

namespace onnx_runner {

/**
 * @brief Wrapper gọn chỉ trả về chuỗi index top-1.
 *
 * @param env Ort::Env đã khởi tạo.
 * @param model_path Đường dẫn model .onnx.
 * @param nhwc_u8 Con trỏ dữ liệu ảnh NHWC uint8.
 * @param h Chiều cao input.
 * @param w Chiều rộng input.
 * @param c Số kênh input.
 * @return std::vector<int64_t> Danh sách index top-1 theo timestep.
 */
std::vector<int64_t> RunModelGetArgMax(
    Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c) {
	// Wrapper gọn: tái sử dụng đường infer đầy đủ và chỉ lấy indices.
	auto r = RunModelGetArgMaxAndConf(env, model_path, nhwc_u8, h, w, c);
	return std::move(r.indices);
}

/**
 * @brief Infer model OCR ONNX và tra về index + confidence.
 *
 * @param env Ort::Env đã khởi tạo.
 * @param model_path Đường dẫn model .onnx.
 * @param nhwc_u8 Con trỏ dữ liệu ảnh NHWC uint8.
 * @param h Chiều cao input.
 * @param w Chiều rộng input.
 * @param c Số kênh input.
 * @return ArgMaxWithConfResult Kết quả top-1 theo timestep.
 */
ArgMaxWithConfResult RunModelGetArgMaxAndConf(
    Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c) {
    // ORT_ENABLE_ALL bật tối ưu graph để giảm độ trễ infer.

    Ort::SessionOptions sess_options;
    sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Mac định ONNX Runtime tren Linux dùng CPUExecutionProvider.
    Ort::Session session(env, model_path.c_str(), sess_options);

    Ort::AllocatorWithDefaultOptions allocator;
    // Lấy tên input/output tu graph để run bang API ten-node.
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
	// Con trỏ tên input hợp lệ trong vòng doi cua input_name_alloc.
    const char* input_name = input_name_alloc.get();

    // Tao tensor input: (1, H, W, C), uint8.
    // Luu y: ham nay kỳ vọng nhwc_u8 da dùng layout NHWC.
    const std::vector<int64_t> input_shape = {1, h, w, c};
	// So phan tu tensor input (không tinh bytes) cho shape (1,H,W,C).
    const size_t input_tensor_size = static_cast<size_t>(h * w * c);

    // Tạo mô tả bộ nhớ CPU cho tensor đầu vào.
    // ONNX Runtime cần metadata này để bịết vùng nhớ nằm trên CPU và cách quản lý allocator.
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Tạo tensor input từ buffer ảnh NHWC uint8 có sẵn.
    // Lưu ý về const_cast:
    // - API CreateTensor nhận con trỏ không const vì có thể tái sử dụng chung cho nhiều trường hợp.
    // - Trong luồng hiện tại, ONNX Runtime chỉ đọc dữ liệu đầu vào, không ghi đè lên buffer nhwc_u8.
    // - Vì vậy const_cast ở đây dùng để tương thích chữ ký hàm, không thay đổi dữ liệu gốc.
    Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(
        mem_info,
        const_cast<uint8_t*>(nhwc_u8),
        input_tensor_size,
        input_shape.data(),
        input_shape.size());

    // Đếm số output node để chuẩn bị danh sách tên output truyền vào session.Run().
    // Việc gọi theo tên node giúp mã rõ ràng và ổn định khi model có nhiều output.
    const size_t output_count = session.GetOutputCount();
    if (output_count == 0) {
        throw std::runtime_error("Model không có output");
    }

    // output_name_alloc giữ quyền sở hữu chuỗi tên output do ONNX Runtime cấp phát.
    // output_names chỉ lưu con trỏ const char* để truyền nhánh vào API Run().
    // Cần giữ output_name_alloc sống đến hết phiên gọi Run(), nếu không các con trỏ sẽ bị dangling.
    std::vector<Ort::AllocatedStringPtr> output_name_alloc;
    std::vector<const char*> output_names;

    // Reserve trước để giảm cấp phát lại và giữ ổn định vùng nhớ khi push_back.
    output_name_alloc.reserve(output_count);
    output_names.reserve(output_count);

    // Thu thập toàn bộ tên output theo thứ tự index trong graph.
    for (size_t i = 0; i < output_count; ++i) {
        output_name_alloc.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_alloc.back().get());
    }

    const std::vector<const char*> input_names = {input_name};
    // Infer 1 lan, lấy tắt ca output nhung uu tien output[0] cho decode CTC.
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        output_names.size());

    // infer.py lấy outs[0]
    Ort::Value& out0 = outputs.at(0);
    if (!out0.IsTensor()) {
        throw std::runtime_error("Output[0] không phải tensor");
    }

    auto shape_info = out0.GetTensorTypeAndShapeInfo();
	// out_shape dùng để suy luận rank va kích thước T/C phục vụ decode.
    const auto out_shape = shape_info.GetShape();
    const ONNXTensorElementDataType out_type = shape_info.GetElementType();

    // Kỳ vọng output co dạng (N, T, C) hoặc (T, C)
    int64_t time_dim = -1;
    int64_t class_dim = -1;

    if (out_shape.size() == 3) {
		// Rank-3: [N, T, C]. O day chi decode sample dau tien (batch[0]).
        time_dim = out_shape[1];
        class_dim = out_shape[2];
    } else if (out_shape.size() == 2) {
		// Rank-2: [T, C], không co truc batch.
        time_dim = out_shape[0];
        class_dim = out_shape[1];
    } else {
        throw std::runtime_error("Output rank không đúng (cần 2 hoặc 3), rank=" + std::to_string(out_shape.size()));
    }

    if (time_dim <= 0 || class_dim <= 0) {
		// Bao về trước khi decode để tranh truy cap bộ nhớ sai.
        throw std::runtime_error("Output shape không hợp lệ");
    }

    if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = out0.GetTensorData<float>();
		// first tro den dau vung [T,C] cần decode.
        const float* first = (out_shape.size() == 3) ? (data + 0 * time_dim * class_dim) : data;
        return onnx_decode_utils::ArgMaxWithConf(first, time_dim, class_dim);
    }

    if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double* data = out0.GetTensorData<double>();
		// Tuong tu branch float32, nhung danh cho output float64.
        const double* first = (out_shape.size() == 3) ? (data + 0 * time_dim * class_dim) : data;
        return onnx_decode_utils::ArgMaxWithConf(first, time_dim, class_dim);
    }

    throw std::runtime_error("Kiểu output chưa hỗ trợ (cần float hoặc double)");
}

} // namespace onnx_runner
