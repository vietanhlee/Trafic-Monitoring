#include "ocrplate/services/onnx_runner.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "ocrplate/utils/onnx_decode_utils.h"

namespace onnx_runner {

std::vector<int64_t> RunModelGetArgMax(
    Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c) {
	auto r = RunModelGetArgMaxAndConf(env, model_path, nhwc_u8, h, w, c);
	return std::move(r.indices);
}

ArgMaxWithConfResult RunModelGetArgMaxAndConf(
    Ort::Env& env,
    const std::string& model_path,
    const uint8_t* nhwc_u8,
    int h,
    int w,
    int c) {

    Ort::SessionOptions sess_options;
    sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Mặc định ONNX Runtime trên Linux dùng CPUExecutionProvider.
    Ort::Session session(env, model_path.c_str(), sess_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();

    // Tạo tensor input: (1, H, W, C), uint8
    const std::vector<int64_t> input_shape = {1, h, w, c};
    const size_t input_tensor_size = static_cast<size_t>(h * w * c);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(
        mem_info,
        const_cast<uint8_t*>(nhwc_u8),
        input_tensor_size,
        input_shape.data(),
        input_shape.size());

    // Lấy tên output
    const size_t output_count = session.GetOutputCount();
    if (output_count == 0) {
        throw std::runtime_error("Model không có output");
    }

    std::vector<Ort::AllocatedStringPtr> output_name_alloc;
    std::vector<const char*> output_names;
    output_name_alloc.reserve(output_count);
    output_names.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        output_name_alloc.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_alloc.back().get());
    }

    const std::vector<const char*> input_names = {input_name};
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
    const auto out_shape = shape_info.GetShape();
    const ONNXTensorElementDataType out_type = shape_info.GetElementType();

    // Kỳ vọng output có dạng (N, T, C) hoặc (T, C)
    int64_t time_dim = -1;
    int64_t class_dim = -1;

    if (out_shape.size() == 3) {
        // chỉ decode batch[0]
        time_dim = out_shape[1];
        class_dim = out_shape[2];
    } else if (out_shape.size() == 2) {
        time_dim = out_shape[0];
        class_dim = out_shape[1];
    } else {
        throw std::runtime_error("Output rank không đúng (cần 2 hoặc 3), rank=" + std::to_string(out_shape.size()));
    }

    if (time_dim <= 0 || class_dim <= 0) {
        throw std::runtime_error("Output shape không hợp lệ");
    }

    if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = out0.GetTensorData<float>();
        const float* first = (out_shape.size() == 3) ? (data + 0 * time_dim * class_dim) : data;
        return onnx_decode_utils::ArgMaxWithConf(first, time_dim, class_dim);
    }

    if (out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double* data = out0.GetTensorData<double>();
        const double* first = (out_shape.size() == 3) ? (data + 0 * time_dim * class_dim) : data;
        return onnx_decode_utils::ArgMaxWithConf(first, time_dim, class_dim);
    }

    throw std::runtime_error("Kiểu output chưa hỗ trợ (cần float hoặc double)");
}

} // namespace onnx_runner
