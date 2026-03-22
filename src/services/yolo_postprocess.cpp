/**
 * @file yolo_postprocess.cpp
 * @brief Trien khai parse output YOLO, map bbox về anh goc va hoan tắt NMS.
 */
#include "ocrplate/services/yolo_detector_internal.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo_detector {
namespace detail {

namespace {

/**
 * @brief Hoan tắt danh sach detection sau loc score bang NMS hoặc top-1 mode.
 *
 * @param dets Danh sach detection trước khi finalize.
 * @param nms_iou_threshold Ngưỡng IoU NMS; <=0 thi giu top-1.
 * @return std::vector<Detection> Danh sach detection sau finalize.
 */
std::vector<Detection> FinalizeDetections(std::vector<Detection> dets, float nms_iou_threshold) {
	if (dets.empty()) {
		return {};
	}

	// No-NMS mode: chi giu 1 bbox co score cao nhất cho moi anh.
	if (nms_iou_threshold <= 0.0f) {
		auto best_it = std::max_element(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
			return a.score < b.score;
		});
		return { *best_it };
	}

	return ApplyNMS(std::move(dets), nms_iou_threshold);
}

} // namespace

// ── Geometry helpers ──────────────────────────────────────────────────

/**
 * @brief Map bbox tu he tọa độ letterbox về he tọa độ anh goc.
 *
 * @param in BBox trong he tọa độ input model.
 * @param info Thong tin letterbox cua anh do.
 * @return Detection BBox da map về anh goc.
 */
Detection MapBackToOriginal(const Detection& in, const LetterboxInfo& info) {
	Detection out = in;
	// Hoan tac letterbox: bo padding va chia theo scale để về tọa độ anh goc.
	out.x1 = (out.x1 - static_cast<float>(info.pad_x)) / info.scale;
	out.y1 = (out.y1 - static_cast<float>(info.pad_y)) / info.scale;
	out.x2 = (out.x2 - static_cast<float>(info.pad_x)) / info.scale;
	out.y2 = (out.y2 - static_cast<float>(info.pad_y)) / info.scale;

	out.x1 = std::clamp(out.x1, 0.0f, static_cast<float>(info.orig_w - 1));
	out.y1 = std::clamp(out.y1, 0.0f, static_cast<float>(info.orig_h - 1));
	out.x2 = std::clamp(out.x2, 0.0f, static_cast<float>(info.orig_w - 1));
	out.y2 = std::clamp(out.y2, 0.0f, static_cast<float>(info.orig_h - 1));
	return out;
}

// ── Parse standard YOLO output ────────────────────────────────────────
// Output shape: (batch, 4+num_classes, num_anchors)
// Layout mỗi channel: [cx, cy, w, h, cls0_score, cls1_score, ...]
// Transpose ngầm rồi lấy max class score, cuối cùng áp NMS.

/**
 * @brief Parse output YOLO layout chuan (batch, channels, anchors).
 *
 * @param data Con tro dữ liệu float output tensor.
 * @param batch So anh trong batch.
 * @param channels So channel output (4 + num_classes).
 * @param num_anchors So anchor/prediction points.
 * @param infos Metadata letterbox theo tung anh.
 * @param conf_threshold Ngưỡng confidence loc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<std::vector<Detection>> Detection theo batch.
 */
static std::vector<std::vector<Detection>> ParseStandardYOLO(
		const float* data,
		int64_t batch,
		int64_t channels, // 4 + num_classes
		int64_t num_anchors,
		const std::vector<LetterboxInfo>& infos,
		float conf_threshold,
		float nms_iou_threshold) {
	const int64_t num_classes = channels - 4;
	std::vector<std::vector<Detection>> all;
	all.resize(static_cast<size_t>(batch));

	for (int64_t n = 0; n < batch; ++n) {
		// base tro den block [channels, num_anchors] cua sample n.
		const float* base = data + n * channels * num_anchors;
		const auto& info = infos[static_cast<size_t>(n)];
		std::vector<Detection> dets;
		dets.reserve(256);

		for (int64_t a = 0; a < num_anchors; ++a) {
			// Tìm max class score
			float best_score = -1.0f;
			int best_cls = 0;
			for (int64_t c = 0; c < num_classes; ++c) {
				float s = base[(4 + c) * num_anchors + a];
				if (s > best_score) {
					best_score = s;
					best_cls = static_cast<int>(c);
				}
			}

			if (!(best_score >= conf_threshold)) {
				// Cat ngưỡng score trước NMS để giảm so luong box xu ly.
				continue;
			}

			// bbox: cx, cy, w, h
			float cx = base[0 * num_anchors + a];
			float cy = base[1 * num_anchors + a];
			float bw = base[2 * num_anchors + a];
			float bh = base[3 * num_anchors + a];

			// cxcywh → xyxy
			float x1 = cx - bw / 2.0f;
			float y1 = cy - bh / 2.0f;
			float x2 = cx + bw / 2.0f;
			float y2 = cy + bh / 2.0f;

			Detection det;
			det.x1 = x1;
			det.y1 = y1;
			det.x2 = x2;
			det.y2 = y2;
			det.score = best_score;
			det.cls = best_cls;

			Detection mapped = MapBackToOriginal(det, info);
			if (mapped.x2 <= mapped.x1 || mapped.y2 <= mapped.y1) {
				// Loai bo box suy bien sau khi map nguoc.
				continue;
			}
			dets.push_back(mapped);
		}

		all[static_cast<size_t>(n)] = FinalizeDetections(std::move(dets), nms_iou_threshold);
		// NMS mode: ap NMS theo tung anh trong batch.
		// No-NMS mode: chi giu top-1 detection theo score.
	}
	return all;
}

// ── ParseOutput: auto-detect output format ────────────────────────────

/**
 * @brief Parse output tensor YOLO theo nhiều layout có thể gap.
 *
 * @param out0 Tensor output dau tien cua model.
 * @param infos Metadata letterbox theo tung anh input.
 * @param conf_threshold Ngưỡng confidence loc detection.
 * @param nms_iou_threshold Ngưỡng IoU cho NMS.
 * @return std::vector<std::vector<Detection>> Detection theo tung anh.
 */
std::vector<std::vector<Detection>> ParseOutput(
		Ort::Value& out0,
		const std::vector<LetterboxInfo>& infos,
		float conf_threshold,
		float nms_iou_threshold) {
	if (!out0.IsTensor()) {
		throw std::runtime_error("YOLO output[0] không phai tensor");
	}
	auto type_info = out0.GetTensorTypeAndShapeInfo();
	const auto shape = type_info.GetShape();
	const auto elem_type = type_info.GetElementType();

	int64_t batch = 1;
	int64_t dim1 = 0;
	int64_t dim2 = 0;
	if (shape.size() == 3) {
		batch = shape[0];
		dim1 = shape[1];
		dim2 = shape[2];
	} else if (shape.size() == 2) {
		batch = 1;
		dim1 = shape[0];
		dim2 = shape[1];
	} else {
		throw std::runtime_error("YOLO output shape không hop le (cần rank 2/3), rank=" + std::to_string(shape.size()));
	}

	if (static_cast<size_t>(batch) != infos.size()) {
		// infos được tao tu preprocess theo so anh input, phai khớp batch output.
		throw std::runtime_error("YOLO output batch != so anh input");
	}

	std::vector<float> fp32_buffer;
	const float* data = nullptr;
	if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		// data tro truc tiep vào bộ nhớ tensor float32 cua ORT.
		data = out0.GetTensorData<float>();
	} else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
		const auto* fp16 = out0.GetTensorData<Ort::Float16_t>();
		size_t total_values = 1;
		for (int64_t d : shape) {
			if (d <= 0) {
				throw std::runtime_error("YOLO output shape chưa xác định được kích thước");
			}
			total_values *= static_cast<size_t>(d);
		}
		fp32_buffer.resize(total_values);
		for (size_t i = 0; i < total_values; ++i) {
			// Convert fp16 -> fp32 để dùng chung parser float.
			fp32_buffer[i] = static_cast<float>(fp16[i]);
		}
		data = fp32_buffer.data();
	} else {
		throw std::runtime_error("YOLO output type chưa hỗ trợ (chi hỗ trợ float32/float16)");
	}

	// Standard YOLO: (batch, 4+num_classes, num_anchors)
	//   dim1 nhỏ (vd: 6 cho 2 class), dim2 lớn (vd: 8400)
	const bool is_standard_yolo = (dim1 < dim2 && dim1 >= 5);

	if (is_standard_yolo) {
		// Fast-path cho layout output pho bien (batch, channels, anchors).
		return ParseStandardYOLO(data, batch, dim1, dim2, infos, conf_threshold, nms_iou_threshold);
	}

	// Fallback: đảo dim nếu dim2 nhỏ hơn dim1 nhưng >= 5
	// (trường hợp model output (batch, num_anchors, 4+num_classes))
	if (dim2 >= 5 && dim1 > dim2) {
		// Coi như (batch, num_anchors, channels) → cần transpose
		// Parse mỗi row trực tiếp
		const int64_t num_anchors = dim1;
		const int64_t channels = dim2;
		const int64_t num_classes = channels - 4;

		std::vector<std::vector<Detection>> all;
		all.resize(static_cast<size_t>(batch));

		for (int64_t n = 0; n < batch; ++n) {
			// bdata tro den block [num_anchors, channels] cua sample n.
			const float* bdata = data + n * num_anchors * channels;
			const auto& info = infos[static_cast<size_t>(n)];
			std::vector<Detection> dets;
			dets.reserve(256);

			for (int64_t a = 0; a < num_anchors; ++a) {
				const float* row = bdata + a * channels;
				// Layout nay da dạng row-major theo anchor, không cần transpose thật sự.
				float best_score = -1.0f;
				int best_cls = 0;
				for (int64_t c = 0; c < num_classes; ++c) {
					float s = row[4 + c];
					if (s > best_score) {
						best_score = s;
						best_cls = static_cast<int>(c);
					}
				}
				if (!(best_score >= conf_threshold)) {
					continue;
				}

				float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
				Detection det;
				det.x1 = cx - bw / 2.0f;
				det.y1 = cy - bh / 2.0f;
				det.x2 = cx + bw / 2.0f;
				det.y2 = cy + bh / 2.0f;
				det.score = best_score;
				det.cls = best_cls;

				Detection mapped = MapBackToOriginal(det, info);
				if (mapped.x2 <= mapped.x1 || mapped.y2 <= mapped.y1) {
					continue;
				}
				dets.push_back(mapped);
			}

			all[static_cast<size_t>(n)] = FinalizeDetections(std::move(dets), nms_iou_threshold);
		}
		return all;
	}

	throw std::runtime_error(
		"YOLO output shape không nhan dạng được: dim1=" + std::to_string(dim1) +
		", dim2=" + std::to_string(dim2));
}

} // namespace detail
} // namespace yolo_detector
