#include "ocrplate/services/yolo_detector_internal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo_detector {
namespace detail {

// ── Geometry helpers ──────────────────────────────────────────────────

Detection MapBackToOriginal(const Detection& in, const LetterboxInfo& info) {
	Detection out = in;
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
				continue;
			}
			dets.push_back(mapped);
		}

		all[static_cast<size_t>(n)] = ApplyNMS(std::move(dets), nms_iou_threshold);
	}
	return all;
}

// ── ParseOutput: auto-detect output format ────────────────────────────

std::vector<std::vector<Detection>> ParseOutput(
		Ort::Value& out0,
		const std::vector<LetterboxInfo>& infos,
		float conf_threshold,
		float nms_iou_threshold) {
	if (!out0.IsTensor()) {
		throw std::runtime_error("YOLO output[0] khong phai tensor");
	}
	auto type_info = out0.GetTensorTypeAndShapeInfo();
	const auto shape = type_info.GetShape();
	const auto elem_type = type_info.GetElementType();

	if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		throw std::runtime_error("YOLO output type chua ho tro (chi ho tro float32)");
	}

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
		throw std::runtime_error("YOLO output shape khong hop le (can rank 2/3), rank=" + std::to_string(shape.size()));
	}

	if (static_cast<size_t>(batch) != infos.size()) {
		throw std::runtime_error("YOLO output batch != so anh input");
	}

	const float* data = out0.GetTensorData<float>();

	// Standard YOLO: (batch, 4+num_classes, num_anchors)
	//   dim1 nhỏ (vd: 6 cho 2 class), dim2 lớn (vd: 8400)
	const bool is_standard_yolo = (dim1 < dim2 && dim1 >= 5);

	if (is_standard_yolo) {
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
			const float* bdata = data + n * num_anchors * channels;
			const auto& info = infos[static_cast<size_t>(n)];
			std::vector<Detection> dets;
			dets.reserve(256);

			for (int64_t a = 0; a < num_anchors; ++a) {
				const float* row = bdata + a * channels;
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

			all[static_cast<size_t>(n)] = ApplyNMS(std::move(dets), nms_iou_threshold);
		}
		return all;
	}

	throw std::runtime_error(
		"YOLO output shape khong nhan dang duoc: dim1=" + std::to_string(dim1) +
		", dim2=" + std::to_string(dim2));
}

} // namespace detail
} // namespace yolo_detector
