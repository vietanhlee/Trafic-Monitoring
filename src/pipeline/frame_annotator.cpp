/*
 * Mo ta file: Trien khai pipeline annotate frame: detect xe, detect bien, OCR, brand, tracking.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/pipeline/frame_annotator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ocrplate/core/app_config.h"
#include "ocrplate/services/brand_classifier.h"
#include "ocrplate/services/ocr_batch.h"
#include "ocrplate/utils/plate_parallel.h"
#include "ocrplate/tracking/vehicle_identity_store.h"
#include "ocrplate/services/yolo_detector.h"
#include "ocrplate/pipeline/track_trace.h"

namespace {

// Anh xa id lop xe sang ten hien thi.
const char* VehicleClassName(int cls) {
	switch (cls) {
	case 0:
		return "car";
	case 1:
		return "motor";
	default:
		return "unknown";
	}
}

// Anh xa id thuong hieu sang ten brand de hien thi.
const char* BrandClassName(int brand_id) {
	switch (brand_id) {
	case 0: return "BMW";
	case 1: return "ChengLong";
	case 2: return "DongFeng";
	case 3: return "Ford";
	case 4: return "Hino";
	case 5: return "Honda";
	case 6: return "Howo";
	case 7: return "Hyundai";
	case 8: return "Isuzu";
	case 9: return "Jac";
	case 10: return "Kia";
	case 11: return "KimLong";
	case 12: return "Lexus";
	case 13: return "Mazda";
	case 14: return "Mercedes-Benz";
	case 15: return "Mitsubishi";
	case 16: return "Other";
	case 17: return "Peugeot";
	case 18: return "Samco";
	case 19: return "Suzuki";
	case 20: return "Thaco";
	case 21: return "Toyota";
	case 22: return "VinFast";
	default: return "UnknownBrand";
	}
}

// Chuan hoa bounding box ve mien anh hop le, tranh out-of-bound.
cv::Rect ToRectClamped(float x1, float y1, float x2, float y2, int w, int h) {
	int ix1 = std::max(0, std::min(static_cast<int>(std::floor(x1)), w - 1));
	int iy1 = std::max(0, std::min(static_cast<int>(std::floor(y1)), h - 1));
	int ix2 = std::max(0, std::min(static_cast<int>(std::ceil(x2)), w - 1));
	int iy2 = std::max(0, std::min(static_cast<int>(std::ceil(y2)), h - 1));
	int rw = std::max(0, ix2 - ix1);
	int rh = std::max(0, iy2 - iy1);
	return cv::Rect(ix1, iy1, rw, rh);
}

bool IsPointAboveGateLine(const cv::Point& p, const cv::Point& p1, const cv::Point& p2) {
	// Chuan hoa huong line theo truc x de quy uoc "phia tren" on dinh.
	cv::Point a = p1;
	cv::Point b = p2;
	if (a.x > b.x || (a.x == b.x && a.y > b.y)) {
		std::swap(a, b);
	}

	const long long dx = static_cast<long long>(b.x) - static_cast<long long>(a.x);
	const long long dy = static_cast<long long>(b.y) - static_cast<long long>(a.y);
	if (dx == 0) {
		// Truong hop line dung: quy uoc "tren" la y nho hon diem cao hon cua line.
		return p.y < std::min(a.y, b.y);
	}

	const long long cross = dx * (static_cast<long long>(p.y) - static_cast<long long>(a.y))
		- dy * (static_cast<long long>(p.x) - static_cast<long long>(a.x));
	// He toa do anh: y huong xuong duoi, nen cross < 0 tuong ung nam phia tren line.
	return cross < 0;
}


// Ve bounding box bien so + nhan text OCR va do tin cay.
void DrawPlate(cv::Mat& bgr, const yolo_detector::Detection& det, const std::string& text, float conf_avg) {
	const bool is_low_ocr_conf = (conf_avg < app_config::kOcrConfAvgThresh);
	const cv::Scalar color = is_low_ocr_conf ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
	const cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
	const cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
	cv::rectangle(bgr, p1, p2, color, 2);

	char buf[256];
	std::snprintf(
		buf,
		sizeof(buf),
		"%s p%.2f o%.2f",
		text.empty() ? "?" : text.c_str(),
		det.score,
		conf_avg);
	std::string label = buf;
	if (label.empty()) {
		label = "?";
	}
	int baseline = 0;
	const double font_scale = 0.8;
	const int thickness = 2;
	const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const int x = std::max(0, p1.x);
	const int y = std::max(sz.height + 2, p1.y);
	cv::rectangle(bgr,
		cv::Rect(x, y - sz.height - 2, sz.width + 4, sz.height + baseline + 4),
		color,
		cv::FILLED);
	cv::putText(bgr, label, cv::Point(x + 2, y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), thickness);
}

// Ve bounding box phuong tien + thong tin track, brand va plate da chap nhan.
void DrawVehicle(
	cv::Mat& bgr,
	const yolo_detector::Detection& det,
	int track_id,
	bool has_plate,
	int brand_id,
	const std::string& accepted_plate_text) {
	const bool is_motor = (det.cls == 1);
	const cv::Scalar color = is_motor
		? (has_plate ? cv::Scalar(255, 0, 255) : cv::Scalar(170, 0, 170))
		: (has_plate ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255));
	const cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
	const cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
	cv::rectangle(bgr, p1, p2, color, 2);

	char line1_buf[256];
	if (track_id > 0) {
		if (brand_id >= 0) {
			std::snprintf(line1_buf, sizeof(line1_buf), "ID:%d %s %s %.2f", track_id, VehicleClassName(det.cls), BrandClassName(brand_id), det.score);
		} else {
			std::snprintf(line1_buf, sizeof(line1_buf), "ID:%d %s %.2f", track_id, VehicleClassName(det.cls), det.score);
		}
	} else {
		if (brand_id >= 0) {
			std::snprintf(line1_buf, sizeof(line1_buf), "%s %s %.2f", VehicleClassName(det.cls), BrandClassName(brand_id), det.score);
		} else {
			std::snprintf(line1_buf, sizeof(line1_buf), "%s %.2f", VehicleClassName(det.cls), det.score);
		}
	}

	char line2_buf[256];
	if (!accepted_plate_text.empty()) {
		std::snprintf(line2_buf, sizeof(line2_buf), "Plate: %s", accepted_plate_text.c_str());
	} else {
		std::snprintf(line2_buf, sizeof(line2_buf), "Plate: ?");
	}

	const std::string line1 = line1_buf;
	const std::string line2 = line2_buf;
	int baseline = 0;
	const double font_scale = 0.7;
	const int thickness = 2;
	const auto sz1 = cv::getTextSize(line1, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const auto sz2 = cv::getTextSize(line2, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const int line_gap = 4;
	const int text_w = std::max(sz1.width, sz2.width);
	const int text_h = sz1.height + sz2.height + line_gap;
	const int x = std::max(0, p1.x);
	const int y = std::max(text_h + 2, p1.y);
	cv::rectangle(bgr,
		cv::Rect(x, y - text_h - 2, text_w + 6, text_h + baseline + 6),
		color,
		cv::FILLED);
	const int line1_y = y - sz2.height - line_gap;
	const int line2_y = y;
	cv::putText(bgr, line1, cv::Point(x + 2, line1_y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
	cv::putText(bgr, line2, cv::Point(x + 2, line2_y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
}

struct PlatePipelineResult {
	std::vector<ocr_batch::OcrText> texts;
	std::vector<yolo_detector::Detection> plate_boxes_in_image;
	std::vector<size_t> plate_vehicle_indices;
	std::vector<bool> vehicle_has_plate;
	bool has_any_plate = false;
};

// Chay nhanh detect plate + OCR tren danh sach crop xe, tra ve ket qua da map ve anh goc.
PlatePipelineResult RunPlatePipeline(
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	const std::vector<cv::Mat>& vehicle_crops,
	const std::vector<cv::Rect>& vehicle_rects) {
	PlatePipelineResult result;
	result.vehicle_has_plate.assign(vehicle_rects.size(), false);

	// 1) Detect bien so trong tung crop phuong tien (da luong o plate_parallel).
	auto plates_per_vehicle = plate_parallel::DetectPlatesPerVehicleParallel(
		plate_sess,
		vehicle_crops,
		app_config::kPlateConfThresh,
		app_config::kPlateNmsIouThresh);

	// 2) Loc box bien so hop le va quy doi ve candidate OCR.
	std::vector<plate_parallel::PlateCandidate> candidates = plate_parallel::BuildPlateCandidatesParallel(
		plates_per_vehicle,
		vehicle_crops,
		app_config::kPlateConfThresh);

	auto map_work = [&]() {
		// Map box bien so tu toa do trong crop xe ve toa do tren anh goc.
		std::vector<yolo_detector::Detection> mapped;
		std::vector<size_t> mapped_vehicle_indices;
		mapped.reserve(candidates.size());
		mapped_vehicle_indices.reserve(candidates.size());
		std::vector<bool> has_plate(vehicle_rects.size(), false);
		for (const auto& c : candidates) {
			const cv::Rect& vr = vehicle_rects[c.vehicle_index];
			yolo_detector::Detection in_img = c.plate_in_vehicle;
			in_img.x1 += static_cast<float>(vr.x);
			in_img.y1 += static_cast<float>(vr.y);
			in_img.x2 += static_cast<float>(vr.x);
			in_img.y2 += static_cast<float>(vr.y);
			mapped.push_back(std::move(in_img));
			mapped_vehicle_indices.push_back(c.vehicle_index);
			has_plate[c.vehicle_index] = true;
		}
		return std::make_tuple(std::move(mapped), std::move(mapped_vehicle_indices), std::move(has_plate));
	};

	auto crop_work = [&]() {
		// Crop + preprocess OCR input (RGB uint8 HWC).
		return plate_parallel::PreprocessPlatesParallel(candidates, vehicle_crops, app_config::kInputW, app_config::kInputH);
	};

	std::vector<cv::Mat> plate_rgb_ocr;
	if (candidates.size() < 4) {
		// Workload nho: chay tuan tu de tranh chi phi tao thread.
		auto mapped_out = map_work();
		result.plate_boxes_in_image = std::move(std::get<0>(mapped_out));
		result.plate_vehicle_indices = std::move(std::get<1>(mapped_out));
		result.vehicle_has_plate = std::move(std::get<2>(mapped_out));
		plate_rgb_ocr = crop_work();
	} else {
		auto map_future = std::async(std::launch::async, map_work);
		auto crop_future = std::async(std::launch::async, crop_work);
		auto mapped_out = map_future.get();
		result.plate_boxes_in_image = std::move(std::get<0>(mapped_out));
		result.plate_vehicle_indices = std::move(std::get<1>(mapped_out));
		result.vehicle_has_plate = std::move(std::get<2>(mapped_out));
		plate_rgb_ocr = crop_future.get();
	}

	if (plate_rgb_ocr.empty()) {
		result.has_any_plate = false;
		return result;
	}

	result.texts = ocr_batch::RunBatch(ocr_sess, plate_rgb_ocr, app_config::kAlphabet);
	if (result.texts.size() != result.plate_boxes_in_image.size()) {
		throw std::runtime_error("Loi noi bo: so OCR text != so plate boxes");
	}
	result.has_any_plate = true;
	return result;
}

} // namespace

// Khoi tao context tracking (ByteTrack + identity store) voi cau hinh tu app_config.
TrackingRuntimeContext::TrackingRuntimeContext()
	: tracker(
		app_config::kTrackerIouThreshold,
		app_config::kTrackerMaxMissedFrames,
		app_config::kTrackerMinConfirmedHits,
		app_config::kTrackerHighScoreThreshold,
		app_config::kTrackerLowScoreThreshold,
		app_config::kTrackerIouThresholdLow),
	  identity_store(
		  app_config::kTrackBrandAcceptConf,
		  app_config::kTrackBrandMaxAttempts,
		  app_config::kTrackPlateOcrAcceptConf,
		  app_config::kTrackPlateMaxDetectAttempts,
		  app_config::kTrackPlateMaxOcrAttempts,
		  app_config::kTrackPlateUnknownText,
		  app_config::kTrackPlateNoPlateText,
		  app_config::kPlateTextMinLen,
		  app_config::kPlateTextMaxLen) {}

// Ve FPS hien tai len goc trai frame.
void DrawFps(cv::Mat& bgr, double fps) {
	char fps_buf[64];
	std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.2f", fps);
	const std::string label = fps_buf;

	int baseline = 0;
	const double font_scale = 0.8;
	const int thickness = 2;
	const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const int margin = 8;
	const int x = margin;
	const int y = margin + sz.height;

	cv::rectangle(
		bgr,
		cv::Rect(x - 4, y - sz.height - 4, sz.width + 8, sz.height + baseline + 8),
		cv::Scalar(0, 0, 0),
		cv::FILLED);
	cv::putText(
		bgr,
		label,
		cv::Point(x, y + baseline / 2),
		cv::FONT_HERSHEY_SIMPLEX,
		font_scale,
		cv::Scalar(0, 255, 255),
		thickness);
}

// Infer 1 frame va dong goi ket qua ve vao overlay de co the draw/caching.
bool InferFrameOverlay(
	const cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	FrameOverlayResult& out_overlay,
	bool verbose,
	TrackingRuntimeContext* tracking_ctx) {
	out_overlay.vehicles.clear();
	out_overlay.plates.clear();
	out_overlay.traces.clear();

	auto vehicles_batch = yolo_detector::RunBatch(vehicle_sess, {bgr}, app_config::kVehicleConfThresh, app_config::kNmsIouThresh);
	const auto& vehicles = vehicles_batch.at(0);
	if (vehicles.empty()) {
		if (verbose) {
			std::cout << "Khong phat hien phuong tien\n";
		}
		return false;
	}
	if (verbose) {
		for (size_t i = 0; i < vehicles.size(); ++i) {
			std::cout
				<< "vehicle " << i
				<< ": cls=" << vehicles[i].cls << "(" << VehicleClassName(vehicles[i].cls) << ")"
				<< " conf=" << vehicles[i].score
				<< " box=[" << vehicles[i].x1 << "," << vehicles[i].y1 << "," << vehicles[i].x2 << "," << vehicles[i].y2 << "]\n";
		}
	}

	std::vector<cv::Mat> vehicle_crops;
	std::vector<cv::Rect> vehicle_rects;
	std::vector<yolo_detector::Detection> vehicles_used;
	vehicle_crops.reserve(vehicles.size());
	vehicle_rects.reserve(vehicles.size());
	vehicles_used.reserve(vehicles.size());
	for (const auto& v : vehicles) {
		cv::Rect r = ToRectClamped(v.x1, v.y1, v.x2, v.y2, bgr.cols, bgr.rows);
		if (r.width <= 2 || r.height <= 2) {
			// Bo qua bbox qua nho vi OCR/plate detect thuong khong on dinh.
			continue;
		}
		vehicle_rects.push_back(r);
		// Tối ưu: dùng ROI view thay vì clone (giảm copy bộ nhớ).
		vehicle_crops.push_back(bgr(r));
		vehicles_used.push_back(v);
	}
	if (vehicle_crops.empty()) {
		if (verbose) {
			std::cout << "Khong co crop phuong tien hop le\n";
		}
		return false;
	}

	std::vector<int> track_ids(vehicles_used.size(), -1);
	if (tracking_ctx != nullptr) {
		// Tracking cap ID ben vung theo frame, giup hop nhat plate/brand theo thoi gian.
		track_ids = tracking_ctx->tracker.Update(vehicles_used);
	}

	std::vector<bool> allow_predict_by_line(vehicles_used.size(), true);
	if (tracking_ctx != nullptr && tracking_ctx->enable_predict_on_line_cross) {
		for (size_t i = 0; i < vehicles_used.size(); ++i) {
			const auto& det = vehicles_used[i];
			const cv::Point center(
				static_cast<int>(std::lround((det.x1 + det.x2) * 0.5f)),
				static_cast<int>(std::lround((det.y1 + det.y2) * 0.5f)));
			allow_predict_by_line[i] = IsPointAboveGateLine(
				center,
				tracking_ctx->gate_line_p1,
				tracking_ctx->gate_line_p2);
		}
	}

	std::vector<size_t> need_brand_indices;
	std::vector<cv::Mat> need_brand_crops;
	std::vector<size_t> need_plate_indices;
	std::vector<cv::Mat> need_plate_crops;
	std::vector<cv::Rect> need_plate_rects;
	need_brand_indices.reserve(vehicles_used.size());
	need_plate_indices.reserve(vehicles_used.size());

	for (size_t i = 0; i < vehicles_used.size(); ++i) {
		const bool allow_predict = allow_predict_by_line[i];
		if (vehicles_used[i].cls == 0) {
			// Brand chi ap dung cho car va bo qua neu track da co ket qua accepted.
			const bool has_brand = (tracking_ctx != nullptr) && tracking_ctx->identity_store.HasBrandResolved(track_ids[i]);
			if (allow_predict && !has_brand) {
				need_brand_indices.push_back(i);
				need_brand_crops.push_back(vehicle_crops[i]);
			}
		}

		const bool has_plate = (tracking_ctx != nullptr) && tracking_ctx->identity_store.HasPlateAccepted(track_ids[i]);
		if (allow_predict && !has_plate) {
			// Chi infer plate/OCR cho track chua co bien so accepted de tiet kiem tai nguyen.
			need_plate_indices.push_back(i);
			need_plate_crops.push_back(vehicle_crops[i]);
			need_plate_rects.push_back(vehicle_rects[i]);
		}
	}

	std::vector<brand_classifier::BrandResult> car_brand_results;
	PlatePipelineResult plate_result;
	if (!need_brand_crops.empty() && !need_plate_crops.empty()) {
		// Chay song song 2 nhanh doc lap de rut ngan latency frame:
		// - nhanh brand classify (chi xe hoi)
		// - nhanh plate detect + OCR
		auto brand_future = std::async(std::launch::async, [&]() {
			return brand_classifier::ClassifyBatch(
				brand_sess,
				need_brand_crops,
				app_config::kBrandInputH,
				app_config::kBrandInputW);
		});
		auto plate_future = std::async(std::launch::async, [&]() {
			return RunPlatePipeline(plate_sess, ocr_sess, need_plate_crops, need_plate_rects);
		});
		car_brand_results = brand_future.get();
		plate_result = plate_future.get();
	} else if (!need_brand_crops.empty()) {
		car_brand_results = brand_classifier::ClassifyBatch(
			brand_sess,
			need_brand_crops,
			app_config::kBrandInputH,
			app_config::kBrandInputW);
	} else if (!need_plate_crops.empty()) {
		plate_result = RunPlatePipeline(plate_sess, ocr_sess, need_plate_crops, need_plate_rects);
	}

	std::vector<int> brand_id_per_vehicle(vehicles_used.size(), -1);
	for (size_t k = 0; k < need_brand_indices.size() && k < car_brand_results.size(); ++k) {
		const size_t vehicle_idx = need_brand_indices[k];
		// Map ket qua classify ve dung vi tri xe ban dau.
		brand_id_per_vehicle[vehicle_idx] = car_brand_results[k].class_id;
		if (tracking_ctx != nullptr) {
			tracking_ctx->identity_store.UpdateBrand(track_ids[vehicle_idx], car_brand_results[k].class_id, car_brand_results[k].conf);
		}
	}

	std::vector<bool> vehicle_has_plate(vehicles_used.size(), false);
	for (size_t sub_idx = 0; sub_idx < need_plate_indices.size(); ++sub_idx) {
		const size_t global_idx = need_plate_indices[sub_idx];
		if (sub_idx < plate_result.vehicle_has_plate.size()) {
			// Map bool has_plate tu index subset ve index toan bo vehicles_used.
			vehicle_has_plate[global_idx] = plate_result.vehicle_has_plate[sub_idx];
		}
		if (tracking_ctx != nullptr && !vehicle_has_plate[global_idx]) {
			// Khong detect duoc plate o frame nay => van tinh 1 lan thu cho track.
			tracking_ctx->identity_store.MarkPlateMiss(track_ids[global_idx]);
		}
	}

	for (size_t plate_idx = 0; plate_idx < plate_result.texts.size(); ++plate_idx) {
		if (plate_idx >= plate_result.plate_vehicle_indices.size()) {
			continue;
		}
		const size_t sub_vehicle_idx = plate_result.plate_vehicle_indices[plate_idx];
		if (sub_vehicle_idx >= need_plate_indices.size()) {
			continue;
		}
		const size_t global_vehicle_idx = need_plate_indices[sub_vehicle_idx];
		if (tracking_ctx != nullptr) {
			// Luu tich luy ket qua OCR vao identity store de chap nhan theo nhieu lan do.
			tracking_ctx->identity_store.UpdatePlate(
				track_ids[global_vehicle_idx],
				plate_result.texts[plate_idx].text,
				plate_result.plate_boxes_in_image[plate_idx].score,
				plate_result.texts[plate_idx].conf_avg);
		}
	}

	if (verbose) {
		for (size_t i = 0; i < vehicles_used.size(); ++i) {
			int bid = brand_id_per_vehicle[i];
			if (tracking_ctx != nullptr) {
				const auto* iden = tracking_ctx->identity_store.Get(track_ids[i]);
				if (iden != nullptr && iden->brand_accepted) {
					bid = iden->brand_id;
				}
			}
			if (bid >= 0) {
				std::cout
					<< "vehicle " << i
					<< " (track=" << track_ids[i] << ")"
					<< ": brand=" << bid << "(" << BrandClassName(bid) << ")"
					<< "\n";
			}
		}
	}

	for (size_t i = 0; i < vehicles_used.size(); ++i) {
		const auto* iden = (tracking_ctx != nullptr) ? tracking_ctx->identity_store.Get(track_ids[i]) : nullptr;
		VehicleOverlayResult v;
		v.det = vehicles_used[i];
		// Luon hien ID tu dau neu dang bat tracking; thuoc tinh (bien so/hang) se cap nhat dan.
		v.track_id = (tracking_ctx != nullptr) ? track_ids[i] : -1;
		v.has_plate = vehicle_has_plate[i] || (iden != nullptr && iden->plate_accepted);
		v.brand_id = (iden != nullptr && iden->brand_accepted) ? iden->brand_id : brand_id_per_vehicle[i];
		if (iden != nullptr && iden->plate_accepted) {
			// Hien text da chap nhan (on dinh hon text OCR tung frame).
			v.accepted_plate_text = iden->plate_text;
		}
		out_overlay.vehicles.push_back(v);
	}

	// Cap nhat trace (lich su tam bbox) theo track_id va dua vao overlay de co the cache/draw lai.
	if (tracking_ctx != nullptr) {
		UpdateTrackTraces(bgr, out_overlay.vehicles, *tracking_ctx, out_overlay.traces);
	}

	for (size_t i = 0; i < plate_result.texts.size(); ++i) {
		if (i >= plate_result.plate_vehicle_indices.size()) {
			continue;
		}
		const size_t sub_vehicle_idx = plate_result.plate_vehicle_indices[i];
		if (sub_vehicle_idx >= need_plate_indices.size()) {
			continue;
		}
		const size_t global_vehicle_idx = need_plate_indices[sub_vehicle_idx];
		PlateOverlayResult p;
		p.det = plate_result.plate_boxes_in_image[i];
		// Track ID gan theo xe cha de de truy vet tren UI/log.
		p.track_id = track_ids[global_vehicle_idx];
		p.text = plate_result.texts[i].text;
		p.conf_avg = plate_result.texts[i].conf_avg;
		out_overlay.plates.push_back(std::move(p));
		if (verbose) {
			std::cout
				<< "plate " << i
				<< " (track=" << track_ids[global_vehicle_idx] << ")"
				<< ": text=" << plate_result.texts[i].text
				<< " plate_conf=" << plate_result.plate_boxes_in_image[i].score
				<< " ocr_conf_avg=" << plate_result.texts[i].conf_avg
				<< " box=[" << plate_result.plate_boxes_in_image[i].x1 << "," << plate_result.plate_boxes_in_image[i].y1 << "," << plate_result.plate_boxes_in_image[i].x2 << "," << plate_result.plate_boxes_in_image[i].y2 << "]\n";
		}
	}

	if (verbose && tracking_ctx != nullptr) {
		const auto snapshot = tracking_ctx->identity_store.Snapshot();
		for (const auto& one : snapshot) {
			std::cout
				<< "map id=" << one.track_id
				<< " brand=" << (one.brand_accepted ? BrandClassName(one.brand_id) : (one.brand_forced_unknown ? "unknown" : "?"))
				<< " brand_attempts=" << one.brand_attempts
				<< " brand_forced_unknown=" << (one.brand_forced_unknown ? "yes" : "no")
				<< " plate=" << (one.plate_accepted ? one.plate_text : "?")
				<< " plate_detect_attempts=" << one.plate_detect_attempts
				<< " plate_ocr_attempts=" << one.plate_ocr_attempts
				<< " plate_forced_no_plate=" << (one.plate_forced_no_plate ? "yes" : "no")
				<< " plate_forced_unknown=" << (one.plate_forced_unknown ? "yes" : "no")
				<< " done=" << (one.IsComplete() ? "yes" : "no")
				<< "\n";
		}
	}

	return true;
}

// Ve toan bo overlay (trace, xe, bien so) len frame.
void DrawFrameOverlay(cv::Mat& bgr, const FrameOverlayResult& overlay) {
	DrawTrackTraces(bgr, overlay.traces);
	for (const auto& v : overlay.vehicles) {
		DrawVehicle(bgr, v.det, v.track_id, v.has_plate, v.brand_id, v.accepted_plate_text);
	}
	for (const auto& p : overlay.plates) {
		DrawPlate(bgr, p.det, p.text, p.conf_avg);
	}
}

// API muc cao: infer + ve truc tiep len frame dau vao.
bool AnnotateFrame(
	cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool verbose,
	TrackingRuntimeContext* tracking_ctx) {
	FrameOverlayResult overlay;
	const bool detected = InferFrameOverlay(
		bgr,
		vehicle_sess,
		plate_sess,
		ocr_sess,
		brand_sess,
		overlay,
		verbose,
		tracking_ctx);
	if (detected) {
		DrawFrameOverlay(bgr, overlay);
	}
	return detected;
}
