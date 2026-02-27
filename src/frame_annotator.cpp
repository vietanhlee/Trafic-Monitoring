#include "frame_annotator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "app_config.h"
#include "brand_classifier.h"
#include "image_preprocess.h"
#include "ocr_batch.h"
#include "yolo_detector.h"

namespace {

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

cv::Rect ToRectClamped(float x1, float y1, float x2, float y2, int w, int h) {
	int ix1 = std::max(0, std::min(static_cast<int>(std::floor(x1)), w - 1));
	int iy1 = std::max(0, std::min(static_cast<int>(std::floor(y1)), h - 1));
	int ix2 = std::max(0, std::min(static_cast<int>(std::ceil(x2)), w - 1));
	int iy2 = std::max(0, std::min(static_cast<int>(std::ceil(y2)), h - 1));
	int rw = std::max(0, ix2 - ix1);
	int rh = std::max(0, iy2 - iy1);
	return cv::Rect(ix1, iy1, rw, rh);
}

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

void DrawVehicle(cv::Mat& bgr, const yolo_detector::Detection& det, bool has_plate, int brand_id) {
	const cv::Scalar color = has_plate ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
	const cv::Point p1(static_cast<int>(det.x1), static_cast<int>(det.y1));
	const cv::Point p2(static_cast<int>(det.x2), static_cast<int>(det.y2));
	cv::rectangle(bgr, p1, p2, color, 2);

	char buf[128];
	if (brand_id >= 0) {
		std::snprintf(buf, sizeof(buf), "%s %s %.2f", VehicleClassName(det.cls), BrandClassName(brand_id), det.score);
	} else {
		std::snprintf(buf, sizeof(buf), "%s %.2f", VehicleClassName(det.cls), det.score);
	}
	const std::string label = buf;
	int baseline = 0;
	const double font_scale = 0.7;
	const int thickness = 2;
	const auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
	const int x = std::max(0, p1.x);
	const int y = std::max(sz.height + 2, p1.y);
	cv::rectangle(bgr,
		cv::Rect(x, y - sz.height - 2, sz.width + 4, sz.height + baseline + 4),
		color,
		cv::FILLED);
	cv::putText(bgr, label, cv::Point(x + 2, y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
}

struct PlatePipelineResult {
	std::vector<ocr_batch::OcrText> texts;
	std::vector<yolo_detector::Detection> plate_boxes_in_image;
	std::vector<bool> vehicle_has_plate;
	bool has_any_plate = false;
};

struct PlateCandidate {
	size_t vehicle_index = 0;
	yolo_detector::Detection plate_in_vehicle;
	cv::Rect plate_rect_in_vehicle;
};

PlatePipelineResult RunPlatePipeline(
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	const std::vector<cv::Mat>& vehicle_crops,
	const std::vector<cv::Rect>& vehicle_rects) {
	PlatePipelineResult result;
	result.vehicle_has_plate.assign(vehicle_rects.size(), false);

	auto plates_per_vehicle = yolo_detector::RunBatch(plate_sess, vehicle_crops, app_config::kPlateConfThresh, app_config::kNmsIouThresh);
	std::vector<PlateCandidate> candidates;
	for (size_t i = 0; i < plates_per_vehicle.size(); ++i) {
		const auto& dets = plates_per_vehicle[i];
		for (const auto& p : dets) {
			cv::Rect pr_local = ToRectClamped(
				p.x1,
				p.y1,
				p.x2,
				p.y2,
				vehicle_crops[i].cols,
				vehicle_crops[i].rows);
			if (pr_local.width <= 2 || pr_local.height <= 2) {
				continue;
			}
			PlateCandidate c;
			c.vehicle_index = i;
			c.plate_in_vehicle = p;
			c.plate_rect_in_vehicle = pr_local;
			candidates.push_back(std::move(c));
		}
	}

	auto map_future = std::async(std::launch::async, [&]() {
		std::vector<yolo_detector::Detection> mapped;
		mapped.reserve(candidates.size());
		std::vector<bool> has_plate(vehicle_rects.size(), false);
		for (const auto& c : candidates) {
			const cv::Rect& vr = vehicle_rects[c.vehicle_index];
			yolo_detector::Detection in_img = c.plate_in_vehicle;
			in_img.x1 += static_cast<float>(vr.x);
			in_img.y1 += static_cast<float>(vr.y);
			in_img.x2 += static_cast<float>(vr.x);
			in_img.y2 += static_cast<float>(vr.y);
			mapped.push_back(std::move(in_img));
			has_plate[c.vehicle_index] = true;
		}
		return std::make_pair(std::move(mapped), std::move(has_plate));
	});

	auto crop_future = std::async(std::launch::async, [&]() {
		std::vector<cv::Mat> plate_rgb_ocr;
		plate_rgb_ocr.reserve(candidates.size());
		for (const auto& c : candidates) {
			const cv::Mat& vehicle_bgr = vehicle_crops[c.vehicle_index];
			cv::Mat plate_bgr = vehicle_bgr(c.plate_rect_in_vehicle);
			cv::Mat plate_rgb = image_preprocess::PreprocessMatRgbU8Hwc(plate_bgr, app_config::kInputW, app_config::kInputH);
			plate_rgb_ocr.push_back(std::move(plate_rgb));
		}
		return plate_rgb_ocr;
	});

	auto mapped_out = map_future.get();
	result.plate_boxes_in_image = std::move(mapped_out.first);
	result.vehicle_has_plate = std::move(mapped_out.second);
	std::vector<cv::Mat> plate_rgb_ocr = crop_future.get();

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

bool InferFrameOverlay(
	const cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	FrameOverlayResult& out_overlay,
	bool verbose) {
	out_overlay.vehicles.clear();
	out_overlay.plates.clear();

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
		const float box_h = v.y2 - v.y1;
		const float expand_y2 = v.y2 + box_h * 0.05f;
		cv::Rect r = ToRectClamped(v.x1, v.y1, v.x2, expand_y2, bgr.cols, bgr.rows);
		if (r.width <= 2 || r.height <= 2) {
			continue;
		}
		vehicle_rects.push_back(r);
		vehicle_crops.push_back(bgr(r).clone());
		yolo_detector::Detection v_expanded = v;
		v_expanded.y2 = std::min(expand_y2, static_cast<float>(bgr.rows));
		vehicles_used.push_back(v_expanded);
	}
	if (vehicle_crops.empty()) {
		if (verbose) {
			std::cout << "Khong co crop phuong tien hop le\n";
		}
		return false;
	}

	auto brand_future = std::async(std::launch::async, [&]() {
		return brand_classifier::ClassifyBatch(
			brand_sess,
			vehicle_crops,
			app_config::kBrandInputH,
			app_config::kBrandInputW);
	});

	auto plate_future = std::async(std::launch::async, [&]() {
		return RunPlatePipeline(plate_sess, ocr_sess, vehicle_crops, vehicle_rects);
	});

	std::vector<brand_classifier::BrandResult> brand_results = brand_future.get();
	if (brand_results.size() != vehicle_crops.size()) {
		throw std::runtime_error("Loi noi bo: so ket qua brand != so vehicle crops");
	}

	PlatePipelineResult plate_result = plate_future.get();

	if (verbose) {
		for (size_t i = 0; i < brand_results.size(); ++i) {
			std::cout
				<< "vehicle " << i
				<< ": brand=" << brand_results[i].class_id << "(" << BrandClassName(brand_results[i].class_id) << ")"
				<< " brand_conf=" << brand_results[i].conf << "\n";
		}
	}

	if (!plate_result.has_any_plate) {
		if (verbose) {
			std::cout << "Khong phat hien bien so\n";
		}
		for (size_t i = 0; i < vehicles_used.size(); ++i) {
			VehicleOverlayResult v;
			v.det = vehicles_used[i];
			v.has_plate = false;
			v.brand_id = brand_results[i].class_id;
			out_overlay.vehicles.push_back(v);
		}
		return true;
	}

	for (size_t i = 0; i < vehicles_used.size(); ++i) {
		VehicleOverlayResult v;
		v.det = vehicles_used[i];
		v.has_plate = plate_result.vehicle_has_plate[i];
		v.brand_id = brand_results[i].class_id;
		out_overlay.vehicles.push_back(v);
	}

	for (size_t i = 0; i < plate_result.texts.size(); ++i) {
		PlateOverlayResult p;
		p.det = plate_result.plate_boxes_in_image[i];
		p.text = plate_result.texts[i].text;
		p.conf_avg = plate_result.texts[i].conf_avg;
		out_overlay.plates.push_back(std::move(p));
		if (verbose) {
			std::cout
				<< "plate " << i
				<< ": text=" << plate_result.texts[i].text
				<< " plate_conf=" << plate_result.plate_boxes_in_image[i].score
				<< " ocr_conf_avg=" << plate_result.texts[i].conf_avg
				<< " box=[" << plate_result.plate_boxes_in_image[i].x1 << "," << plate_result.plate_boxes_in_image[i].y1 << "," << plate_result.plate_boxes_in_image[i].x2 << "," << plate_result.plate_boxes_in_image[i].y2 << "]\n";
		}
	}

	return true;
}

void DrawFrameOverlay(cv::Mat& bgr, const FrameOverlayResult& overlay) {
	for (const auto& v : overlay.vehicles) {
		DrawVehicle(bgr, v.det, v.has_plate, v.brand_id);
	}
	for (const auto& p : overlay.plates) {
		DrawPlate(bgr, p.det, p.text, p.conf_avg);
	}
}

bool AnnotateFrame(
	cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool verbose) {
	FrameOverlayResult overlay;
	const bool detected = InferFrameOverlay(
		bgr,
		vehicle_sess,
		plate_sess,
		ocr_sess,
		brand_sess,
		overlay,
		verbose);
	if (detected) {
		DrawFrameOverlay(bgr, overlay);
	}
	return detected;
}
