#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

// Vẽ FPS lên frame/video output.
void DrawFps(cv::Mat& bgr, double fps);

// Chạy pipeline annotate cho 1 frame:
// detect vehicle -> branch brand(batch) song song với plate detect(batch)+ocr(batch) -> vẽ kết quả.
bool AnnotateFrame(
	cv::Mat& bgr,
	Ort::Session& vehicle_sess,
	Ort::Session& plate_sess,
	Ort::Session& ocr_sess,
	Ort::Session& brand_sess,
	bool verbose);
