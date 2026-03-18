/*
 * Mo ta file: Dinh nghia kieu du lieu detection dung chung cho pipeline YOLO.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

namespace yolo_detector {

// Lightweight detection type (pure geometry + score/class).
// Kept in the existing namespace to avoid cascading renames.
struct Detection {
	float x1 = 0.0f;
	float y1 = 0.0f;
	float x2 = 0.0f;
	float y2 = 0.0f;
	float score = 0.0f;
	int cls = -1;
};

} // namespace yolo_detector
