/*
 * Mo ta file: Kho luu danh tinh xe theo track-id va ket qua bien so thu thap theo thoi gian.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace vehicle_identity_store {

struct VehicleIdentity {
	int track_id = -1;
	bool brand_accepted = false;
	bool brand_forced_unknown = false;
	int brand_attempts = 0;
	int brand_id = -1;
	float brand_conf = 0.0f;

	bool plate_accepted = false;
	bool plate_forced_unknown = false;
	bool plate_forced_no_plate = false;
	int plate_detect_attempts = 0;
	int plate_ocr_attempts = 0;
	std::string plate_text;
	float plate_det_conf = 0.0f;
	float plate_ocr_conf = 0.0f;

	// Consider a track "identified" as soon as either attribute is resolved.
	bool IsIdentified() const { return brand_accepted || brand_forced_unknown || plate_accepted; }

	bool IsComplete() const { return (brand_accepted || brand_forced_unknown) && plate_accepted; }
};

class VehicleIdentityStore {
public:
	VehicleIdentityStore(
		float brand_accept_threshold,
		int brand_max_attempts,
		float plate_ocr_accept_threshold,
		int plate_max_detect_attempts,
		int plate_max_ocr_attempts,
		std::string plate_unknown_text,
		std::string plate_no_plate_text,
		int plate_min_length,
		int plate_max_length);

	void UpdateBrand(int track_id, int brand_id, float brand_conf);
	void UpdatePlate(int track_id, const std::string& plate_text, float plate_det_conf, float plate_ocr_conf);
	void MarkPlateMiss(int track_id);

	bool HasBrandAccepted(int track_id) const;
	bool HasBrandResolved(int track_id) const;
	bool HasPlateAccepted(int track_id) const;
	bool IsIdentified(int track_id) const;
	bool IsComplete(int track_id) const;

	const VehicleIdentity* Get(int track_id) const;
	std::vector<VehicleIdentity> Snapshot() const;

private:
	VehicleIdentity& Ensure(int track_id);
	const VehicleIdentity* Find(int track_id) const;

	float brand_accept_threshold_ = 0.8f;
	int brand_max_attempts_ = 3;
	float plate_ocr_accept_threshold_ = 0.8f;
	int plate_max_detect_attempts_ = 3;
	int plate_max_ocr_attempts_ = 3;
	std::string plate_unknown_text_ = "unknown";
	std::string plate_no_plate_text_ = "no_plate";
	int plate_min_length_ = 0;
	int plate_max_length_ = 999;
	std::unordered_map<int, VehicleIdentity> identities_;
};

} // namespace vehicle_identity_store
