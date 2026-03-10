#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace vehicle_identity_store {

struct VehicleIdentity {
	int track_id = -1;
	bool brand_accepted = false;
	int brand_id = -1;
	float brand_conf = 0.0f;

	bool plate_accepted = false;
	std::string plate_text;
	float plate_det_conf = 0.0f;
	float plate_ocr_conf = 0.0f;

	// Consider a track "identified" as soon as either attribute is accepted.
	bool IsIdentified() const { return brand_accepted || plate_accepted; }

	bool IsComplete() const { return brand_accepted && plate_accepted; }
};

class VehicleIdentityStore {
public:
	VehicleIdentityStore(
		float brand_accept_threshold,
		float plate_ocr_accept_threshold,
		int plate_min_length,
		int plate_max_length);

	void UpdateBrand(int track_id, int brand_id, float brand_conf);
	void UpdatePlate(int track_id, const std::string& plate_text, float plate_det_conf, float plate_ocr_conf);

	bool HasBrandAccepted(int track_id) const;
	bool HasPlateAccepted(int track_id) const;
	bool IsIdentified(int track_id) const;
	bool IsComplete(int track_id) const;

	const VehicleIdentity* Get(int track_id) const;
	std::vector<VehicleIdentity> Snapshot() const;

private:
	VehicleIdentity& Ensure(int track_id);
	const VehicleIdentity* Find(int track_id) const;

	float brand_accept_threshold_ = 0.8f;
	float plate_ocr_accept_threshold_ = 0.8f;
	int plate_min_length_ = 0;
	int plate_max_length_ = 999;
	std::unordered_map<int, VehicleIdentity> identities_;
};

} // namespace vehicle_identity_store
