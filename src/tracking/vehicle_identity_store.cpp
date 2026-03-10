#include "ocrplate/tracking/vehicle_identity_store.h"

#include <algorithm>
#include <stdexcept>

namespace vehicle_identity_store {

VehicleIdentityStore::VehicleIdentityStore(
	float brand_accept_threshold,
	float plate_ocr_accept_threshold,
	int plate_min_length,
	int plate_max_length)
	: brand_accept_threshold_(brand_accept_threshold),
	  plate_ocr_accept_threshold_(plate_ocr_accept_threshold),
	  plate_min_length_(plate_min_length),
	  plate_max_length_(plate_max_length) {
	if (brand_accept_threshold_ <= 0.0f || brand_accept_threshold_ > 1.0f) {
		throw std::runtime_error("Nguong brand accept khong hop le");
	}
	if (plate_ocr_accept_threshold_ <= 0.0f || plate_ocr_accept_threshold_ > 1.0f) {
		throw std::runtime_error("Nguong plate ocr accept khong hop le");
	}
	if (plate_min_length_ <= 0 || plate_max_length_ < plate_min_length_) {
		throw std::runtime_error("Khoang do dai bien so khong hop le");
	}
}

VehicleIdentity& VehicleIdentityStore::Ensure(int track_id) {
	auto it = identities_.find(track_id);
	if (it != identities_.end()) {
		return it->second;
	}
	VehicleIdentity fresh;
	fresh.track_id = track_id;
	auto inserted = identities_.emplace(track_id, std::move(fresh));
	return inserted.first->second;
}

const VehicleIdentity* VehicleIdentityStore::Find(int track_id) const {
	auto it = identities_.find(track_id);
	if (it == identities_.end()) {
		return nullptr;
	}
	return &it->second;
}

void VehicleIdentityStore::UpdateBrand(int track_id, int brand_id, float brand_conf) {
	if (track_id <= 0 || brand_id < 0) {
		return;
	}
	VehicleIdentity& one = Ensure(track_id);
	if (one.brand_accepted) {
		return;
	}
	if (brand_conf > brand_accept_threshold_) {
		one.brand_accepted = true;
		one.brand_id = brand_id;
		one.brand_conf = brand_conf;
	}
}

void VehicleIdentityStore::UpdatePlate(int track_id, const std::string& plate_text, float plate_det_conf, float plate_ocr_conf) {
	if (track_id <= 0) {
		return;
	}
	VehicleIdentity& one = Ensure(track_id);
	if (one.plate_accepted) {
		return;
	}
	if (plate_text.empty()) {
		return;
	}
	const int plate_len = static_cast<int>(plate_text.size());
	if (plate_len < plate_min_length_ || plate_len > plate_max_length_) {
		return;
	}
	if (plate_ocr_conf > plate_ocr_accept_threshold_) {
		one.plate_accepted = true;
		one.plate_text = plate_text;
		one.plate_det_conf = plate_det_conf;
		one.plate_ocr_conf = plate_ocr_conf;
	}
}

bool VehicleIdentityStore::HasBrandAccepted(int track_id) const {
	const VehicleIdentity* one = Find(track_id);
	return one ? one->brand_accepted : false;
}

bool VehicleIdentityStore::HasPlateAccepted(int track_id) const {
	const VehicleIdentity* one = Find(track_id);
	return one ? one->plate_accepted : false;
}

bool VehicleIdentityStore::IsIdentified(int track_id) const {
	const VehicleIdentity* one = Find(track_id);
	return one ? one->IsIdentified() : false;
}

bool VehicleIdentityStore::IsComplete(int track_id) const {
	const VehicleIdentity* one = Find(track_id);
	return one ? one->IsComplete() : false;
}

const VehicleIdentity* VehicleIdentityStore::Get(int track_id) const {
	return Find(track_id);
}

std::vector<VehicleIdentity> VehicleIdentityStore::Snapshot() const {
	std::vector<VehicleIdentity> out;
	out.reserve(identities_.size());
	for (const auto& kv : identities_) {
		out.push_back(kv.second);
	}
	std::sort(out.begin(), out.end(), [](const VehicleIdentity& a, const VehicleIdentity& b) {
		return a.track_id < b.track_id;
	});
	return out;
}

} // namespace vehicle_identity_store
