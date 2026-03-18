/*
 * Mo ta file: Trien khai bo nho danh tinh xe: hop nhat plate/brand theo track.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#include "ocrplate/tracking/vehicle_identity_store.h"

#include <algorithm>
#include <stdexcept>

namespace vehicle_identity_store {

namespace {

bool HasUnderscoreOnlyAtEnd(const std::string& plate_text) {
	// Chap nhan chuoi co '_' chi o cuoi (padding), khong chap nhan '_' xen giua.
	const size_t first_underscore = plate_text.find('_');
	if (first_underscore == std::string::npos) {
		return true;
	}
	for (size_t i = first_underscore; i < plate_text.size(); ++i) {
		if (plate_text[i] != '_') {
			return false;
		}
	}
	return true;
}

std::string TrimTrailingUnderscore(std::string plate_text) {
	// Bo ky tu padding o cuoi de lay text bien so thuc.
	while (!plate_text.empty() && plate_text.back() == '_') {
		plate_text.pop_back();
	}
	return plate_text;
}

} // namespace

VehicleIdentityStore::VehicleIdentityStore(
	float brand_accept_threshold,
	int brand_max_attempts,
	float plate_ocr_accept_threshold,
	int plate_max_detect_attempts,
	int plate_max_ocr_attempts,
	std::string plate_unknown_text,
	std::string plate_no_plate_text,
	int plate_min_length,
	int plate_max_length)
	: brand_accept_threshold_(brand_accept_threshold),
	  brand_max_attempts_(brand_max_attempts),
	  plate_ocr_accept_threshold_(plate_ocr_accept_threshold),
	  plate_max_detect_attempts_(plate_max_detect_attempts),
	  plate_max_ocr_attempts_(plate_max_ocr_attempts),
	  plate_unknown_text_(std::move(plate_unknown_text)),
	  plate_no_plate_text_(std::move(plate_no_plate_text)),
	  plate_min_length_(plate_min_length),
	  plate_max_length_(plate_max_length) {
	if (brand_accept_threshold_ <= 0.0f || brand_accept_threshold_ > 1.0f) {
		throw std::runtime_error("Nguong brand accept khong hop le");
	}
	if (brand_max_attempts_ <= 0) {
		throw std::runtime_error("So lan predict brand toi da khong hop le");
	}
	if (plate_ocr_accept_threshold_ <= 0.0f || plate_ocr_accept_threshold_ > 1.0f) {
		throw std::runtime_error("Nguong plate ocr accept khong hop le");
	}
	if (plate_max_detect_attempts_ <= 0) {
		throw std::runtime_error("So lan detect plate toi da khong hop le");
	}
	if (plate_max_ocr_attempts_ <= 0) {
		throw std::runtime_error("So lan OCR plate toi da khong hop le");
	}
	if (plate_unknown_text_.empty()) {
		throw std::runtime_error("Gia tri plate unknown khong duoc rong");
	}
	if (plate_no_plate_text_.empty()) {
		throw std::runtime_error("Gia tri plate no_plate khong duoc rong");
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
	// Tao moi record identity khi track_id xuat hien lan dau.
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
	if (one.brand_accepted || one.brand_forced_unknown) {
		// Da chap nhan brand hoac het budget thi khoa ket qua, khong ghi de.
		return;
	}

	++one.brand_attempts;
	if (brand_conf > brand_accept_threshold_) {
		one.brand_accepted = true;
		one.brand_forced_unknown = false;
		one.brand_id = brand_id;
		one.brand_conf = brand_conf;
		return;
	}

	if (one.brand_attempts >= brand_max_attempts_) {
		// Het budget predict brand: khoa lai de frame sau khong classify nua.
		one.brand_forced_unknown = true;
	}
}

void VehicleIdentityStore::UpdatePlate(int track_id, const std::string& plate_text, float plate_det_conf, float plate_ocr_conf) {
	if (track_id <= 0) {
		return;
	}
	VehicleIdentity& one = Ensure(track_id);
	if (one.plate_accepted) {
		// Da chap nhan plate (hoac forced unknown) thi bo qua lan cap nhat tiep.
		return;
	}

	// Moi lan goi UpdatePlate duoc xem la 1 lan da doc OCR plate cho track nay.
	++one.plate_ocr_attempts;

	bool accepted = false;
	std::string normalized_plate_text;
	if (!plate_text.empty() && HasUnderscoreOnlyAtEnd(plate_text)) {
		// Chuan hoa text OCR truoc khi kiem tra do dai/nguong conf.
		normalized_plate_text = TrimTrailingUnderscore(plate_text);
		if (!normalized_plate_text.empty()) {
			const int plate_len = static_cast<int>(normalized_plate_text.size());
			if (plate_len >= plate_min_length_ && plate_len <= plate_max_length_ && plate_ocr_conf > plate_ocr_accept_threshold_) {
				accepted = true;
			}
		}
	}

	if (accepted) {
		// Khi dat dieu kien, dong bang ket qua bien so cho track nay.
		one.plate_accepted = true;
		one.plate_forced_unknown = false;
		one.plate_forced_no_plate = false;
		one.plate_text = normalized_plate_text;
		one.plate_det_conf = plate_det_conf;
		one.plate_ocr_conf = plate_ocr_conf;
		return;
	}

	// Vuot budget OCR ma van khong chap nhan duoc -> khoa unknown.
	if (one.plate_ocr_attempts >= plate_max_ocr_attempts_) {
		// Het budget OCR: danh dau unknown de ket thuc qua trinh tim bien so.
		one.plate_accepted = true;
		one.plate_forced_unknown = true;
		one.plate_forced_no_plate = false;
		one.plate_text = plate_unknown_text_;
		one.plate_det_conf = plate_det_conf;
		one.plate_ocr_conf = plate_ocr_conf;
	}
}

void VehicleIdentityStore::MarkPlateMiss(int track_id) {
	if (track_id <= 0) {
		return;
	}
	VehicleIdentity& one = Ensure(track_id);
	if (one.plate_accepted) {
		return;
	}

	++one.plate_detect_attempts;
	if (one.plate_detect_attempts >= plate_max_detect_attempts_) {
		// Het budget detect: khoa plate=no_plate va bo qua detect/oCR tu frame sau.
		one.plate_accepted = true;
		one.plate_forced_unknown = false;
		one.plate_forced_no_plate = true;
		one.plate_text = plate_no_plate_text_;
		one.plate_det_conf = 0.0f;
		one.plate_ocr_conf = 0.0f;
	}
}

bool VehicleIdentityStore::HasBrandAccepted(int track_id) const {
	const VehicleIdentity* one = Find(track_id);
	return one ? one->brand_accepted : false;
}

bool VehicleIdentityStore::HasBrandResolved(int track_id) const {
	const VehicleIdentity* one = Find(track_id);
	return one ? (one->brand_accepted || one->brand_forced_unknown) : false;
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
