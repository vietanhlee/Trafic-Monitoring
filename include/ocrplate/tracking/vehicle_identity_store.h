/*
 * Mô tả file: Kho lưu danh tính xe theo track-id và kết quả biển số thu thập theo thời gian.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace vehicle_identity_store {

struct VehicleIdentity {
	// Track ID mà bản ghi danh tính này đại diện.
	int track_id = -1;
	// Đã có kết quả brand đạt ngưỡng chấp nhận hay chưa.
	bool brand_accepted = false;
	// Đã khóa brand về trạng thái unknown sau nhiều lần thất bại hay chưa.
	bool brand_forced_unknown = false;
	// Số lần thử brand classify đã thực hiện.
	int brand_attempts = 0;
	// Brand id đã chấp nhận (hoặc mới nhất).
	int brand_id = -1;
	// Confidence của brand_id.
	float brand_conf = 0.0f;

	// Đã chấp nhận biển số hợp lệ hay chưa.
	bool plate_accepted = false;
	// Đã ép biển số = unknown sau quá nhiều lần OCR thất bại hay chưa.
	bool plate_forced_unknown = false;
	// Đã ép biển số = no_plate sau quá nhiều lần detect thất bại hay chưa.
	bool plate_forced_no_plate = false;
	// Số lần detect plate đã thử trên track này.
	int plate_detect_attempts = 0;
	// Số lần OCR plate đã thử trên track này.
	int plate_ocr_attempts = 0;
	// Biển số đã chấp nhận hoặc fallback text.
	std::string plate_text;
	// Score detector của plate text được giữ lại.
	float plate_det_conf = 0.0f;
	// Score OCR trung bình của plate text được giữ lại.
	float plate_ocr_conf = 0.0f;

	// Track được xem là "identified" khi ít nhất một nhánh đã được giải quyết.
	// Mục tiêu: cho phép thống kê sớm cả khi brand/plate chưa đầy đủ.
	bool IsIdentified() const { return brand_accepted || brand_forced_unknown || plate_accepted; }

	// Track được xem là "complete" khi brand và plate đều đã có kết luận chấp nhận.
	bool IsComplete() const { return (brand_accepted || brand_forced_unknown) && plate_accepted; }
};

class VehicleIdentityStore {
public:
	// Khởi tạo kho lưu với các ngưỡng chấp nhận và ngưỡng fallback.
	// Các tham số này quyết định chiến lược "thử bao nhiêu lần" trước khi khóa kết quả.
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

	// Cập nhật kết quả brand cho 1 track.
	// Hàm sẽ tự tăng số lần thử và quyết định chấp nhận/force unknown theo cấu hình.
	void UpdateBrand(int track_id, int brand_id, float brand_conf);
	// Cập nhật kết quả plate OCR cho 1 track (gồm text + conf detect + conf OCR).
	// Hàm sẽ validate độ dài text và quyết định chấp nhận/fallback khi cần.
	void UpdatePlate(int track_id, const std::string& plate_text, float plate_det_conf, float plate_ocr_conf);
	// Đánh dấu một lần detect plate thất bại cho track.
	// Dùng khi detector không tìm thấy plate trong frame hiện tại.
	void MarkPlateMiss(int track_id);

	// Kiểm tra track đã có brand chấp nhận hay chưa.
	bool HasBrandAccepted(int track_id) const;
	// Kiểm tra track đã giải quyết brand (accepted hoặc forced_unknown) hay chưa.
	bool HasBrandResolved(int track_id) const;
	// Kiểm tra track da co plate chap nhan hay chưa.
	bool HasPlateAccepted(int track_id) const;
	// Kiểm tra track da du dieu kien identified hay chưa.
	bool IsIdentified(int track_id) const;
	// Kiểm tra track da complete hay chưa.
	bool IsComplete(int track_id) const;

	// Lấy con tro read-only den ban ghi theo track_id, nullptr nếu không ton tai.
	const VehicleIdentity* Get(int track_id) const;
	// Lấy snapshot toàn bộ ban ghi để dump report/log.
	std::vector<VehicleIdentity> Snapshot() const;

private:
	// Dam bao ban ghi ton tai cho track_id va tra về tham chieu để cap nhất.
	VehicleIdentity& Ensure(int track_id);
	// Tim ban ghi theo track_id (read-only), nullptr nếu không co.
	const VehicleIdentity* Find(int track_id) const;

	// Ngưỡng chap nhan confidence cho brand.
	float brand_accept_threshold_ = 0.8f;
	// So lan thu tối đa cho brand trước khi force unknown.
	int brand_max_attempts_ = 3;
	// Ngưỡng chap nhan confidence OCR cho plate.
	float plate_ocr_accept_threshold_ = 0.8f;
	// So lan detect plate that bai tối đa trước khi force no_plate.
	int plate_max_detect_attempts_ = 3;
	// So lan OCR plate that bai tối đa trước khi force unknown.
	int plate_max_ocr_attempts_ = 3;
	// Chuoi fallback khi OCR that bai qua ngưỡng.
	std::string plate_unknown_text_ = "unknown";
	// Chuoi fallback khi detect plate that bai qua ngưỡng.
	std::string plate_no_plate_text_ = "no_plate";
	// Gioi han duoi do dai bịển số hop le.
	int plate_min_length_ = 0;
	// Gioi han tren do dai bịển số hop le.
	int plate_max_length_ = 999;
	// Bang luu danh tinh theo track_id.
	std::unordered_map<int, VehicleIdentity> identities_;
};

} // namespace vehicle_identity_store
