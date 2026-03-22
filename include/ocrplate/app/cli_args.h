/*
 * Mô tả file: Khai báo cấu trúc và hàm parse tham số dòng lệnh cho ứng dụng.
 * Ghi chú: Comment tiếng Việt được bổ sung để dễ đọc và bảo trì.
 */
#pragma once

#include <filesystem>
#include <iosfwd>

namespace cli_args {

struct Options {
	// Đường dẫn ảnh đầu vào khi chạy mode xử lý 1 ảnh.
	// Rỗng nếu người dùng không chọn mode image.
	std::filesystem::path image_path;
	// Đường dẫn thư mục ảnh đầu vào khi chạy mode batch-folder.
	std::filesystem::path folder_path;
	// Đường dẫn video đầu vào khi chạy mode video.
	std::filesystem::path video_path;
	// Bật cửa sổ preview realtime.
	// true: hiển thị ảnh/video annotated trên màn hình.
	bool show = false;
	// true: không ghi file output ra ổ đĩa.
	bool no_save = false;
	// true: in hướng dẫn sử dụng và thoát.
	bool show_help = false;
};

// In hướng dẫn sử dụng CLI lên stream truyền vào (stdout/stderr).
// Dùng cho cả trường hợp user gọi --help và trường hợp parse lỗi.
void PrintUsage(const char* argv0, std::ostream& os);
// Parse tham số dòng lệnh thành cấu trúc Options.
// Hàm này không chạy infer, chỉ xác định chế độ và tham số runtime.
Options Parse(int argc, char** argv);

} // namespace cli_args
