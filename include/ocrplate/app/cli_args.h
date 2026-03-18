/*
 * Mo ta file: Khai bao cau truc va ham parse tham so dong lenh cho ung dung.
 * Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
 */
#pragma once

#include <filesystem>
#include <iosfwd>

namespace cli_args {

struct Options {
	std::filesystem::path image_path;
	std::filesystem::path folder_path;
	std::filesystem::path video_path;
	bool show = false;
	bool no_save = false;
	bool show_help = false;
};

// In huong dan su dung len output stream (stdout/stderr).
void PrintUsage(const char* argv0, std::ostream& os);
// Parse tham so dong lenh va tra ve cau hinh mode chay.
Options Parse(int argc, char** argv);

} // namespace cli_args
