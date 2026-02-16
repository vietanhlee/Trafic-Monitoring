#pragma once

#include <filesystem>
#include <iosfwd>

namespace cli_args {

struct Options {
	std::filesystem::path image_path;
	std::filesystem::path folder_path;
	std::filesystem::path video_path;
	bool show = false;
	bool show_help = false;
};

void PrintUsage(const char* argv0, std::ostream& os);
Options Parse(int argc, char** argv);

} // namespace cli_args
