#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR"
BUILD_DIR="$SOURCE_DIR/build"
OUT_DIR="$SOURCE_DIR/out/build"

readonly SCRIPT_NAME="$(basename "$0")"

info() {
  printf '[INFO] %s\n' "$*"
}

error() {
  printf '[ERROR] %s\n' "$*" >&2
}

usage() {
  cat <<EOF
Usage: ./$SCRIPT_NAME [options]

Configure and build OCR Plate with CMake.

Options:
  --build-type <type>   Build type (default: Release)
  --jobs <n>            Parallel jobs (default: nproc)
  --clean               Remove existing build and output folders first
  --target <name>       Build target (repeatable). Default: main benchmark
  --help, -h            Show this help message

Examples:
  ./$SCRIPT_NAME
  ./$SCRIPT_NAME --build-type Debug --jobs 8
  ./$SCRIPT_NAME --clean --target main
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    error "Missing required command: $1"
    exit 1
  fi
}

main() {
  local build_type="Release"
  local jobs
  jobs="$(nproc)"
  local clean=0
  local -a targets=()

  while (($#)); do
    case "$1" in
      --build-type)
        shift
        [[ $# -gt 0 ]] || { error "--build-type requires a value"; exit 1; }
        build_type="$1"
        ;;
      --jobs)
        shift
        [[ $# -gt 0 ]] || { error "--jobs requires a value"; exit 1; }
        jobs="$1"
        ;;
      --clean)
        clean=1
        ;;
      --target)
        shift
        [[ $# -gt 0 ]] || { error "--target requires a value"; exit 1; }
        targets+=("$1")
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        error "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done

  if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(main benchmark)
  fi

  require_command cmake

  if ((clean)); then
    info "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR" "$OUT_DIR"
  fi

  info "Configuring CMake (type: $build_type)..."
  cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$build_type"

  info "Building targets: ${targets[*]} (jobs: $jobs)"
  cmake --build "$BUILD_DIR" -j"$jobs" --target "${targets[@]}"

  info "Build completed. Binaries expected in: $OUT_DIR/bin"
}

main "$@"
