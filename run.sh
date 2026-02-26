#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAIN_BIN="$SCRIPT_DIR/out/build/bin/main"
BENCHMARK_BIN="$SCRIPT_DIR/out/build/bin/benchmark"

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

Run OCR Plate binaries with convenient shortcuts.

Modes:
  main mode (default):
    ./$SCRIPT_NAME --image <path> [--show|--no-show]
    ./$SCRIPT_NAME --folder <path> [--show|--no-show]
    ./$SCRIPT_NAME --video <path> [--show|--no-show]

  benchmark mode:
    ./$SCRIPT_NAME --benchmark --image <path> [--warmup <n>] [--runs <n>]

Options:
  --benchmark      Run benchmark binary instead of main
  --help, -h       Show this help message

Notes:
  - Any additional arguments are forwarded to the selected binary.
  - Ensure you already built the project via ./build.sh.
EOF
}

require_file() {
  local file_path="$1"
  if [[ ! -x "$file_path" ]]; then
    error "Executable not found: $file_path"
    error "Run ./build.sh first."
    exit 1
  fi
}

main() {
  local use_benchmark=0
  local -a forwarded_args=()

  while (($#)); do
    case "$1" in
      --benchmark)
        use_benchmark=1
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        forwarded_args+=("$1")
        ;;
    esac
    shift
  done

  local selected_bin="$MAIN_BIN"
  if ((use_benchmark)); then
    selected_bin="$BENCHMARK_BIN"
  fi

  require_file "$selected_bin"

  info "Executing: $selected_bin ${forwarded_args[*]:-}"
  "$selected_bin" "${forwarded_args[@]}"
}

main "$@"
