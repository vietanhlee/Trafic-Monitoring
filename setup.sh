#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_PACKAGES=(build-essential cmake pkg-config libopencv-dev)

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

error() {
  printf '[ERROR] %s\n' "$*" >&2
}

usage() {
  cat <<EOF
Usage: ./$SCRIPT_NAME [options]

Install required system dependencies for OCR Plate.

Options:
  --no-sudo        Run apt commands without sudo
  --skip-update    Skip 'apt update'
  --help, -h       Show this help message

Examples:
  ./$SCRIPT_NAME
  ./$SCRIPT_NAME --no-sudo --skip-update
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    error "Missing required command: $1"
    exit 1
  fi
}

main() {
  local use_sudo=1
  local skip_update=0

  while (($#)); do
    case "$1" in
      --no-sudo)
        use_sudo=0
        ;;
      --skip-update)
        skip_update=1
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

  if [[ "$(uname -s)" != "Linux" ]]; then
    error "This script currently supports Linux only."
    exit 1
  fi

  require_command apt

  local -a apt_cmd=(apt)
  if ((use_sudo)); then
    require_command sudo
    apt_cmd=(sudo apt)
  fi

  if ((skip_update == 0)); then
    info "Updating apt package index..."
    "${apt_cmd[@]}" update
  else
    info "Skipping apt update as requested."
  fi

  info "Installing dependencies: ${DEFAULT_PACKAGES[*]}"
  "${apt_cmd[@]}" install -y "${DEFAULT_PACKAGES[@]}"

  info "Validating toolchain..."
  require_command cmake
  require_command g++
  require_command pkg-config

  info "Setup completed successfully in: $SCRIPT_DIR"
}

main "$@"
