#!/usr/bin/env bash
# Compile + flash the unified mini-hexapod firmware with the Arduino IDE's bundled CLI.
#   bash mini_hexapod/firmware/flash.sh s3 /dev/cu.usbmodem101      # XIAO ESP32-S3 (in the bot)
#   bash mini_hexapod/firmware/flash.sh c3 /dev/cu.usbmodem101      # original XIAO ESP32-C3
#   bash mini_hexapod/firmware/flash.sh s3                          # compile only
# Flashing the app does NOT touch NVS (trims / signs / limits persist). A board with
# empty NVS starts from the calibration baked into the sketch (DEFAULT_TRIM_US / DEFAULT_SIGN).
# If the S3 does not enumerate after a flash in a different USB mode: hold BOOT, tap RESET,
# release BOOT, flash again.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SKETCH="$HERE/arduino/hexapod_firmware"
ACLI="${ARDUINO_CLI:-}"
if [ -z "$ACLI" ]; then
  for cand in "$(command -v arduino-cli || true)" \
              "/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli"; do
    [ -n "$cand" ] && [ -x "$cand" ] && ACLI="$cand" && break
  done
fi
[ -x "${ACLI:-}" ] || { echo "arduino-cli not found (set ARDUINO_CLI)"; exit 1; }
case "${1:-}" in
  s3) FQBN="esp32:esp32:XIAO_ESP32S3:USBMode=hwcdc" ;;   # same USB-Serial/JTAG path as the C3
  c3) FQBN="esp32:esp32:XIAO_ESP32C3" ;;
  *)  echo "usage: $0 s3|c3 [PORT]"; exit 1 ;;
esac
OUT="${TMPDIR:-/tmp}/hexapod_fw_build_$1"
mkdir -p "$OUT"
"$ACLI" compile --fqbn "$FQBN" --output-dir "$OUT" "$SKETCH"
if [ -n "${2:-}" ]; then
  "$ACLI" upload --fqbn "$FQBN" --port "$2" --input-dir "$OUT"
  echo "flashed $1 on $2 — open the port at 115200 and press ? for the menu"
fi
