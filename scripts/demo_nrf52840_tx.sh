#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  demo_nrf52840_tx.sh [--uart PORT] [--address ADDR] TEXT
  demo_nrf52840_tx.sh [--uart PORT] [--address ADDR] --hex HEX

Configure the already-flashed nRF52840 advertising demo over its 115200-baud
UART. If --uart is omitted, PHANTOM_UART, the known demo DK, or a single
SEGGER J-Link VCOM interface is selected in that order.

Options:
  --uart PORT       UART device, for example /dev/ttyACM2
  --address ADDR    Advertiser address (default: D1:22:33:44:55:66)
  --hex HEX         Supply binary covert data as hexadecimal
  -h, --help        Show this help
EOF
}

uart=""
address="D1:22:33:44:55:66"
mode="text"
data=""

while (($#)); do
    case "$1" in
        --uart)
            [[ $# -ge 2 ]] || { echo "error: --uart requires a port" >&2; exit 2; }
            uart=$2
            shift 2
            ;;
        --address)
            [[ $# -ge 2 ]] || { echo "error: --address requires an address" >&2; exit 2; }
            address=$2
            shift 2
            ;;
        --hex)
            [[ $# -ge 2 ]] || { echo "error: --hex requires data" >&2; exit 2; }
            mode="hex"
            data=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            [[ -z "$data" ]] || { echo "error: supply exactly one covert data value" >&2; exit 2; }
            data=$1
            shift
            ;;
    esac
done

[[ -n "$data" ]] || { usage >&2; exit 2; }
[[ $# -eq 0 ]] || { echo "error: unexpected argument: $1" >&2; exit 2; }

python3 - "$uart" "$address" "$mode" "$data" <<'PY'
import glob
import json
import os
import re
import sys
import time

try:
    import serial
except ImportError as exc:
    raise SystemExit("error: pyserial is required (python3 -m pip install pyserial)") from exc

port, address, mode, value = sys.argv[1:]
if not port:
    port = os.environ.get("PHANTOM_UART", "")
if not port:
    preferred = sorted(glob.glob("/dev/serial/by-id/*1050216757-if00"))
    candidates = sorted(glob.glob("/dev/serial/by-id/*SEGGER_J-Link*-if00"))
    if preferred:
        port = preferred[0]
    elif len(candidates) == 1:
        port = candidates[0]
    elif not candidates:
        raise SystemExit("error: nRF52840 UART not found; pass --uart PORT")
    else:
        raise SystemExit("error: multiple J-Link UARTs found; pass --uart PORT")

if not re.fullmatch(r"(?i)[0-9a-f]{2}(?::[0-9a-f]{2}){5}", address):
    raise SystemExit(f"error: invalid BLE address: {address}")
address = address.upper()

if mode == "hex":
    try:
        payload = bytes.fromhex(value)
    except ValueError as exc:
        raise SystemExit(f"error: invalid --hex value: {exc}") from exc
    shown = payload.hex()
else:
    payload = value.encode("utf-8")
    shown = value
if not 1 <= len(payload) <= 239:
    raise SystemExit("error: covert data must encode to 1..239 bytes")

command = f"phantom covert-hex {payload.hex()}\n".encode("ascii")
ack_pattern = re.compile(rb"PHANTOM_COVERT_ACK\s+(\{[^\r\n]+\})")
transcript = bytearray()
try:
    with serial.Serial(port, 115200, timeout=0.2, write_timeout=3,
                       rtscts=False, dsrdtr=False) as device:
        device.reset_input_buffer()
        boot_deadline = time.monotonic() + 2.0
        while time.monotonic() < boot_deadline:
            device.readline()
        device.write(command)
        device.flush()
        deadline = time.monotonic() + 3.0
        match = None
        while time.monotonic() < deadline:
            line = device.readline()
            if line:
                transcript.extend(line)
                match = ack_pattern.search(transcript)
                if match:
                    break
except (OSError, serial.SerialException) as exc:
    raise SystemExit(f"error: cannot use UART {port}: {exc}") from exc

if not match:
    detail = transcript.decode("utf-8", errors="replace").strip()
    raise SystemExit(f"error: no firmware ACK from {port}" + (f": {detail}" if detail else ""))
try:
    ack = json.loads(match.group(1))
except json.JSONDecodeError as exc:
    raise SystemExit(f"error: malformed firmware ACK: {exc}") from exc
if ack.get("status") != 0 or ack.get("len") != len(payload) or ack.get("hex", "").lower() != payload.hex():
    raise SystemExit(f"error: firmware rejected or changed covert data: {ack}")

normal_payload = bytes(reversed(bytes.fromhex(address.replace(":", "")))).hex()
print("PhantomChannel nRF52840 transmitter")
print(f"Address: {address}")
print(f"Normal payload: {normal_payload}")
print(f"Covert data: {shown}")
if mode == "text":
    print(f"Covert data hex: {payload.hex()}")
print("Advertising started")
PY
