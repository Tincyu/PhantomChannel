#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  demo_nrf52840_peripheral_tx.sh [--uart PORT] [--address ADDR] TEXT
  demo_nrf52840_peripheral_tx.sh [--uart PORT] [--address ADDR] --hex HEX

Set runtime covert data on the connected nRF52840 HRS peripheral. The board
must already run the peripheral-side firmware and be connected to its BLE
central peer.

Options:
  --uart PORT       nRF52840 application UART
  --address ADDR    Peripheral address (default: C0:DE:52:84:00:03)
  --hex HEX         Supply binary covert data as hexadecimal
  -h, --help        Show this help
EOF
}

uart=""
address="C0:DE:52:84:00:03"
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
port = port or os.environ.get("PHANTOM_UART", "")
if not port:
    preferred = sorted(glob.glob("/dev/serial/by-id/*1050216757-if00"))
    candidates = sorted(glob.glob("/dev/serial/by-id/*SEGGER_J-Link*-if00"))
    if preferred:
        port = preferred[0]
    elif len(candidates) == 1:
        port = candidates[0]
    elif candidates:
        raise SystemExit("error: multiple J-Link UARTs found; pass --uart PORT")
    else:
        raise SystemExit("error: nRF52840 UART not found; pass --uart PORT")

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
if not 1 <= len(payload) <= 240:
    raise SystemExit("error: covert data must encode to 1..240 bytes")

expected_hex = payload.hex()
ack_re = re.compile(rb'HRS_COVERT_ACK\s+(\{[^\r\n]+\})')
status_re = re.compile(rb'enabled=1\s+len=(\d+)\s+hex=([0-9a-fA-F]+)')
tx_re = re.compile(rb'HRS_TX\s+\{[^\r\n]*"bpm":(\d+)[^\r\n]*"status":0')
tx_status_re = re.compile(rb'HRS_TX\s+\{[^\r\n]*"status":(-?\d+)')
transcript = bytearray()
ack_ok = False
bpm = None
last_tx_status = None
try:
    with serial.Serial(port, 115200, timeout=0.2, write_timeout=3,
                       rtscts=False, dsrdtr=False) as device:
        device.reset_input_buffer()
        boot_deadline = time.monotonic() + 2.0
        while time.monotonic() < boot_deadline:
            device.readline()
        device.write(f"phantom conn_covert_hex {expected_hex}\n".encode("ascii"))
        device.flush()
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            line = device.readline()
            if not line:
                continue
            transcript.extend(line)
            tx_status_match = tx_status_re.search(line)
            if tx_status_match:
                last_tx_status = int(tx_status_match.group(1))
            tx_match = tx_re.search(line)
            if tx_match:
                bpm = int(tx_match.group(1))
            for match in ack_re.finditer(transcript):
                try:
                    ack = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                ack_ok = (
                    ack.get("status") == 0
                    and ack.get("len") == len(payload)
                    and ack.get("hex", "").lower() == expected_hex
                )
            if ack_ok and bpm is not None:
                break
        if not ack_ok:
            device.write(b"phantom conn_covert_status\n")
            device.flush()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                line = device.readline()
                if line:
                    transcript.extend(line)
                    match = status_re.search(line)
                    if match and int(match.group(1)) == len(payload) and match.group(2).decode().lower() == expected_hex:
                        ack_ok = True
                        break
except (OSError, serial.SerialException) as exc:
    raise SystemExit(f"error: cannot use UART {port}: {exc}") from exc

if not ack_ok:
    raise SystemExit("error: firmware did not acknowledge the exact covert data")
if bpm is None:
    if last_tx_status is not None and last_tx_status != 0:
        raise SystemExit(
            f"error: covert data was accepted, but HRS notify status={last_tx_status}; "
            "reconnect the phone and keep HRS notifications enabled"
        )
    raise SystemExit("error: covert data was accepted, but no connected HRS transmission was observed")

normal_payload = bytes((0x06, bpm)).hex()
print("PhantomChannel nRF52840 peripheral transmitter")
print(f"Address: {address}")
print(f"Normal payload: {normal_payload}")
print(f"Covert data: {shown}")
if mode == "text":
    print(f"Covert data hex: {expected_hex}")
print("Connected HRS transmission active")
PY
