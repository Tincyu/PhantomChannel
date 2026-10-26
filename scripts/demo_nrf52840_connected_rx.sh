#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
receiver_root=${PHANTOM_RECEIVER_ROOT:-"$project_root/../PhantomChannel-receiver"}

usage() {
    cat <<'EOF'
Usage: demo_nrf52840_connected_rx.sh [options]

Capture and parse nRF52840 connected traffic, or replay an existing SC16
capture. Live capture streams IQ into the parser and displays
checksum-valid PhantomChannel frames as they arrive.

Options:
  --replay IQ             Parse an existing SC16 IQ file instead of receiving
  --metadata FILE         Capture metadata JSON; inferred next to --replay
  --output-dir DIR        Result directory
  --duration SEC          Optional live segment duration; 0 runs until Ctrl+C (default)
  --live                  Stream and parse during capture until Ctrl+C (default)
  --batch                 Capture first, then run the two-stage batch parser
  --usrp-args ARGS        UHD device arguments
  --center-frequency HZ   Default: 2406000000 (one BLE data channel)
  --sample-rate SPS       Default: 4000000
  --bandwidth HZ          Default: 4000000
  --gain DB               Default: 70
  --antenna NAME          Default: RX2
  --no-consistency-anchor Disable empty-before-data AA anchoring
  --consistency-window-us N  Empty-to-data pairing window (default: 1000)
  --consistency-min-pairs N  Required pairs per AA (default: 2)
  --consistency-max-addresses N  Maximum anchored AAs (default: 8)
  --known-aa-min-count N  Minimum packets for fallback AA discovery (default: 2)
  --capture-bin FILE      uhd_b210_capture_sc16 executable
  --receiver-root DIR     Receiver checkout containing parser dependencies
  -h, --help              Show this help
EOF
}

replay=""
metadata=""
output_dir=""
duration="0"
live_mode=1
# 40 MHz is required when the operator asks the B210 for a 40 MS/s
# wideband capture.  UHD can otherwise silently negotiate 16 MS/s while the
# parser still assumes the requested rate.  The actual rate is read back from
# capture metadata below, so this clock is also safe for the 4 MS/s default.
usrp_args="type=b200,master_clock_rate=40e6"
center_frequency="2406000000"
sample_rate="4000000"
bandwidth="4000000"
gain="70"
antenna="RX2"
consistency_anchor=1
consistency_window_us="1000"
consistency_min_pairs="2"
consistency_max_addresses="8"
known_aa_min_count="2"
capture_bin="$receiver_root/build-local/uhd_b210_capture_sc16"

while (($#)); do
    case "$1" in
        --replay|--metadata|--output-dir|--duration|--usrp-args|--center-frequency|--sample-rate|--bandwidth|--gain|--antenna|--consistency-window-us|--consistency-min-pairs|--consistency-max-addresses|--known-aa-min-count|--capture-bin|--receiver-root)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --replay) replay=$2 ;;
                --metadata) metadata=$2 ;;
                --output-dir) output_dir=$2 ;;
                --duration) duration=$2 ;;
                --usrp-args) usrp_args=$2 ;;
                --center-frequency) center_frequency=$2 ;;
                --sample-rate) sample_rate=$2 ;;
                --bandwidth) bandwidth=$2 ;;
                --gain) gain=$2 ;;
                --antenna) antenna=$2 ;;
                --consistency-window-us) consistency_window_us=$2 ;;
                --consistency-min-pairs) consistency_min_pairs=$2 ;;
                --consistency-max-addresses) consistency_max_addresses=$2 ;;
                --known-aa-min-count) known_aa_min_count=$2 ;;
                --capture-bin) capture_bin=$2 ;;
                --receiver-root)
                    receiver_root=$2
                    capture_bin="$receiver_root/build-local/uhd_b210_capture_sc16"
                    ;;
            esac
            shift 2
            ;;
        --consistency-anchor)
            consistency_anchor=1
            shift
            ;;
        --no-consistency-anchor)
            consistency_anchor=0
            shift
            ;;
        --live)
            live_mode=1
            shift
            ;;
        --batch)
            live_mode=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$output_dir" ]]; then
    output_dir="$project_root/artifacts/connected-runs/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$output_dir"

parser_python="$receiver_root/.venv-cuda/bin/python"
parser_entry="$receiver_root/vendor/BLE_encrypt_check/experiment/bt_40m_pfb_realtime.py"
parser_tool="$project_root/tools/two_stage_known_aa_parse.py"
parse_dir="$output_dir/parser"
[[ -x "$parser_python" ]] || { echo "error: parser Python not found: $parser_python" >&2; exit 1; }
[[ -f "$parser_entry" ]] || { echo "error: parser entry point not found: $parser_entry" >&2; exit 1; }

if (( live_mode )) && [[ -z "$replay" ]]; then
    [[ -x "$capture_bin" ]] || { echo "error: capture binary not executable: $capture_bin" >&2; exit 1; }
    exec python3 "$project_root/tools/live_connected_stream.py" \
        --output-dir "$output_dir" \
        --receiver-root "$receiver_root" \
        --capture-bin "$capture_bin" \
        --parser-python "$parser_python" \
        --parser-entrypoint "$parser_entry" \
        --duration "$duration" \
        --usrp-args "$usrp_args" \
        --center-frequency "$center_frequency" \
        --sample-rate "$sample_rate" \
        --bandwidth "$bandwidth" \
        --gain "$gain" \
        --antenna "$antenna"
fi

if [[ -z "$replay" ]] && (( ! live_mode )); then
    if ! awk -v value="$duration" 'BEGIN { exit !(value > 0) }'; then
        echo "error: --batch requires a positive --duration; use --live for Ctrl+C-ended capture" >&2
        exit 2
    fi
fi

if [[ -n "$replay" ]]; then
    iq=$replay
    if [[ -z "$metadata" ]]; then
        base=${replay%.sc16}
        metadata="${base}-metadata.json"
        [[ -f "$metadata" ]] || metadata="${base}.metadata.json"
    fi
else
    iq="$output_dir/capture.sc16"
    metadata="$output_dir/capture-metadata.json"
    [[ -x "$capture_bin" ]] || { echo "error: capture binary not executable: $capture_bin" >&2; exit 1; }
    echo "PhantomChannel nRF52840 connected receiver"
    echo "Capturing connected BLE: $center_frequency Hz, $sample_rate S/s, $bandwidth Hz"
    set +e
    "$capture_bin" \
        --args "$usrp_args" \
        --rate "$sample_rate" \
        --bandwidth "$bandwidth" \
        --frequency "$center_frequency" \
        --gain "$gain" \
        --antenna "$antenna" \
        --duration "$duration" \
        --block-samples 4096 \
        --output "$iq" \
        --metadata "$metadata" >"$output_dir/capture.log" 2>&1
    capture_status=$?
    set -e
    [[ $capture_status -eq 0 || $capture_status -eq 2 ]] || {
        echo "error: USRP capture failed; see $output_dir/capture.log" >&2
        exit "$capture_status"
    }
fi

[[ -f "$iq" ]] || { echo "error: IQ file not found: $iq" >&2; exit 1; }
[[ -f "$metadata" ]] || { echo "error: metadata not found: $metadata" >&2; exit 1; }

# UHD may choose a rate/frequency slightly different from the request.  Feed
# the parser what was actually captured; this is especially important for a
# B210 whose 16 MHz master clock can otherwise turn a requested 40 MS/s run
# into a mislabeled 16 MS/s file.
read -r parse_sample_rate parse_center_frequency parse_bandwidth < <(
    python3 - "$metadata" "$sample_rate" "$center_frequency" "$bandwidth" <<'PY'
import json
import sys

path, sample_rate, center_frequency, bandwidth = sys.argv[1:]
try:
    metadata = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    metadata = {}

def value(key, fallback):
    candidate = metadata.get(key)
    return candidate if isinstance(candidate, (int, float)) and candidate > 0 else fallback

print(
    value("actual_sample_rate_sps", sample_rate),
    value("actual_center_frequency_hz", center_frequency),
    value("actual_rx_bandwidth_hz", bandwidth),
)
PY
)
if [[ "$parse_sample_rate" != "$sample_rate" ]]; then
    echo "Capture actual sample rate: $parse_sample_rate S/s (requested $sample_rate)"
fi

consistency_args=()
if (( consistency_anchor )); then
    consistency_args+=(
        --consistency-anchor
        --consistency-window-us "$consistency_window_us"
        --consistency-min-pairs "$consistency_min_pairs"
        --consistency-max-addresses "$consistency_max_addresses"
    )
fi

"$parser_python" "$parser_tool" \
    --iq-path "$iq" \
    --metadata-path "$metadata" \
    --output-dir "$parse_dir" \
    --parser-python "$parser_python" \
    --parser-entrypoint "$parser_entry" \
    --ble-root "$receiver_root/vendor/BLE_encrypt_check" \
    --native-backend-dir "$receiver_root/artifacts/native/phantom_bt_native" \
    --sample-rate-sps "$parse_sample_rate" \
    --subband-sample-rate-sps 4000000 \
    --center-frequency-hz "$parse_center_frequency" \
    --bandwidth-hz "$parse_bandwidth" \
    --bootstrap-duration-s 2 \
    --known-aa-min-count "$known_aa_min_count" \
    --stage2-known-aa-bit-tolerance 0 \
    --frontend-backend auto \
    --stage1-ble-parser-backend cpp \
    --stage2-ble-parser-backend cpp \
    --measure-physical-length \
    --physical-length-backend auto \
    "${consistency_args[@]}" >"$output_dir/parser.log" 2>&1

python3 - "$parse_dir/ble_packets.csv" "$output_dir" <<'PY'
import collections
import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
hits = collections.Counter()
normal_payloads = {}
seen_frames = set()
rows_seen = 0
post_crc_rows = 0
pc_rows = 0
access_addresses = set()
with csv_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        rows_seen += 1
        if row.get("access_address"):
            access_addresses.add(row["access_address"])
        try:
            tail = bytes.fromhex(row.get("post_crc_hex", ""))
        except ValueError:
            continue
        if tail:
            post_crc_rows += 1
        position = tail.find(b"PC")
        if position < 0 or position + 6 > len(tail):
            continue
        pc_rows += 1
        length = tail[position + 4]
        frame = tail[position : position + 6 + length]
        if len(frame) != 6 + length:
            continue
        check = 0
        for byte in frame[:-1]:
            check ^= byte
        if check != frame[-1]:
            continue
        payload = frame[5 : 5 + length]
        access_address = row.get("access_address", "")
        frame_sample = (
            row.get("wideband_sample_index")
            or row.get("sample_index")
            or row.get("timestamp_s", "")
        )
        frame_key = (access_address, payload.hex(), frame_sample)
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)
        key = (access_address, payload.hex())
        hits[key] += 1
        try:
            decoded = bytes.fromhex(row.get("dewhitened_pdu_hex", ""))
            ll_length = int(row.get("payload_len", "0"))
            ll_payload = decoded[2 : 2 + ll_length]
        except (ValueError, TypeError):
            ll_payload = b""
        normal_payloads[key] = ll_payload[-2:].hex() if len(ll_payload) >= 2 else ll_payload.hex()

if not hits:
    validation_path = output_dir / "parser" / "validation_summary.json"
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        validation = {}
    anchor = validation.get("consistency_anchor") or {}
    fallback = anchor.get("fallback", "none")
    raise SystemExit(
        "error: no checksum-valid connected covert frame recovered\n"
        f"diagnostic: decoded_rows={rows_seen}, unique_access_addresses={len(access_addresses)}, "
        f"post_crc_rows={post_crc_rows}, pc_marker_rows={pc_rows}, "
        f"aa_anchor_pairs={sum(int(item.get('pair_count', 0)) for item in anchor.get('selected', []))}, "
        f"aa_discovery_fallback={fallback}\n"
        "hint: this receiver is intentionally single-channel; choose an active BLE data "
        "channel with --center-frequency, use at least 8 MS/s/8 MHz for the physical tail, "
        "or constrain the central/peer channel map to the selected channel."
    )
(access_address, covert_hex), count = hits.most_common(1)[0]
covert = bytes.fromhex(covert_hex)
try:
    text = covert.decode("utf-8")
    display = text if text.isprintable() else covert_hex
except UnicodeDecodeError:
    display = covert_hex
result = {
    "status": "passed",
    "access_address": access_address,
    "normal_payload_hex": normal_payloads[(access_address, covert_hex)],
    "covert_hex": covert_hex,
    "covert_display": display,
    "valid_frame_count": count,
}
validation_path = output_dir / "parser" / "validation_summary.json"
try:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    validation = {}
anchor = validation.get("consistency_anchor") or {}
if anchor:
    result["aa_discovery_mode"] = "empty_before_data_consistency"
    result["aa_anchor_pair_count"] = sum(
        int(item.get("pair_count", 0)) for item in anchor.get("selected", [])
    )
    result["aa_anchor_candidates"] = anchor.get("selected", [])
(output_dir / "demo_result.json").write_text(json.dumps(result, indent=2) + "\n")
print("PhantomChannel nRF52840 connected receiver")
if anchor:
    print("AA discovery: empty-before-data consistency")
    print(f"AA anchor pairs: {result['aa_anchor_pair_count']}")
print(f"Connection access address: {access_address}")
print(f"Normal payload: {result['normal_payload_hex']}")
print(f"Covert data: {display}")
if display != covert_hex:
    print(f"Covert data hex: {covert_hex}")
print(f"Valid frames: {count}")
print(f"Results: {output_dir}")
PY
