#!/usr/bin/env python3
"""Run the opt-in two-stage known-BLE-access-address parser.

The implementation lives in PhantomChannel and deliberately treats
BLE_encrypt_check as an external parser dependency.  It feeds two IQ windows
through the parser's existing stdin interface:

* the first window uses the normal blind parser and bootstraps tolerant AA
  clusters;
* the second window uses the selected AAs.  The Python backend passes them via
  ``--known-ble-aa``; the local C++ backend parses native candidates and the
  PhantomChannel remapping stage applies the same AA/Hamming filter.

No files in the BLE_encrypt_check project are modified.  Stage-2 rows are
sample-offset back to the original IQ stream before they are merged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from portable_paths import BLE_ROOT, PARSER_PYTHON  # noqa: E402

from tools.phantom_postprocess_scorer import (  # noqa: E402
    estimate_burst_duration_us,
    fill_short_false_gaps,
)


DEFAULT_BLE_ROOT = BLE_ROOT
# This is a PhantomChannel-local facade.  It compiles the read-only native
# signal source from BLE_encrypt_check and adds Phantom frame fields without
# changing that project's source tree or build directory.
DEFAULT_NATIVE_BACKEND_DIR = PROJECT_ROOT / "artifacts/native/phantom_bt_native"
# Keep the optional PhantomChannel CUDA post-processing in this project's own
# environment.  The parser source/entrypoint still comes from BLE_encrypt_check.
DEFAULT_PARSER_PYTHON = PARSER_PYTHON
DEFAULT_PARSER_ENTRYPOINT = DEFAULT_BLE_ROOT / "experiment/bt_40m_pfb_realtime.py"

# The first spelling is the BLE specification value.  The second is the
# little-endian byte order emitted by the current BLE_encrypt_check parser.
DEFAULT_ADVERTISING_ACCESS_ADDRESSES = ("8E89BED6", "D6BE898E")
DEFAULT_CHUNK_SAMPLES = 16_000_000
DEFAULT_OVERLAP_SAMPLES = 200_000
DEFAULT_SUBBAND_SAMPLE_RATE = 4_000_000.0
DEFAULT_PHY_RATE = 1_000_000.0
DEFAULT_AA_TOLERANCE_BITS = 2
DEFAULT_MIN_AA_COUNT = 10
DEFAULT_STAGE2_AA_BIT_TOLERANCE = 1
DEFAULT_STAGE1_BLE_SCORE_THRESHOLD = 3.0
DEFAULT_STAGE2_BLE_SCORE_THRESHOLD = 6.0
DEFAULT_FRONTEND_BACKEND = "auto"
SC16_BYTES_PER_SAMPLE = 4

_SEGMENT_START_RE = re.compile(r"segment_start_sample=(-?\d+)")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    materialized = list(rows)
    names: list[str] = []
    seen: set[str] = set()
    for name in fieldnames or []:
        if name not in seen:
            names.append(name)
            seen.add(name)
    for row in materialized:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def normalize_hex_id(value: Any, digits: int = 8) -> str:
    text = str(value or "").strip().replace("0x", "").replace("0X", "")
    text = "".join(ch for ch in text if ch in "0123456789abcdefABCDEF")
    if not text:
        return ""
    try:
        return f"{int(text, 16):0{digits}X}"[-digits:]
    except ValueError:
        return text.upper().zfill(digits)[-digits:]


def hamming_distance(left: str, right: str) -> int:
    if not left or not right or len(left) != len(right):
        return 999
    return (int(left, 16) ^ int(right, 16)).bit_count()


def parse_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, AttributeError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None


def row_is_connection(row: dict[str, Any]) -> bool:
    packet_type = str(row.get("packet_type") or "").upper()
    direction = str(row.get("direction_hint") or "").lower()
    pdu_type = str(row.get("ble_pdu_type") or "")
    return packet_type != "BLE_ADV" and direction != "advertising" and not pdu_type


def row_wideband_sample(row: dict[str, Any], decim: float) -> int | None:
    wide = parse_int(row.get("wideband_sample_index"))
    if wide is not None:
        return wide
    sub = parse_int(row.get("sample_index"))
    return int(round(sub * decim)) if sub is not None else None


def tolerant_access_address_clusters(
    rows: Iterable[dict[str, Any]],
    *,
    tolerance_bits: int,
    min_count: int,
    excluded_addresses: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Group parser AA candidates using the existing count-first policy.

    Representatives are selected in descending exact-observation count.  An
    observed value is assigned to the nearest representative within the
    configured Hamming radius.  Advertising AA aliases are removed before
    clustering so a one- or two-bit advertising decode cannot seed a
    connection address.
    """

    excluded = {normalize_hex_id(item) for item in excluded_addresses if normalize_hex_id(item)}
    counts: Counter[str] = Counter()
    rows_by_value: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_advertising = 0
    rejected_non_connection = 0

    for row in rows:
        if not row_is_connection(row):
            rejected_non_connection += 1
            continue
        value = normalize_hex_id(row.get("access_address"))
        if not value:
            continue
        if any(hamming_distance(value, excluded_value) <= tolerance_bits for excluded_value in excluded):
            rejected_advertising += 1
            continue
        counts[value] += 1
        rows_by_value[value].append(row)

    representatives: list[str] = []
    for value, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if any(hamming_distance(value, representative) <= tolerance_bits for representative in representatives):
            continue
        representatives.append(value)

    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for value, value_rows in rows_by_value.items():
        matching = [
            representative
            for representative in representatives
            if hamming_distance(value, representative) <= tolerance_bits
        ]
        if matching:
            representative = min(
                matching,
                key=lambda item: (hamming_distance(value, item), -counts[item], item),
            )
        else:
            # This should not occur because every exact value is considered as
            # a representative unless it is close to an earlier one, but keep
            # the fallback explicit for malformed input.
            representative = value
        groups[representative].extend(value_rows)
        aliases[representative][value] += len(value_rows)

    selected: list[dict[str, Any]] = []
    for representative, grouped_rows in groups.items():
        if len(grouped_rows) < min_count:
            continue
        samples = [
            sample
            for sample in (row_wideband_sample(row, 1.0) for row in grouped_rows)
            if sample is not None
        ]
        timestamps = [
            timestamp
            for timestamp in (parse_float(row.get("timestamp_s")) for row in grouped_rows)
            if timestamp is not None
        ]
        channels = sorted(
            {
                channel
                for channel in (parse_int(row.get("channel")) for row in grouped_rows)
                if channel is not None
            }
        )
        confidence = [
            score
            for score in (parse_float(row.get("confidence_score")) for row in grouped_rows)
            if score is not None
        ]
        selected.append(
            {
                "cluster_id": f"aa:0x{representative}",
                "canonical_access_address": f"0x{representative}",
                "canonical_access_address_hex": representative,
                "observation_count": len(grouped_rows),
                "exact_value_count": counts[representative],
                "aliases": [
                    {
                        "access_address": f"0x{value}",
                        "count": count,
                        "hamming_distance": hamming_distance(value, representative),
                    }
                    for value, count in sorted(aliases[representative].items(), key=lambda item: (-item[1], item[0]))
                ],
                "hamming_tolerance_bits": tolerance_bits,
                "first_sample": min(samples) if samples else None,
                "last_sample": max(samples) if samples else None,
                "first_timestamp_s": min(timestamps) if timestamps else None,
                "last_timestamp_s": max(timestamps) if timestamps else None,
                "channels": channels,
                "mean_confidence_score": sum(confidence) / len(confidence) if confidence else None,
            }
        )
    selected.sort(key=lambda item: (-item["observation_count"], item["canonical_access_address_hex"]))
    return selected, {
        "connection_rows_considered": sum(counts.values()),
        "unique_exact_access_addresses": len(counts),
        "rejected_advertising_rows": rejected_advertising,
        "rejected_non_connection_rows": rejected_non_connection,
        "qualifying_clusters": len(selected),
    }


def expand_stage2_known_access_addresses(
    clusters: Iterable[dict[str, Any]],
    *,
    bit_tolerance: int,
    excluded_addresses: Iterable[str],
) -> tuple[list[str], dict[str, int]]:
    """Build the exact AA table used by the stage-2 parser.

    The external BLE parser performs an exact string membership check before
    decoding the rest of a candidate.  Stage 1 already provides observations
    that have survived the blind parser, so those observed aliases are safe to
    carry forward.  We additionally add a small XOR Hamming neighbourhood
    around each selected canonical value.  Keeping synthetic variants tied to
    canonicals avoids multiplying noise from noisy aliases while widening
    recall for a one-bit AA decode error.
    """

    clusters = list(clusters)
    excluded = {normalize_hex_id(item) for item in excluded_addresses if normalize_hex_id(item)}
    observed: set[str] = set()
    for cluster in clusters:
        canonical = normalize_hex_id(cluster.get("canonical_access_address"))
        if canonical:
            observed.add(canonical)
        for alias in cluster.get("aliases", []):
            value = normalize_hex_id(alias.get("access_address")) if isinstance(alias, dict) else normalize_hex_id(alias)
            if value:
                observed.add(value)

    expanded = set(observed)
    canonicals = {
        normalize_hex_id(cluster.get("canonical_access_address"))
        for cluster in clusters
        if normalize_hex_id(cluster.get("canonical_access_address"))
    }
    if bit_tolerance:
        for value in canonicals:
            base = int(value, 16)
            # BLE access addresses are 32-bit values.  The CLI validation
            # keeps this neighbourhood intentionally small (normally 1 bit).
            for bit in range(32):
                expanded.add(f"{base ^ (1 << bit):08X}")
            if bit_tolerance >= 2:
                for left in range(32):
                    for right in range(left + 1, 32):
                        expanded.add(f"{base ^ (1 << left) ^ (1 << right):08X}")

    expanded.difference_update(excluded)
    addresses = [f"0x{value}" for value in sorted(expanded)]
    return addresses, {
        "stage1_observed_alias_count": len(observed),
        "stage2_known_address_count": len(addresses),
        "stage2_synthetic_address_count": max(0, len(addresses) - len(observed - excluded)),
    }


def parser_environment(ble_root: Path, native_backend_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = []
    if native_backend_dir is not None and native_backend_dir.is_dir():
        pythonpath.append(str(native_backend_dir))
    pythonpath.extend(
        [str(ble_root / "experiment"), str(ble_root / "ble_fun_test"), str(ble_root / "build-native")]
    )
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def cuda_device_available() -> tuple[bool, str]:
    """Probe the local CUDA runtime without changing the external parser."""

    try:
        import cupy  # type: ignore

        count = int(cupy.cuda.runtime.getDeviceCount())
        if count > 0:
            return True, f"{count} CUDA device(s) detected"
        return False, "cudaErrorNoDevice: no CUDA-capable device is detected"
    except Exception as exc:  # noqa: BLE001 - diagnostic is recorded in the manifest
        return False, f"CUDA runtime probe failed: {exc}"


def resolve_frontend_backend(args: argparse.Namespace) -> tuple[str, str]:
    requested = args.frontend_backend
    available, diagnostic = cuda_device_available()
    if requested == "cuda":
        if not available:
            raise RuntimeError(f"--frontend-backend cuda requested but unavailable: {diagnostic}")
        return "cuda", diagnostic
    if requested == "cpu":
        return "cpu", "explicit CPU frontend requested"
    if available:
        return "cuda", diagnostic
    return "cpu", f"auto fallback to CPU: {diagnostic}"


def build_parser_command(
    args: argparse.Namespace,
    output_dir: Path,
    known_addresses: list[str],
    ble_parser_backend: str,
    score_threshold: float | None = None,
) -> list[str]:
    if score_threshold is None:
        score_threshold = DEFAULT_STAGE1_BLE_SCORE_THRESHOLD
    # The external parser's BLE demodulator is fixed to one bit per nominal
    # 1-Mbit/s symbol, so a 2-M PHY is represented by a 2-MS/s channel stream.
    # Keep the historical 4-MS/s filter settings unchanged; for the explicit
    # 2-MS/s diagnostic path, lower both cutoffs below Nyquist instead of
    # letting the external PFB reject the command before it reads IQ.
    if args.subband_sample_rate_sps < 3_000_000.0:
        ble_lpf_cutoff = min(900_000.0, args.subband_sample_rate_sps * 0.45)
        pfb_cutoff = min(900_000.0, args.subband_sample_rate_sps * 0.45)
    else:
        ble_lpf_cutoff = 1_000_000.0
        pfb_cutoff = 1_900_000.0
    command = [
        str(args.parser_python),
        str(args.parser_entrypoint),
        "--source",
        "stdin",
        "--timestamp-mode",
        "sample_index",
        "--sample-rate",
        str(args.sample_rate_sps),
        "--center-freq",
        str(args.center_frequency_hz),
        "--bandwidth",
        str(args.bandwidth_hz),
        "--subband-sample-rate",
        str(args.subband_sample_rate_sps),
        "--iq-format",
        "int16",
        "--output-dir",
        str(output_dir),
        "--chunk-samples",
        str(args.chunk_samples),
        "--overlap-samples",
        str(args.overlap_samples),
        "--ble-candidate-detector",
        "fixed",
        "--ble-threshold",
        "0.01",
        "--ble-score-threshold",
        str(score_threshold),
        "--ble-lpf-cutoff",
        str(ble_lpf_cutoff),
        "--pfb-cutoff",
        str(pfb_cutoff),
        "--cleanup-numtaps",
        "31",
        "--cuda-device",
        str(args.cuda_device),
        "--cuda-pfb-backend",
        "kernel_multi_float_phase_t",
        "--cuda-cleanup-backend",
        "kernel_multi_float",
        "--cuda-target-dsp-materialization",
        "full",
        "--cuda-segment-copy-merge-gap-samples",
        "4096",
        "--cuda-segment-copy-max-merged-samples",
        "262144",
        "--ble-parser-backend",
        ble_parser_backend,
        "--bredr-parser-backend",
        "hybrid",
        "--cpp-parser-threads",
        str(args.cpp_parser_threads),
        "--native-segment-input",
        "legacy",
        "--target-selection",
        "full",
        "--cuda-fuse-target-dsp",
        "--cuda-batch-targets",
        "--cuda-threshold-detect",
        "--cuda-known-candidate-filter",
        "--no-learned-parser-fast-path",
        "--skip-bredr",
        "--quiet",
        "--timing",
    ]
    if getattr(args, "_frontend_backend_used", "cuda") == "cuda":
        command.append("--use-cuda")
    else:
        command.append("--no-use-cuda")
    # The unmodified BLE_encrypt_check native wrapper rejects known-AA
    # arguments.  For the local C++ stage we therefore parse with its native
    # candidate decoder and apply the known-AA filter in PhantomChannel during
    # remapping.  The Python backend continues to receive the fast-path list.
    if known_addresses and ble_parser_backend == "python":
        command.extend(["--known-ble-aa", ",".join(known_addresses)])
    if args.parser_cpus:
        command = ["taskset", "-c", args.parser_cpus] + command
    return command


def stream_iq_window(
    process: subprocess.Popen[bytes],
    iq_path: Path,
    start_sample: int,
    end_sample: int,
) -> int:
    sent = 0
    start_byte = max(0, int(start_sample)) * SC16_BYTES_PER_SAMPLE
    end_byte = max(start_byte, int(end_sample) * SC16_BYTES_PER_SAMPLE)
    with iq_path.open("rb") as source:
        source.seek(start_byte)
        remaining = end_byte - start_byte
        try:
            assert process.stdin is not None
            while remaining > 0:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    break
                process.stdin.write(block)
                sent += len(block)
                remaining -= len(block)
            process.stdin.close()
        except BrokenPipeError:
            if process.stdin is not None:
                process.stdin.close()
    return sent // SC16_BYTES_PER_SAMPLE


def run_parser_window(
    args: argparse.Namespace,
    iq_path: Path,
    output_dir: Path,
    start_sample: int,
    end_sample: int,
    known_addresses: list[str],
    ble_parser_backend: str,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"parser window output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    command = build_parser_command(
        args,
        output_dir,
        known_addresses,
        ble_parser_backend,
        score_threshold=score_threshold,
    )
    command_path = output_dir / "command.txt"
    log_path = output_dir / "parse.log"
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    started = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=args.ble_root,
            env=parser_environment(args.ble_root, args.native_backend_dir),
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        sent_samples = stream_iq_window(process, iq_path, start_sample, end_sample)
        returncode = process.wait()
    ended = time.time_ns()
    outputs = {
        name: output_dir / name
        for name in ("ble_packets.csv", "btclassic_packets.csv", "packet_events.csv", "target_selection.csv")
    }
    result = {
        "start_sample": start_sample,
        "end_sample": end_sample,
        "requested_samples": max(0, end_sample - start_sample),
        "sent_samples": sent_samples,
        "known_access_addresses": known_addresses if ble_parser_backend == "python" else [],
        "known_access_addresses_filtered_in_phantom": (
            known_addresses if ble_parser_backend == "cpp" else []
        ),
        "native_backend_dir": str(args.native_backend_dir),
        "ble_parser_backend": ble_parser_backend,
        "ble_score_threshold": (
            DEFAULT_STAGE1_BLE_SCORE_THRESHOLD if score_threshold is None else score_threshold
        ),
        "returncode": returncode,
        "start_epoch_ns": started,
        "end_epoch_ns": ended,
        "elapsed_s": (ended - started) / 1e9,
        "command": command,
        "command_path": str(command_path),
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_exists": {name: path.is_file() for name, path in outputs.items()},
        "valid": returncode == 0 and outputs["ble_packets.csv"].is_file(),
    }
    write_json(output_dir / "parse_status.json", result)
    return result


def clean_hex(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch in "0123456789abcdefABCDEF")


def decode_phantom_frame(hex_text: Any) -> dict[str, Any] | None:
    """Decode the local Phantom ``PC`` frame from a post-CRC byte stream."""

    try:
        data = bytes.fromhex(clean_hex(hex_text))
    except ValueError:
        return None
    first_candidate: dict[str, Any] | None = None
    for start in range(max(0, len(data) - 5)):
        if data[start : start + 2] != b"PC" or start + 6 > len(data):
            continue
        covert_len = data[start + 4]
        end = start + 5 + covert_len + 1
        if end > len(data):
            continue
        frame = data[start:end]
        checksum = 0
        for value in frame[:-1]:
            checksum ^= value
        payload = frame[5:-1]
        candidate = {
            "covert_len": len(payload),
            "covert_hex": payload.hex().upper(),
            "covert_marker_hex": payload[:1].hex().upper(),
            "covert_data_len": max(0, len(payload) - 1),
            "covert_data_hex": payload[1:].hex().upper(),
            "covert_data": payload[1:].hex().upper(),
            "covert_frame_hex": frame.hex().upper(),
            "covert_seq": data[start + 2] | (data[start + 3] << 8),
            "covert_integrity_ok": "1" if checksum == frame[-1] else "0",
        }
        if candidate["covert_integrity_ok"] == "1":
            return candidate
        if first_candidate is None:
            first_candidate = candidate
    return first_candidate


def remap_stage2_row(
    row: dict[str, Any],
    *,
    source_start_sample: int,
    boundary_sample: int,
    total_samples: int,
    decim: float,
    sample_rate_sps: float,
    subband_sample_rate_sps: float,
    phy_rate_sps: float,
    known_clusters: list[dict[str, Any]],
    tolerance_bits: int,
) -> dict[str, Any] | None:
    local_wide = row_wideband_sample(row, decim)
    if local_wide is None:
        return None
    global_wide = local_wide + source_start_sample
    if global_wide < boundary_sample or global_wide >= total_samples:
        return None
    observed_aa = normalize_hex_id(row.get("access_address"))
    if not observed_aa:
        return None
    candidates = [
        (hamming_distance(observed_aa, normalize_hex_id(cluster["canonical_access_address"])), cluster)
        for cluster in known_clusters
    ]
    if not candidates:
        return None
    aa_distance, cluster = min(candidates, key=lambda item: (item[0], item[1]["canonical_access_address_hex"]))
    if aa_distance > tolerance_bits:
        return None

    remapped = dict(row)
    remapped["parse_mode"] = "known_aa_stage2"
    remapped["known_aa_cluster_id"] = cluster["cluster_id"]
    remapped["known_aa_canonical"] = cluster["canonical_access_address"]
    remapped["aa_hamming_distance"] = aa_distance
    remapped["cross_layer_consistency_required"] = "false"
    remapped["known_aa_match"] = "true"
    remapped["wideband_sample_index"] = global_wide
    remapped["sample_index"] = int(round(global_wide / decim))
    remapped["subband_sample_index"] = int(round(global_wide / decim))
    timestamp_s = global_wide / sample_rate_sps
    remapped["timestamp_s"] = f"{timestamp_s:.9f}"
    remapped["timestamp_us"] = f"{timestamp_s * 1e6:.3f}"
    remapped["timestamp_status"] = "sample_index_global_remap"

    covert_fields = decode_phantom_frame(row.get("post_crc_hex"))
    if covert_fields is None:
        covert_fields = {
            "covert_len": parse_int(row.get("covert_len")) or 0,
            "covert_hex": clean_hex(row.get("covert_hex")),
            "covert_marker_hex": clean_hex(row.get("covert_marker_hex")),
            "covert_data_len": parse_int(row.get("covert_data_len")) or 0,
            "covert_data_hex": clean_hex(row.get("covert_data_hex") or row.get("covert_data")),
            "covert_data": clean_hex(row.get("covert_data") or row.get("covert_data_hex")),
            "covert_frame_hex": clean_hex(row.get("covert_frame_hex")),
            "covert_seq": row.get("covert_seq", ""),
            "covert_integrity_ok": row.get("covert_integrity_ok", ""),
        }
    remapped.update(covert_fields)

    offset_match = _SEGMENT_START_RE.search(str(row.get("raw_offset_info") or ""))
    if offset_match:
        local_segment_start = int(offset_match.group(1))
        global_segment_start = local_segment_start + int(round(source_start_sample / decim))
        remapped["raw_offset_info"] = _SEGMENT_START_RE.sub(
            f"segment_start_sample={global_segment_start}",
            str(row.get("raw_offset_info") or ""),
        )

    payload_len = parse_int(row.get("payload_len"))
    tail_hex = clean_hex(row.get("post_crc_hex"))
    tail_bytes = len(tail_hex) // 2 if len(tail_hex) % 2 == 0 else 0
    link_bytes = payload_len + 10 if payload_len is not None and 0 <= payload_len <= 255 else None
    samples_per_bit = sample_rate_sps / phy_rate_sps if phy_rate_sps > 0 else None
    link_samples = int(round(link_bytes * 8 * samples_per_bit)) if link_bytes is not None and samples_per_bit else None
    tail_samples = int(round(tail_bytes * 8 * samples_per_bit)) if samples_per_bit is not None else None
    remapped["physical_burst_start_sample"] = global_wide
    remapped["link_layer_length_bytes"] = link_bytes if link_bytes is not None else ""
    remapped["link_layer_end_sample_estimate"] = (
        global_wide + link_samples if link_samples is not None else ""
    )
    remapped["post_crc_or_tail_length_bytes"] = tail_bytes
    remapped["tail_length_samples_estimate"] = tail_samples if tail_samples is not None else ""
    remapped["physical_burst_end_sample_estimate"] = (
        global_wide + link_samples + tail_samples
        if link_samples is not None and tail_samples is not None
        else ""
    )
    remapped["tail_length_estimation_method"] = (
        "post_crc_demodulated" if tail_bytes else "no_post_crc_bytes"
    )
    remapped["physical_burst_duration_us_observed"] = ""
    remapped["physical_burst_end_sample_observed"] = ""
    remapped["standard_link_layer_duration_us"] = (
        link_bytes * 8 * 1_000_000.0 / phy_rate_sps
        if link_bytes is not None and phy_rate_sps > 0
        else ""
    )
    remapped["physical_residual_tail_duration_us"] = ""
    remapped["physical_residual_tail_length_bytes"] = ""
    remapped["physical_length_estimation_method"] = "not_measured"
    remapped["physical_length_backend"] = "not_measured"
    remapped["sample_rate_sps_for_length"] = sample_rate_sps
    remapped["subband_sample_rate_sps_for_length"] = subband_sample_rate_sps
    remapped["phy_rate_sps_for_length"] = phy_rate_sps
    return remapped


def add_stage1_fields(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    covert_fields = decode_phantom_frame(row.get("post_crc_hex"))
    if covert_fields is not None:
        result.update(covert_fields)
    else:
        result.update(
            {
                "covert_len": row.get("covert_len", ""),
                "covert_hex": row.get("covert_hex", ""),
                "covert_marker_hex": row.get("covert_marker_hex", ""),
                "covert_data_len": row.get("covert_data_len", ""),
                "covert_data_hex": row.get("covert_data_hex", ""),
                "covert_data": row.get("covert_data", ""),
                "covert_frame_hex": row.get("covert_frame_hex", ""),
                "covert_seq": row.get("covert_seq", ""),
                "covert_integrity_ok": row.get("covert_integrity_ok", ""),
            }
        )
    result.update(
        {
            "parse_mode": "blind_stage1",
            "known_aa_cluster_id": "",
            "known_aa_canonical": "",
            "aa_hamming_distance": "",
            "cross_layer_consistency_required": "true",
            "known_aa_match": "",
            "physical_burst_start_sample": "",
            "link_layer_length_bytes": "",
            "link_layer_end_sample_estimate": "",
            "post_crc_or_tail_length_bytes": "",
            "tail_length_samples_estimate": "",
            "physical_burst_end_sample_estimate": "",
            "tail_length_estimation_method": "not_stage2",
            "physical_burst_duration_us_observed": "",
            "physical_burst_end_sample_observed": "",
            "standard_link_layer_duration_us": "",
            "physical_residual_tail_duration_us": "",
            "physical_residual_tail_length_bytes": "",
            "physical_length_estimation_method": "not_stage2",
            "physical_length_backend": "not_stage2",
            "sample_rate_sps_for_length": "",
            "subband_sample_rate_sps_for_length": "",
            "phy_rate_sps_for_length": "",
        }
    )
    return result


def deduplicate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("parse_mode", ""),
            row.get("wideband_sample_index", ""),
            normalize_hex_id(row.get("access_address")),
            row.get("channel", ""),
        )
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        previous_score = parse_float(previous.get("confidence_score")) or 0.0
        current_score = parse_float(row.get("confidence_score")) or 0.0
        if current_score > previous_score:
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            parse_int(row.get("wideband_sample_index")) or 0,
            parse_int(row.get("channel")) or -1,
            str(row.get("access_address") or ""),
        ),
    )


def merge_event_rows(
    stage1_dir: Path,
    stage2_dir: Path,
    *,
    stage2_source_start_sample: int,
    boundary_sample: int,
    total_samples: int,
    decim: float,
    args: argparse.Namespace,
    known_clusters: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    for path in (stage1_dir / "packet_events.csv", stage2_dir / "packet_events.csv"):
        if not path.is_file():
            continue
        window_rows, window_fields = read_csv(path)
        if path.parent == stage1_dir:
            rows.extend(add_stage1_fields(row) for row in window_rows)
        else:
            for row in window_rows:
                remapped = remap_stage2_row(
                    row,
                    source_start_sample=stage2_source_start_sample,
                    boundary_sample=boundary_sample,
                    total_samples=total_samples,
                    decim=decim,
                    sample_rate_sps=args.sample_rate_sps,
                    subband_sample_rate_sps=args.subband_sample_rate_sps,
                    phy_rate_sps=args.phy_rate_sps,
                    known_clusters=known_clusters,
                    tolerance_bits=args.known_aa_hamming_tolerance,
                )
                if remapped is not None:
                    rows.append(remapped)
        for field in window_fields:
            if field not in fields:
                fields.append(field)
    return rows, fields


def measure_physical_length(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Measure channel energy and subtract the decoded standard BLE duration.

    This is an IQ-duration diagnostic, not a covert-frame acceptance decision.
    The external parser currently does not export a physical segment end and
    its Python known-AA backend does not export post-CRC bytes, so this is the
    PhantomChannel-side fallback.
    """

    packet_start = parse_int(row.get("physical_burst_start_sample"))
    channel = parse_int(row.get("channel"))
    row["physical_length_backend"] = "cpu"
    if packet_start is None or channel is None or not 0 <= channel <= 36:
        row["physical_length_estimation_method"] = "invalid_start_or_channel"
        return row
    packet_frequency_hz = parse_float(row.get("subband_freq_mhz"))
    if packet_frequency_hz is None:
        packet_frequency_hz = 2402.0 + channel
    else:
        packet_frequency_hz *= 1e6
    measurement = estimate_burst_duration_us(
        args.iq_path,
        sample_rate_hz=args.sample_rate_sps,
        center_frequency_hz=args.center_frequency_hz,
        packet_start_sample=packet_start,
        packet_frequency_hz=packet_frequency_hz,
        expected_phantom_us=args.length_expected_max_us,
        pre_margin_us=args.length_pre_margin_us,
        post_margin_us=args.length_post_margin_us,
        lowpass_hz=args.length_lowpass_hz,
        smooth_us=args.length_smooth_us,
        threshold_sigma=args.length_threshold_sigma,
        min_threshold_ratio=args.length_min_threshold_ratio,
    )
    measured_us = parse_float(measurement.get("measured_duration_us"))
    if measured_us is None:
        row["physical_length_estimation_method"] = (
            f"iq_energy_unavailable:{measurement.get('notes', 'unknown')}"
        )
        return row
    if "duration_hits_search_end" in str(measurement.get("notes") or ""):
        row["physical_burst_duration_us_observed"] = f"{measured_us:.3f}"
        row["physical_burst_end_sample_observed"] = ""
        row["physical_residual_tail_duration_us"] = ""
        row["physical_residual_tail_length_bytes"] = ""
        row["physical_length_estimation_method"] = "iq_energy_unbounded_search_end"
        row["physical_length_measurement_notes"] = measurement.get("notes", "")
        row["physical_length_measurement_threshold"] = measurement.get("threshold", "")
        row["physical_length_measurement_confidence_db"] = measurement.get("duration_confidence", "")
        return row
    standard_us = parse_float(row.get("standard_link_layer_duration_us"))
    if standard_us is None:
        row["physical_length_estimation_method"] = "iq_energy_missing_link_length"
        return row
    residual_us = measured_us - standard_us
    residual_bytes = max(
        0,
        int(round(residual_us * args.phy_rate_sps / (8.0 * 1_000_000.0))),
    )
    row["physical_burst_duration_us_observed"] = f"{measured_us:.3f}"
    row["physical_burst_end_sample_observed"] = packet_start + int(
        round(measured_us * args.sample_rate_sps / 1_000_000.0)
    )
    row["physical_residual_tail_duration_us"] = f"{residual_us:.3f}"
    row["physical_residual_tail_length_bytes"] = residual_bytes
    row["physical_length_estimation_method"] = "iq_energy_minus_link_layer"
    row["physical_length_measurement_notes"] = measurement.get("notes", "")
    row["physical_length_measurement_threshold"] = measurement.get("threshold", "")
    row["physical_length_measurement_confidence_db"] = measurement.get("duration_confidence", "")
    return row


def _apply_duration_measurement(
    row: dict[str, Any],
    args: argparse.Namespace,
    measurement: dict[str, Any],
    *,
    method_prefix: str,
) -> dict[str, Any]:
    """Apply one CPU/GPU duration result using the same acceptance rules."""

    row["physical_length_backend"] = "cuda" if method_prefix.startswith("iq_energy_cuda") else "cpu"
    measured_us = parse_float(measurement.get("measured_duration_us"))
    if measured_us is None:
        row["physical_length_estimation_method"] = (
            f"{method_prefix}_unavailable:{measurement.get('notes', 'unknown')}"
        )
        return row
    if "duration_hits_search_end" in str(measurement.get("notes") or ""):
        row["physical_burst_duration_us_observed"] = f"{measured_us:.3f}"
        row["physical_burst_end_sample_observed"] = ""
        row["physical_residual_tail_duration_us"] = ""
        row["physical_residual_tail_length_bytes"] = ""
        row["physical_length_estimation_method"] = f"{method_prefix}_unbounded_search_end"
        row["physical_length_measurement_notes"] = measurement.get("notes", "")
        row["physical_length_measurement_threshold"] = measurement.get("threshold", "")
        row["physical_length_measurement_confidence_db"] = measurement.get("duration_confidence", "")
        return row
    packet_start = parse_int(row.get("physical_burst_start_sample"))
    standard_us = parse_float(row.get("standard_link_layer_duration_us"))
    if packet_start is None or standard_us is None:
        row["physical_length_estimation_method"] = f"{method_prefix}_missing_link_length"
        return row
    residual_us = measured_us - standard_us
    residual_bytes = max(
        0,
        int(round(residual_us * args.phy_rate_sps / (8.0 * 1_000_000.0))),
    )
    row["physical_burst_duration_us_observed"] = f"{measured_us:.3f}"
    row["physical_burst_end_sample_observed"] = packet_start + int(
        round(measured_us * args.sample_rate_sps / 1_000_000.0)
    )
    row["physical_residual_tail_duration_us"] = f"{residual_us:.3f}"
    row["physical_residual_tail_length_bytes"] = residual_bytes
    row["physical_length_estimation_method"] = f"{method_prefix}_minus_link_layer"
    row["physical_length_measurement_notes"] = measurement.get("notes", "")
    row["physical_length_measurement_threshold"] = measurement.get("threshold", "")
    row["physical_length_measurement_confidence_db"] = measurement.get("duration_confidence", "")
    return row


def _measure_physical_lengths_cuda(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Measure many burst windows on CUDA without changing the external parser.

    This follows the same signal path as ``smoothed_channel_power`` in the
    local PhantomChannel scorer—per-channel frequency shift, 129-tap FIR,
    power and moving-average smoothing—but processes a batch of equal-sized
    windows on the GPU.  The raw SC16 read and final scalar decisions remain on
    the host.  The external BLE_encrypt_check CUDA channelizer is intentionally
    not imported or modified here.
    """

    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cupy_ndimage
        from cupyx.scipy import signal as cupy_signal
    except Exception as exc:  # pragma: no cover - depends on host CUDA setup
        raise RuntimeError(f"CUDA physical-length backend unavailable: {exc}") from exc

    if not rows:
        return
    if args.sample_rate_sps <= 0:
        raise ValueError("sample rate must be positive for CUDA physical-length measurement")

    pre_samples = int(round(args.sample_rate_sps * args.length_pre_margin_us / 1_000_000.0))
    expected_samples = int(round(args.sample_rate_sps * args.length_expected_max_us / 1_000_000.0))
    post_samples = int(round(args.sample_rate_sps * args.length_post_margin_us / 1_000_000.0))
    search_samples = expected_samples + post_samples
    window_samples = pre_samples + search_samples
    smooth_samples = max(1, int(round(args.sample_rate_sps * args.length_smooth_us / 1_000_000.0)))
    gap_samples = int(round(args.sample_rate_sps * 2.0 / 1_000_000.0))
    batch_size = max(1, int(args.length_cuda_batch_size))
    if pre_samples < 20 or search_samples <= 0:
        raise ValueError("invalid physical-length CUDA window configuration")

    taps = np.asarray(signal.firwin(129, args.length_lowpass_hz, fs=args.sample_rate_sps), dtype=np.float32)
    raw = np.memmap(args.iq_path, dtype="<i2", mode="r")
    complex_samples = raw.size // 2

    records: list[tuple[dict[str, Any], int, int, float]] = []
    for row in rows:
        packet_start = parse_int(row.get("physical_burst_start_sample"))
        channel = parse_int(row.get("channel"))
        if packet_start is None or channel is None or not 0 <= channel <= 36:
            measure_physical_length(row, args)
            continue
        if packet_start < pre_samples:
            # The start-of-file short-noise case has a variable noise region;
            # retain the proven scalar implementation for that rare case.
            measure_physical_length(row, args)
            continue
        packet_frequency_hz = parse_float(row.get("subband_freq_mhz"))
        packet_frequency_hz = (
            (packet_frequency_hz * 1e6)
            if packet_frequency_hz is not None
            else (2402.0 + channel) * 1e6
        )
        records.append((row, packet_start, channel, packet_frequency_hz))

    if not records:
        return

    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        host = np.zeros((len(batch), window_samples), dtype=np.complex64)
        valid_lengths = np.zeros(len(batch), dtype=np.int32)
        frequency_offsets = np.empty(len(batch), dtype=np.float32)
        for index, (_row, packet_start, _channel, packet_frequency_hz) in enumerate(batch):
            start_sample = packet_start - pre_samples
            available = max(0, min(window_samples, complex_samples - start_sample))
            if available:
                interleaved = np.asarray(
                    raw[start_sample * 2 : (start_sample + available) * 2],
                    dtype=np.float32,
                )
                host[index, :available] = interleaved[0::2] + 1j * interleaved[1::2]
            valid_lengths[index] = available
            frequency_offsets[index] = packet_frequency_hz - args.center_frequency_hz

        with cp.cuda.Device(args.cuda_device):
            # Allocate all device-side constants after selecting the requested
            # device.  This matters when --cuda-device is not device 0.
            cp_taps = cp.asarray(taps)
            cp_den = cp.asarray([1.0], dtype=cp.float32)
            cp_n = cp.arange(window_samples, dtype=cp.float32)
            cp_indices = cp.arange(search_samples, dtype=cp.int32)[None, :]
            iq_gpu = cp.asarray(host)
            offsets_gpu = cp.asarray(frequency_offsets)
            shifted = iq_gpu * cp.exp(
                (-2j * cp.pi * offsets_gpu[:, None] / args.sample_rate_sps)
                * cp_n[None, :]
            )
            filtered = cupy_signal.lfilter(cp_taps, cp_den, shifted, axis=1).astype(
                cp.complex64,
                copy=False,
            )
            power = (cp.abs(filtered) ** 2).astype(cp.float32, copy=False)
            smoothed = cupy_ndimage.uniform_filter1d(
                power,
                size=smooth_samples,
                axis=1,
                mode="constant",
                cval=0.0,
            )
            noise = smoothed[:, :pre_samples]
            noise_floor = cp.median(noise, axis=1)
            mad = cp.median(cp.abs(noise - noise_floor[:, None]), axis=1)
            sigma = 1.4826 * mad
            threshold = noise_floor + cp.maximum(
                float(args.length_threshold_sigma) * sigma,
                noise_floor * float(args.length_min_threshold_ratio),
            )
            search = smoothed[:, pre_samples : pre_samples + search_samples]
            valid_search_lengths = cp.minimum(
                cp.asarray(valid_lengths, dtype=cp.int32) - pre_samples,
                cp.int32(search_samples),
            )
            valid_search_lengths = cp.maximum(valid_search_lengths, cp.int32(0))
            valid_mask = cp_indices < valid_search_lengths[:, None]
            above = (search > threshold[:, None]) & valid_mask
            peak_power = cp.max(cp.where(valid_mask, search, cp.float32(0.0)), axis=1)
            confidence = 10.0 * cp.log10((peak_power + 1e-12) / (threshold + 1e-12))

            # CuPy 14 does not implement ufunc.accumulate for maximum/minimum.
            # Transfer only this compact boolean mask and apply the existing
            # short-gap rule on the host; FIR/power/smoothing remain on CUDA.
            above_host = cp.asnumpy(above)
            noise_floor_host = cp.asnumpy(noise_floor)
            threshold_host = cp.asnumpy(threshold)
            peak_power_host = cp.asnumpy(peak_power)
            confidence_host = cp.asnumpy(confidence)
            valid_search_host = cp.asnumpy(valid_search_lengths)

        for index, (row, _packet_start, _channel, _frequency) in enumerate(batch):
            filled = fill_short_false_gaps(above_host[index], gap_samples)
            energy_indices = np.flatnonzero(filled)
            if energy_indices.size == 0:
                measurement = {
                    "measured_duration_us": "",
                    "threshold": float(threshold_host[index]),
                    "noise_floor": float(noise_floor_host[index]),
                    "peak_power": float(peak_power_host[index]),
                    "notes": "no_energy_above_threshold",
                }
            else:
                end_offset = int(energy_indices[-1]) + 1
                measurement = {
                    "measured_duration_us": end_offset * 1_000_000.0 / args.sample_rate_sps,
                    "threshold": float(threshold_host[index]),
                    "noise_floor": float(noise_floor_host[index]),
                    "peak_power": float(peak_power_host[index]),
                    "duration_confidence": float(confidence_host[index]),
                    "notes": (
                        "duration_hits_search_end"
                        if end_offset >= int(valid_search_host[index]) - 1
                        else ""
                    ),
                }
            _apply_duration_measurement(
                row,
                args,
                measurement,
                method_prefix="iq_energy_cuda",
            )

        del iq_gpu, shifted, filtered, power, smoothed
        cp.get_default_memory_pool().free_all_blocks()


def measure_physical_lengths(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    """Measure all stage-2 rows, preferring the local CUDA batch path."""

    backend = args.physical_length_backend
    args._physical_length_backend_diagnostic = {
        "requested_backend": backend,
        "rows": len(rows),
    }
    if backend == "cpu":
        for row in rows:
            measure_physical_length(row, args)
        args._physical_length_backend_diagnostic.update(
            {"backend": "cpu", "fallback": False, "reason": "explicit_cpu"}
        )
        return "cpu"
    try:
        _measure_physical_lengths_cuda(rows, args)
        args._physical_length_backend_diagnostic.update(
            {"backend": "cuda", "fallback": False}
        )
        return "cuda"
    except Exception as exc:
        if backend == "cuda":
            raise
        # ``auto`` must remain usable on hosts without CuPy/CUDA.  Recompute
        # every row with the scalar implementation in case the CUDA path had
        # already updated part of a batch before a device/runtime failure.
        for row in rows:
            measure_physical_length(row, args)
        args._physical_length_backend_diagnostic.update(
            {
                "backend": "cpu",
                "fallback": True,
                "reason": str(exc),
            }
        )
        return "cpu_fallback"


def build_summary(
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    boundary_sample: int,
    total_samples: int,
    stage1_rows: list[dict[str, Any]],
    stage2_raw_rows: list[dict[str, Any]],
    merged_rows: list[dict[str, Any]],
    known_clusters: list[dict[str, Any]],
    known_addresses: list[str],
    known_address_stats: dict[str, int],
    cluster_stats: dict[str, int],
    stage1_status: dict[str, Any],
    stage2_status: dict[str, Any],
) -> dict[str, Any]:
    stage2_rows = [row for row in merged_rows if row.get("parse_mode") == "known_aa_stage2"]
    address_counts = Counter(row.get("known_aa_canonical", "") for row in stage2_rows)
    covert_rows = [
        row
        for row in stage2_rows
        if (parse_int(row.get("covert_data_len")) or 0) > 0
    ]
    tail_rows = [
        row
        for row in stage2_rows
        if (parse_int(row.get("post_crc_or_tail_length_bytes")) or 0) > 0
    ]
    return {
        "schema_version": 1,
        "mode": "two_stage_known_access_address",
        "ble_encrypt_check_modified": False,
        "frontend_backend_requested": args.frontend_backend,
        "frontend_backend_used": getattr(args, "_frontend_backend_used", "unknown"),
        "frontend_backend_diagnostic": getattr(args, "_frontend_backend_diagnostic", ""),
        "stage1_ble_parser_backend_requested": getattr(
            args, "_stage1_ble_parser_backend_requested", args.stage1_ble_parser_backend
        ),
        "stage1_ble_parser_backend_used": args.stage1_ble_parser_backend,
        "stage2_ble_parser_backend_requested": getattr(
            args, "_stage2_ble_parser_backend_requested", args.stage2_ble_parser_backend
        ),
        "stage2_ble_parser_backend_used": args.stage2_ble_parser_backend,
        "input_iq": str(args.iq_path),
        "metadata_path": str(args.metadata_path),
        "sample_rate_sps": args.sample_rate_sps,
        "subband_sample_rate_sps": args.subband_sample_rate_sps,
        "boundary": {
            "bootstrap_duration_s": args.bootstrap_duration_s,
            "boundary_sample": boundary_sample,
            "boundary_time_s": boundary_sample / args.sample_rate_sps,
            "total_samples": total_samples,
            "total_duration_s": total_samples / args.sample_rate_sps,
            "stage2_overlap_samples": args.overlap_samples,
        },
        "bootstrap": {
            "rows": len(stage1_rows),
            "cluster_stats": cluster_stats,
            "minimum_observations": args.known_aa_min_count,
            "hamming_tolerance_bits": args.known_aa_hamming_tolerance,
            "excluded_access_addresses": args.exclude_access_address,
            "selected_address_count": len(known_clusters),
            "selected_addresses": [cluster["canonical_access_address"] for cluster in known_clusters],
            "stage2_known_address_count": len(known_addresses),
            "stage2_known_address_stats": known_address_stats,
            "stage2_known_aa_bit_tolerance": args.stage2_known_aa_bit_tolerance,
        },
        "stage1": stage1_status,
        "stage2": {
            **stage2_status,
            "ble_score_threshold": args.stage2_ble_score_threshold,
            "raw_parser_rows": len(stage2_raw_rows),
            "accepted_known_address_rows": len(stage2_rows),
            "rows_by_known_address": dict(sorted(address_counts.items())),
            "tail_rows_with_post_crc_bytes": len(tail_rows),
            "tail_bytes_from_post_crc": sum(
                parse_int(row.get("post_crc_or_tail_length_bytes")) or 0 for row in tail_rows
            ),
            "covert_frame_rows": sum(bool(row.get("covert_frame_hex")) for row in stage2_rows),
            "covert_integrity_ok_rows": sum(
                str(row.get("covert_integrity_ok", "")) == "1" for row in stage2_rows
            ),
            "covert_data_rows": len(covert_rows),
            "covert_data_bytes": sum(
                parse_int(row.get("covert_data_len")) or 0 for row in covert_rows
            ),
            "physical_length_backend_requested": args.physical_length_backend,
            "physical_length_backend_used": getattr(args, "_physical_length_backend_used", "disabled"),
            "physical_length_backend_diagnostic": getattr(
                args, "_physical_length_backend_diagnostic", {}
            ),
            "physical_measurement_rows": sum(
                str(row.get("physical_length_estimation_method", "")).startswith("iq_energy_")
                for row in stage2_rows
            ),
            "physical_unbounded_rows": sum(
                "unbounded_search_end" in str(row.get("physical_length_estimation_method", ""))
                for row in stage2_rows
            ),
            "physical_length_rows": sum(
                row.get("physical_length_estimation_method") == "iq_energy_minus_link_layer"
                or row.get("physical_length_estimation_method") == "iq_energy_cuda_minus_link_layer"
                for row in stage2_rows
            ),
            "physical_positive_residual_rows": sum(
                row.get("physical_length_estimation_method") in (
                    "iq_energy_minus_link_layer",
                    "iq_energy_cuda_minus_link_layer",
                )
                and (parse_float(row.get("physical_residual_tail_duration_us")) or 0.0) > 0
                for row in stage2_rows
            ),
            "physical_residual_tail_bytes": sum(
                (parse_int(row.get("physical_residual_tail_length_bytes")) or 0)
                if row.get("physical_length_estimation_method") in (
                    "iq_energy_minus_link_layer",
                    "iq_energy_cuda_minus_link_layer",
                )
                else 0
                for row in stage2_rows
            ),
        },
        "merged": {
            "rows": len(merged_rows),
            "stage1_rows": sum(row.get("parse_mode") == "blind_stage1" for row in merged_rows),
            "stage2_rows": len(stage2_rows),
        },
        "metadata_capture_id": metadata.get("capture_id", ""),
        "warnings": [
            "Stage-2 physical burst end is measured from IQ channel energy when --measure-physical-length is enabled; search-window saturation is marked unbounded and excluded from residual-byte totals.",
            "A known access-address match is not counted as covert recovery without post-CRC/frame validation.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parser-python", type=Path, default=DEFAULT_PARSER_PYTHON)
    parser.add_argument("--parser-entrypoint", type=Path, default=DEFAULT_PARSER_ENTRYPOINT)
    parser.add_argument("--ble-root", type=Path, default=DEFAULT_BLE_ROOT)
    parser.add_argument(
        "--native-backend-dir",
        type=Path,
        default=DEFAULT_NATIVE_BACKEND_DIR,
        help="PhantomChannel-local bt_native extension directory",
    )
    parser.add_argument("--sample-rate-sps", type=float, default=100_000_000.0)
    parser.add_argument("--subband-sample-rate-sps", type=float, default=DEFAULT_SUBBAND_SAMPLE_RATE)
    parser.add_argument("--center-frequency-hz", type=float, default=2440e6)
    parser.add_argument("--bandwidth-hz", type=float, default=80e6)
    parser.add_argument("--bootstrap-duration-s", type=float, default=2.0)
    parser.add_argument("--known-aa-min-count", type=int, default=DEFAULT_MIN_AA_COUNT)
    parser.add_argument("--known-aa-hamming-tolerance", type=int, default=DEFAULT_AA_TOLERANCE_BITS)
    parser.add_argument(
        "--stage2-known-aa-bit-tolerance",
        type=int,
        default=DEFAULT_STAGE2_AA_BIT_TOLERANCE,
        help=(
            "Synthetic Hamming neighbourhood added around canonical AAs for stage 2; "
            "observed aliases are always included. 0 keeps aliases only, 1 is recommended."
        ),
    )
    parser.add_argument(
        "--force-known-aa",
        action="append",
        default=[],
        help=(
            "Explicit connection AA to include in stage 2. Use only when an external "
            "controller ledger supplies a verified AA (for example a PIP fake AA)."
        ),
    )
    parser.add_argument(
        "--stage1-ble-score-threshold",
        type=float,
        default=DEFAULT_STAGE1_BLE_SCORE_THRESHOLD,
        help="Stage-1 physical/decode length mismatch threshold in bytes.",
    )
    parser.add_argument(
        "--stage2-ble-score-threshold",
        type=float,
        default=DEFAULT_STAGE2_BLE_SCORE_THRESHOLD,
        help=(
            "Stage-2 physical/decode length mismatch threshold in bytes; the default "
            "is relaxed for noise and covert tails."
        ),
    )
    parser.add_argument(
        "--exclude-access-address",
        action="append",
        default=list(DEFAULT_ADVERTISING_ACCESS_ADDRESSES),
        help="Access address excluded from bootstrap; may be specified more than once.",
    )
    parser.add_argument("--overlap-samples", type=int, default=DEFAULT_OVERLAP_SAMPLES)
    parser.add_argument("--chunk-samples", type=int, default=DEFAULT_CHUNK_SAMPLES)
    parser.add_argument("--cpp-parser-threads", type=int, default=4)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--frontend-backend",
        choices=("auto", "cuda", "cpu"),
        default=DEFAULT_FRONTEND_BACKEND,
        help="DSP frontend backend; auto uses CUDA when a device is available and otherwise CPU.",
    )
    parser.add_argument("--parser-cpus", default="")
    parser.add_argument(
        "--stage1-ble-parser-backend",
        choices=("python", "cpp"),
        default="cpp",
        help="Normal parser backend for the bootstrap window.",
    )
    parser.add_argument(
        "--stage2-ble-parser-backend",
        choices=("python", "cpp"),
        default="cpp",
        help="Stage-2 parser backend; C++ candidates are filtered by known AA in PhantomChannel.",
    )
    parser.add_argument(
        "--measure-physical-length",
        action="store_true",
        help="Measure stage-2 IQ burst duration and subtract the decoded BLE link-layer duration.",
    )
    parser.add_argument(
        "--physical-length-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Physical burst measurement backend. Auto uses the local CUDA batch path when available.",
    )
    parser.add_argument(
        "--length-cuda-batch-size",
        type=int,
        default=16,
        help="Number of equal-sized IQ burst windows processed per CUDA batch.",
    )
    parser.add_argument("--length-lowpass-hz", type=float, default=900_000.0)
    parser.add_argument("--length-expected-max-us", type=float, default=5_000.0)
    parser.add_argument("--length-pre-margin-us", type=float, default=20.0)
    parser.add_argument("--length-post-margin-us", type=float, default=200.0)
    parser.add_argument("--length-smooth-us", type=float, default=1.5)
    parser.add_argument("--length-threshold-sigma", type=float, default=8.0)
    parser.add_argument("--length-min-threshold-ratio", type=float, default=2.0)
    parser.add_argument("--phy-rate-sps", type=float, default=DEFAULT_PHY_RATE)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    args.iq_path = args.iq_path.expanduser().resolve()
    args.metadata_path = args.metadata_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.parser_python = args.parser_python.expanduser()
    args.parser_entrypoint = args.parser_entrypoint.expanduser().resolve()
    args.ble_root = args.ble_root.expanduser().resolve()
    args.native_backend_dir = args.native_backend_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if not args.iq_path.is_file():
        raise FileNotFoundError(args.iq_path)
    if not args.metadata_path.is_file():
        raise FileNotFoundError(args.metadata_path)
    if args.sample_rate_sps <= 0 or args.subband_sample_rate_sps <= 0:
        raise ValueError("sample rates must be positive")
    if args.bootstrap_duration_s <= 0:
        raise ValueError("--bootstrap-duration-s must be positive")
    if args.known_aa_min_count < 1:
        raise ValueError("--known-aa-min-count must be positive")
    if not 0 <= args.known_aa_hamming_tolerance <= 32:
        raise ValueError("--known-aa-hamming-tolerance must be in [0, 32]")
    if not 0 <= args.stage2_known_aa_bit_tolerance <= 2:
        raise ValueError("--stage2-known-aa-bit-tolerance must be in [0, 2]")
    if args.stage1_ble_score_threshold <= 0 or args.stage2_ble_score_threshold <= 0:
        raise ValueError("BLE score thresholds must be positive")
    if args.overlap_samples < 0 or args.chunk_samples < 1:
        raise ValueError("overlap/chunk samples are invalid")
    if args.length_cuda_batch_size < 1:
        raise ValueError("--length-cuda-batch-size must be positive")

    args._stage1_ble_parser_backend_requested = args.stage1_ble_parser_backend
    args._stage2_ble_parser_backend_requested = args.stage2_ble_parser_backend
    frontend_backend, frontend_diagnostic = resolve_frontend_backend(args)
    args._frontend_backend_used = frontend_backend
    args._frontend_backend_diagnostic = frontend_diagnostic
    if frontend_backend == "cpu":
        # The external native parser intentionally requires CUDA threshold
        # segments.  Keep the fallback explicit and local to PhantomChannel;
        # the BLE_encrypt_check project remains untouched.
        args.stage1_ble_parser_backend = "python"
        args.stage2_ble_parser_backend = "python"
    if args.stage2_ble_parser_backend == "cpp" and not any(
        args.native_backend_dir.glob("bt_native*.so")
    ):
        raise FileNotFoundError(
            "PhantomChannel-local C++ backend is not built; run "
            "tools/build_phantom_native_backend.py first"
        )

    metadata = read_json(args.metadata_path)
    file_samples = args.iq_path.stat().st_size // SC16_BYTES_PER_SAMPLE
    metadata_samples = parse_int(metadata.get("samples"))
    total_samples = min(file_samples, metadata_samples) if metadata_samples else file_samples
    boundary_sample = min(total_samples, int(round(args.bootstrap_duration_s * args.sample_rate_sps)))
    if boundary_sample <= 0 or boundary_sample >= total_samples:
        raise ValueError("bootstrap boundary must leave a non-empty stage-2 window")

    stage1_dir = args.output_dir / "stages" / "stage1_blind"
    stage2_dir = args.output_dir / "stages" / "stage2_known_aa"
    stage1_status = run_parser_window(
        args,
        args.iq_path,
        stage1_dir,
        0,
        boundary_sample,
        [],
        args.stage1_ble_parser_backend,
        score_threshold=args.stage1_ble_score_threshold,
    )
    if not stage1_status["valid"]:
        write_json(args.output_dir / "validation_summary.json", {"valid": False, "stage": "stage1", "status": stage1_status})
        return 2

    stage1_rows, stage1_fields = read_csv(stage1_dir / "ble_packets.csv")
    decim = args.sample_rate_sps / args.subband_sample_rate_sps
    known_clusters, cluster_stats = tolerant_access_address_clusters(
        stage1_rows,
        tolerance_bits=args.known_aa_hamming_tolerance,
        min_count=args.known_aa_min_count,
        excluded_addresses=args.exclude_access_address,
    )
    known_addresses, known_address_stats = expand_stage2_known_access_addresses(
        known_clusters,
        bit_tolerance=args.stage2_known_aa_bit_tolerance,
        excluded_addresses=args.exclude_access_address,
    )
    forced_addresses = []
    for value in args.force_known_aa:
        normalized = normalize_hex_id(value)
        if not normalized:
            raise ValueError(f"invalid --force-known-aa value: {value!r}")
        canonical = f"0x{normalized}"
        forced_addresses.append(canonical)
        if canonical not in known_addresses:
            known_addresses.append(canonical)
        if not any(item.get("canonical_access_address_hex") == normalized for item in known_clusters):
            known_clusters.append(
                {
                    "cluster_id": f"forced:aa:0x{normalized}",
                    "canonical_access_address": canonical,
                    "canonical_access_address_hex": normalized,
                    "observation_count": 0,
                    "exact_value_count": 0,
                    "aliases": [],
                    "hamming_tolerance_bits": 0,
                    "first_sample": None,
                    "last_sample": None,
                    "first_timestamp_s": None,
                    "last_timestamp_s": None,
                    "channels": [],
                    "mean_confidence_score": None,
                }
            )
    known_address_stats["forced_known_address_count"] = len(forced_addresses)
    write_json(
        args.output_dir / "known_access_addresses.json",
        {
            "schema_version": 1,
            "parse_mode": "blind_stage1_bootstrap",
            "advertising_access_addresses_excluded": [f"0x{normalize_hex_id(value)}" for value in args.exclude_access_address],
            "hamming_tolerance_bits": args.known_aa_hamming_tolerance,
            "minimum_observations": args.known_aa_min_count,
            "selected_address_count": len(known_clusters),
            "selected_addresses": known_clusters,
            "selection_is_unbounded": True,
            "stage2_known_addresses": known_addresses,
            "stage2_known_address_policy": {
                "observed_aliases_are_included": True,
                "synthetic_hamming_bit_tolerance": args.stage2_known_aa_bit_tolerance,
                "synthetic_addresses_excluded": [
                    f"0x{normalize_hex_id(value)}" for value in args.exclude_access_address
                ],
                "forced_known_addresses": forced_addresses,
            },
            **known_address_stats,
        },
    )
    if not known_addresses:
        summary = {
            "valid": False,
            "stage": "bootstrap_selection",
            "reason": "no qualifying connection access-address cluster",
            "cluster_stats": cluster_stats,
        }
        write_json(args.output_dir / "validation_summary.json", summary)
        return 2

    stage2_start = max(0, boundary_sample - args.overlap_samples)
    stage2_status = run_parser_window(
        args,
        args.iq_path,
        stage2_dir,
        stage2_start,
        total_samples,
        known_addresses,
        args.stage2_ble_parser_backend,
        score_threshold=args.stage2_ble_score_threshold,
    )
    if not stage2_status["valid"]:
        write_json(args.output_dir / "validation_summary.json", {"valid": False, "stage": "stage2", "status": stage2_status})
        return 2

    stage2_raw_rows, stage2_fields = read_csv(stage2_dir / "ble_packets.csv")
    stage1_augmented = [add_stage1_fields(row) for row in stage1_rows]
    stage2_augmented = []
    for row in stage2_raw_rows:
        remapped = remap_stage2_row(
            row,
            source_start_sample=stage2_start,
            boundary_sample=boundary_sample,
            total_samples=total_samples,
            decim=decim,
            sample_rate_sps=args.sample_rate_sps,
            subband_sample_rate_sps=args.subband_sample_rate_sps,
            phy_rate_sps=args.phy_rate_sps,
            known_clusters=known_clusters,
            tolerance_bits=args.known_aa_hamming_tolerance,
        )
        if remapped is not None:
            stage2_augmented.append(remapped)

    physical_length_backend_used = "disabled"
    if args.measure_physical_length:
        physical_length_backend_used = measure_physical_lengths(stage2_augmented, args)
    args._physical_length_backend_used = physical_length_backend_used

    merged_rows = deduplicate_rows(stage1_augmented + stage2_augmented)
    merged_fields = list(stage1_fields)
    for field in stage2_fields:
        if field not in merged_fields:
            merged_fields.append(field)
    write_csv(args.output_dir / "ble_packets.csv", merged_rows, merged_fields)

    event_rows, event_fields = merge_event_rows(
        stage1_dir,
        stage2_dir,
        stage2_source_start_sample=stage2_start,
        boundary_sample=boundary_sample,
        total_samples=total_samples,
        decim=decim,
        args=args,
        known_clusters=known_clusters,
    )
    write_csv(args.output_dir / "packet_events.csv", event_rows, event_fields)
    for filename in ("btclassic_packets.csv", "target_selection.csv"):
        source = stage2_dir / filename
        if source.is_file():
            destination = args.output_dir / filename
            destination.write_bytes(source.read_bytes())

    summary = build_summary(
        args=args,
        metadata=metadata,
        boundary_sample=boundary_sample,
        total_samples=total_samples,
        stage1_rows=stage1_augmented,
        stage2_raw_rows=stage2_raw_rows,
        merged_rows=merged_rows,
        known_clusters=known_clusters,
        known_addresses=known_addresses,
        known_address_stats=known_address_stats,
        cluster_stats=cluster_stats,
        stage1_status=stage1_status,
        stage2_status=stage2_status,
    )
    summary["valid"] = True
    summary["output_dir"] = str(args.output_dir)
    write_json(args.output_dir / "validation_summary.json", summary)
    write_json(
        args.output_dir / "parse_manifest.json",
        {
            "schema_version": 1,
            "command": [str(path) for path in sys.argv],
            "ble_encrypt_check_modified": False,
            "stage1_output": str(stage1_dir),
            "stage2_output": str(stage2_dir),
            "merged_ble_packets": str(args.output_dir / "ble_packets.csv"),
            "known_access_addresses": str(args.output_dir / "known_access_addresses.json"),
            "boundary_sample": boundary_sample,
            "stage2_source_start_sample": stage2_start,
            "stage2_source_end_sample": total_samples,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except Exception as exc:
        print(f"two_stage_known_aa_parse.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
