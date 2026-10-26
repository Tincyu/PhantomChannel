#!/usr/bin/env python3
"""Stream one BLE channel from UHD into the realtime parser and emit PC frames."""

from __future__ import annotations

import argparse
import csv
import json
import os
import stat
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path


def frame_candidates(text: str):
    try:
        data = bytes.fromhex(text or "")
    except ValueError:
        return
    for start in range(max(0, len(data) - 5)):
        if data[start : start + 2] != b"PC" or start + 6 > len(data):
            continue
        length = data[start + 4]
        end = start + 6 + length
        if end > len(data):
            continue
        frame = data[start:end]
        check = 0
        for value in frame[:-1]:
            check ^= value
        if check == frame[-1]:
            yield frame


class LiveMonitor(threading.Thread):
    def __init__(self, csv_path: Path, hits_path: Path, stop: threading.Event):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.hits_path = hits_path
        self.stop = stop
        self.counts: Counter[tuple[str, str]] = Counter()
        self.records: dict[tuple[str, str], dict[str, str]] = {}
        self.rows_seen = 0
        self.pc_marker_rows = 0
        self.valid_frame_rows = 0

    def _process(self, row: dict[str, str]) -> None:
        self.rows_seen += 1
        tail = row.get("post_crc_hex", "")
        if "PC" in tail.upper():
            self.pc_marker_rows += 1
        for frame in frame_candidates(tail):
            self.valid_frame_rows += 1
            payload = frame[5:-1]
            aa = row.get("access_address", "")
            key = (aa, payload.hex())
            self.counts[key] += 1
            # Require two identical valid frames before exposing a live
            # result.  A single noisy PC-like candidate often carries a
            # nearby-bit AA; repeated exact frames suppress that output.
            if key in self.records or self.counts[key] < 2:
                continue
            try:
                decoded = bytes.fromhex(row.get("dewhitened_pdu_hex", ""))
                ll_length = int(row.get("payload_len", "0"))
                normal = decoded[2 : 2 + ll_length][-2:].hex()
            except (ValueError, TypeError):
                normal = ""
            try:
                display = payload.decode("utf-8")
                if not display.isprintable():
                    display = payload.hex()
            except UnicodeDecodeError:
                display = payload.hex()
            record = {
                "access_address": aa,
                "normal_payload_hex": normal,
                "covert_hex": payload.hex(),
                "covert_display": display,
                "sample_index": row.get("sample_index", ""),
                "channel": row.get("channel", ""),
            }
            self.records[key] = record
            with self.hits_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"LIVE covert data: {display} "
                f"(AA {aa}, channel {record['channel']}, sample {record['sample_index']})",
                flush=True,
            )

    def run(self) -> None:
        offset = 0
        fields = None
        while not self.stop.is_set() or not self.csv_path.is_file():
            if self.csv_path.is_file():
                with self.csv_path.open(newline="", encoding="utf-8") as handle:
                    handle.seek(offset)
                    while True:
                        line = handle.readline()
                        if not line:
                            break
                        offset = handle.tell()
                        values = next(csv.reader([line.rstrip("\r\n")]))
                        if fields is None:
                            fields = values
                            continue
                        if len(values) == len(fields):
                            self._process(dict(zip(fields, values)))
            if self.stop.is_set():
                break
            time.sleep(0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receiver-root", type=Path, required=True)
    parser.add_argument("--capture-bin", type=Path, required=True)
    parser.add_argument("--parser-python", type=Path, required=True)
    parser.add_argument("--parser-entrypoint", type=Path, required=True)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="capture duration in seconds; 0 (default) runs until Ctrl+C",
    )
    parser.add_argument("--usrp-args", required=True)
    parser.add_argument("--center-frequency", type=float, required=True)
    parser.add_argument("--sample-rate", type=float, required=True)
    parser.add_argument("--bandwidth", type=float, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--antenna", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parser_dir = output_dir / "parser"
    parser_dir.mkdir(parents=True, exist_ok=True)
    fifo = output_dir / "capture.sc16.pipe"
    if fifo.exists():
        if not stat.S_ISFIFO(fifo.stat().st_mode):
            raise SystemExit(f"error: live path exists and is not a FIFO: {fifo}")
        fifo.unlink()
    os.mkfifo(fifo)
    metadata = output_dir / "capture-metadata.json"
    capture_log = output_dir / "capture.log"
    parser_log = output_dir / "parser.log"
    hits_path = output_dir / "live_hits.jsonl"
    # A repeated live demo reuses its result directory; do not append stale
    # frames from the previous take.
    hits_path.write_text("", encoding="utf-8")
    live_pythonpath = [
        str(args.receiver_root / "artifacts/native/phantom_bt_native"),
        str(args.receiver_root / "vendor/BLE_encrypt_check/experiment"),
        str(args.receiver_root / "vendor/BLE_encrypt_check/ble_fun_test"),
        str(args.receiver_root / "vendor/BLE_encrypt_check/build-native"),
    ]
    if os.environ.get("PYTHONPATH"):
        live_pythonpath.append(os.environ["PYTHONPATH"])
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(live_pythonpath)
    parser_cmd = [
        str(args.parser_python),
        str(args.parser_entrypoint),
        "--source",
        "stdin",
        "--timestamp-mode",
        "sample_index",
        "--sample-rate",
        str(args.sample_rate),
        "--center-freq",
        str(args.center_frequency),
        "--bandwidth",
        str(args.bandwidth),
        "--subband-sample-rate",
        "4000000",
        "--iq-format",
        "int16",
        "--output-dir",
        str(parser_dir),
        "--chunk-samples",
        "4000000",
        "--overlap-samples",
        "200000",
        "--ble-candidate-detector",
        "fixed",
        "--ble-threshold",
        "0.01",
        "--ble-score-threshold",
        "3.0",
        "--ble-lpf-cutoff",
        "1000000",
        "--pfb-cutoff",
        "1900000",
        "--cleanup-numtaps",
        "31",
        "--cuda-device",
        "0",
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
        "cpp",
        "--bredr-parser-backend",
        "hybrid",
        "--cpp-parser-threads",
        "4",
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
        "--use-cuda",
    ]
    print(
        f"Streaming connected BLE: {args.center_frequency:.0f} Hz, "
        f"{args.sample_rate:g} S/s, {args.bandwidth:g} Hz",
        flush=True,
    )
    continuous = args.duration <= 0
    # The UHD helper is intentionally finite-duration.  In the default live
    # mode we run short capture segments back-to-back while keeping the FIFO
    # and parser open, so the user can stop the demo cleanly with Ctrl+C.
    segment_duration = args.duration if args.duration > 0 else 5.0
    capture_cmd_prefix = [
        str(args.capture_bin),
        "--args",
        args.usrp_args,
        "--rate",
        str(args.sample_rate),
        "--bandwidth",
        str(args.bandwidth),
        "--frequency",
        str(args.center_frequency),
        "--gain",
        str(args.gain),
        "--antenna",
        args.antenna,
        "--block-samples",
        "32768",
        "--output",
        str(fifo),
        "--metadata",
        str(metadata),
    ]

    def make_capture_command() -> list[str]:
        return [
            *capture_cmd_prefix,
            "--duration",
            str(segment_duration),
        ]

    # Open the FIFO read end after the capture process has opened its write
    # end.  A small parent-held write descriptor prevents EOF between finite
    # UHD segments in continuous mode; it is closed only during final cleanup.
    capture = None
    fifo_handle = None
    keepalive_fd = None
    parser = None
    monitor = None
    capture_status = None
    parser_status = None
    interrupted = False
    segments = 0
    with parser_log.open("w", encoding="utf-8") as parser_handle, capture_log.open(
        "w", encoding="utf-8"
    ) as capture_handle:
        stop = threading.Event()
        try:
            capture_handle.write("--- capture segment 1 ---\n")
            capture_handle.flush()
            capture = subprocess.Popen(
                make_capture_command(),
                stdout=capture_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            segments = 1
            fifo_handle = open(fifo, "rb", buffering=0)
            # Keep one writer open across segment boundaries.  Without this,
            # the parser would observe EOF as soon as a finite capture exits.
            keepalive_fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            parser = subprocess.Popen(
                parser_cmd,
                stdin=fifo_handle,
                stdout=parser_handle,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=args.receiver_root / "vendor/BLE_encrypt_check",
                start_new_session=True,
            )
            monitor = LiveMonitor(parser_dir / "ble_packets.csv", hits_path, stop)
            monitor.start()

            while True:
                capture_status = capture.wait()
                capture = None
                if not continuous:
                    break
                if capture_status not in (0, 2):
                    break
                segments += 1
                capture_handle.write(f"--- capture segment {segments} ---\n")
                capture_handle.flush()
                capture = subprocess.Popen(
                    make_capture_command(),
                    stdout=capture_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except KeyboardInterrupt:
            interrupted = True
            print("Stopping live capture (Ctrl+C)...", flush=True)
        finally:
            if capture is not None and capture.poll() is None:
                capture.terminate()
                try:
                    capture.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    capture.kill()
                    capture.wait()
                capture_status = capture.returncode
            if keepalive_fd is not None:
                os.close(keepalive_fd)
                keepalive_fd = None
            if fifo_handle is not None:
                fifo_handle.close()
                fifo_handle = None
            if parser is not None:
                parser_status = parser.wait()
            stop.set()
            if monitor is not None:
                monitor.join(timeout=5.0)

    if monitor is None:
        raise SystemExit("error: live parser was not started")
    stats = {
        "decoded_rows": monitor.rows_seen,
        "pc_marker_rows": monitor.pc_marker_rows,
        "valid_frame_rows": monitor.valid_frame_rows,
        "unique_covert_frames": len(monitor.records),
        "capture_returncode": capture_status,
        "parser_returncode": parser_status,
        "continuous": continuous,
        "segments": segments,
        "interrupted": interrupted,
    }
    (output_dir / "live_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    if not interrupted and capture_status not in (0, 2):
        print(f"error: USRP capture failed; see {capture_log}", file=sys.stderr)
        return capture_status
    if not interrupted and parser_status != 0:
        print(f"error: live parser failed; see {parser_log}", file=sys.stderr)
        return parser_status
    if not monitor.records:
        print(
            f"error: live capture ended without a checksum-valid covert frame; "
            f"decoded_rows={monitor.rows_seen}, pc_marker_rows={monitor.pc_marker_rows}",
            file=sys.stderr,
        )
        return 1
    key, count = monitor.counts.most_common(1)[0]
    record = monitor.records[key]
    result = {
        "status": "passed",
        "mode": "live_stream",
        "access_address": record["access_address"],
        "normal_payload_hex": record["normal_payload_hex"],
        "covert_hex": record["covert_hex"],
        "covert_display": record["covert_display"],
        "valid_frame_count": count,
    }
    (output_dir / "demo_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Connection access address: {record['access_address']}")
    print(f"Normal payload: {record['normal_payload_hex']}")
    print(f"Covert data: {record['covert_display']}")
    if record["covert_display"] != record["covert_hex"]:
        print(f"Covert data hex: {record['covert_hex']}")
    print(f"Valid frames: {count}")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
