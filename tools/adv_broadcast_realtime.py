#!/usr/bin/env python3
"""Run and record a BLE advertising-channel-39 PhantomChannel experiment.

This is the small, platform-neutral host path for the first handoff goal.  It
does not build or flash firmware.  The transmitter is configured at runtime
over UART, while the receiver is a live B210 stream (or an equivalent UDP IQ
source) and emits one JSON object per decoded candidate.

The tool deliberately keeps partial candidates, unknown bytes, BER and UHD
quality counters in the result.  A noisy or overflowing capture is evidence
about capture quality, not a reason to silently discard a useful recovery.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - reported by the CLI
    raise SystemExit("numpy is required for live IQ parsing") from exc


BLE_ADV_CHANNEL = 39
BLE_ADV_FREQUENCY_HZ = 2_480_000_000
BLE_ADV_BANDWIDTH_HZ = 4_000_000
BLE_SAMPLE_RATE_SPS = 4_000_000
MAX_COVERT_LEN = 239
ADV_ACCESS_ADDRESS = "d6be898e"
PC_MAGIC = b"PC"
ACK_JSON_RE = re.compile(r"PHANTOM_COVERT_ACK\s+(\{.*\})")
CC_ACK_RE = re.compile(r"COVERT OK len=(\d+)")

# Live B210 captures of the CC2340R5 have a lower amplitude than the Nordic
# reference captures. These defaults keep the weak scan-response tail while
# retaining BLE access-address validation and removing the SDK-appended CRC.
DEFAULT_THRESHOLD = 0.01
DEFAULT_MIN_SEGMENT_LEN = 100
DEFAULT_MAX_SEGMENT_LEN = 40000
DEFAULT_AA_TOLERANCE = 2
DEFAULT_STRIP_HARDWARE_CRC = True


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_payload(*, covert_hex: str | None, covert_data: str | None) -> bytes:
    if covert_hex is not None and covert_data is not None:
        raise ValueError("use only one of --covert-hex and --covert-data")
    if covert_hex is not None:
        try:
            payload = bytes.fromhex(covert_hex)
        except ValueError as exc:
            raise ValueError(f"invalid --covert-hex: {exc}") from exc
    elif covert_data is not None:
        payload = covert_data.encode("utf-8")
    else:
        raise ValueError("one of --covert-hex or --covert-data is required")
    if not 1 <= len(payload) <= MAX_COVERT_LEN:
        raise ValueError(f"covert data length must be in 1..{MAX_COVERT_LEN} bytes")
    return payload


def compare_payload(expected: bytes, observed: bytes) -> dict[str, Any]:
    overlap = min(len(expected), len(observed))
    matched = sum(left == right for left, right in zip(expected, observed))
    bit_errors = sum((left ^ right).bit_count() for left, right in zip(expected, observed))
    unknown_bytes = max(0, len(expected) - overlap)
    overflow_bytes = max(0, len(observed) - len(expected))
    if overlap:
        ber = bit_errors / float(overlap * 8)
    else:
        ber = None
    if observed == expected:
        status = "exact"
    elif overlap and (matched or bit_errors):
        status = "partial"
    else:
        status = "unmatched"
    return {
        "status": status,
        "expected_len": len(expected),
        "observed_len": len(observed),
        "matched_bytes": matched,
        "byte_hit_rate": matched / len(expected) if expected else 0.0,
        "bit_errors": bit_errors,
        "ber": ber,
        "unknown_bytes": unknown_bytes,
        "overflow_bytes": overflow_bytes,
        "unknown_range": [overlap, len(expected)] if unknown_bytes else None,
    }


def unwrap_pc_frame(data: bytes) -> bytes:
    """Remove the optional project PC envelope when its XOR check is valid."""
    payload = extract_pc_frame(data)
    return payload if payload is not None else data


def extract_pc_frame(data: bytes) -> bytes | None:
    """Return a validated PhantomChannel payload, or ``None`` for other tails."""
    if len(data) < 6 or data[:2] != PC_MAGIC:
        return None
    payload_len = data[4]
    frame_len = 6 + payload_len
    if frame_len > len(data):
        return None
    frame = data[:frame_len]
    check = 0
    for byte in frame[:-1]:
        check ^= byte
    return frame[5 : 5 + payload_len] if check == frame[-1] else None


def _swap_bits(value: int) -> int:
    return (value * 0x0202020202 & 0x010884422010) % 1023


def dewhitening(data: Iterable[int], channel: int) -> list[int]:
    result: list[int] = []
    lfsr = _swap_bits(channel) | 2
    for byte in data:
        value = _swap_bits(byte)
        for bit in (128, 64, 32, 16, 8, 4, 2, 1):
            if lfsr & 0x80:
                lfsr ^= 0x11
                value ^= bit
            lfsr <<= 1
        result.append(_swap_bits(value))
    return result


def _bit_reverse8(value: int) -> int:
    value = ((value & 0xF0) >> 4) | ((value & 0x0F) << 4)
    value = ((value & 0xCC) >> 2) | ((value & 0x33) << 2)
    return ((value & 0xAA) >> 1) | ((value & 0x55) << 1)


def ble_crc_bytes(data: Iterable[int], crc_init: int = 0x555555) -> bytes:
    crc = crc_init & 0xFFFFFF
    for byte in data:
        crc ^= _bit_reverse8(byte) << 16
        for _ in range(8):
            crc = ((crc << 1) ^ 0x65B) if crc & 0x800000 else crc << 1
            crc &= 0xFFFFFF
    return bytes(
        (
            _bit_reverse8((crc >> 16) & 0xFF),
            _bit_reverse8((crc >> 8) & 0xFF),
            _bit_reverse8(crc & 0xFF),
        )
    )


def _bits_to_hex(bits: Iterable[int]) -> str:
    bits = list(bits)
    out = []
    for start in range(0, len(bits), 4):
        nibble = bits[start : start + 4]
        if len(nibble) < 4:
            nibble += [0] * (4 - len(nibble))
        nibble.reverse()
        out.append(format(int("".join(str(bit) for bit in nibble), 2), "x"))
    return "".join(out)


def _reverse_bytes(hex_string: str) -> list[int]:
    return [int(hex_string[start : start + 2][::-1], 16) for start in range(0, len(hex_string), 2)]


def _hamming_hex(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 999


def _find_preamble(hex_buffer: str, mode: str) -> tuple[int, str]:
    patterns = ("aaaa", "5555") if mode == "2M" else ("aa", "55")
    for pattern in patterns:
        position = hex_buffer.find(pattern)
        if position >= 0:
            return position, mode
    for pattern in patterns:
        for position in range(max(0, len(hex_buffer) - len(pattern) + 1)):
            if _hamming_hex(hex_buffer[position : position + len(pattern)], pattern) <= 1:
                return position, mode
    return -1, "0M"


def _strip_hardware_crc(decoded: list[int], packet_min_len: int) -> tuple[bytes, bool]:
    for trailing_len in range(5):
        crc_end = len(decoded) - trailing_len
        crc_start = crc_end - 3
        if crc_start < packet_min_len:
            continue
        if bytes(decoded[crc_start:crc_end]) == ble_crc_bytes(decoded[:crc_start]):
            return bytes(decoded[packet_min_len:crc_start]), True
    return bytes(decoded[packet_min_len:]), False


def _decode_packet(bits: np.ndarray, mode: str, iq_len: int, sample_rate: float, channel: int,
                   *, aa_tolerance: int, drop_last_extra: bool, strip_hw_crc: bool) -> list[dict[str, Any]]:
    sps = int(sample_rate / (2e6 if mode == "2M" else 1e6))
    if sps < 1:
        return []
    candidates: list[dict[str, Any]] = []
    for bit_offset in range(4):
        buffer = _bits_to_hex(bits[bit_offset:])
        pre_pos, found_mode = _find_preamble(buffer, mode)
        pre_len = 2 if found_mode == "2M" else 1 if found_mode == "1M" else 0
        if pre_pos < 0 or pre_pos + pre_len + 18 > len(buffer):
            continue
        ble_packet = _reverse_bytes(buffer[pre_pos:])
        access_address = "".join(f"{value:02x}" for value in ble_packet[pre_len : pre_len + 4])
        if _hamming_hex(access_address, ADV_ACCESS_ADDRESS) > aa_tolerance:
            continue
        decoded = dewhitening(ble_packet[pre_len + 4 :], channel)
        if len(decoded) < 2:
            continue
        payload_len = decoded[1]
        packet_min_len = 2 + payload_len + 3
        if len(decoded) < packet_min_len:
            continue
        payload = bytes(decoded[2 : 2 + payload_len])
        tail, hardware_crc_stripped = (
            _strip_hardware_crc(decoded, packet_min_len)
            if strip_hw_crc
            else (bytes(decoded[packet_min_len:]), False)
        )
        if drop_last_extra and tail and not hardware_crc_stripped:
            tail = tail[:-1]
        pc_payload = extract_pc_frame(tail)
        covert = pc_payload if pc_payload is not None else tail
        expected_physical_len = (iq_len - pre_pos * sps * 4 - 8 * sps * 10) / float(8 * sps)
        candidates.append(
            {
                "mode": found_mode,
                "access_address": access_address,
                "advertiser_address": (
                    ":".join(f"{value:02X}" for value in reversed(payload[:6]))
                    if len(payload) >= 6
                    else ""
                ),
                "payload_len": payload_len,
                "normal_payload_hex": payload.hex(),
                "covert_hex": covert.hex(),
                "covert_len": len(covert),
                "pc_frame_valid": pc_payload is not None,
                "physical_length_score": abs(expected_physical_len - payload_len),
                "hardware_crc_stripped": hardware_crc_stripped,
                "bit_offset": bit_offset,
            }
        )
    candidates.sort(key=lambda item: item["physical_length_score"])
    return candidates


def decode_iq(iq: np.ndarray, *, sample_rate: float, channel: int, freq_dev: float,
              aa_tolerance: int, drop_last_extra: bool, strip_hw_crc: bool) -> list[dict[str, Any]]:
    if iq.size < 32:
        return []
    phase = np.angle(iq)
    freq_dev_samples = sample_rate / (2.0 * np.pi * freq_dev) * np.diff(np.unwrap(phase))
    candidates: list[dict[str, Any]] = []
    for mode in ("2M", "1M"):
        sps = int(sample_rate / (2e6 if mode == "2M" else 1e6))
        if sps < 1:
            continue
        indexes = np.arange(sps // 2, len(freq_dev_samples), sps)
        bits = (freq_dev_samples[indexes] > 0.10).astype(np.uint8)
        candidates.extend(
            _decode_packet(
                bits,
                mode,
                len(iq),
                sample_rate,
                channel,
                aa_tolerance=aa_tolerance,
                drop_last_extra=drop_last_extra,
                strip_hw_crc=strip_hw_crc,
            )
        )
    candidates.sort(key=lambda item: item["physical_length_score"])
    return candidates


class ThresholdSegmenter:
    def __init__(self, threshold: float, min_len: int, max_len: int, pre_samples: int):
        self.threshold = float(threshold)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.pre_samples = max(0, int(pre_samples))
        self.buffer = np.empty(0, dtype=np.complex64)
        self.buffer_start = 0
        self.prev_tail = np.empty(0, dtype=np.complex64)
        self.sample_cursor = 0

    def push(self, iq_samples: np.ndarray) -> list[tuple[int, np.ndarray]]:
        if self.buffer.size:
            full = np.concatenate((self.buffer, iq_samples))
            full_start = self.buffer_start
        else:
            full = np.concatenate((self.prev_tail, iq_samples))
            full_start = self.sample_cursor - self.prev_tail.size
        above = np.abs(full) > self.threshold
        segments: list[tuple[int, np.ndarray]] = []
        current_start: int | None = None
        for index, present in enumerate(above):
            if present:
                if current_start is None:
                    current_start = max(0, index - self.pre_samples)
                if index - current_start >= self.max_len:
                    segment = full[current_start:index]
                    if len(segment) >= self.min_len:
                        segments.append((full_start + current_start, segment.copy()))
                    current_start = None
            elif current_start is not None:
                segment = full[current_start:index]
                if len(segment) >= self.min_len:
                    segments.append((full_start + current_start, segment.copy()))
                current_start = None
        if current_start is not None:
            self.buffer = full[current_start:].copy()
            self.buffer_start = full_start + current_start
        else:
            self.buffer = np.empty(0, dtype=np.complex64)
            self.buffer_start = self.sample_cursor + len(iq_samples)
        if self.pre_samples:
            self.prev_tail = full[-self.pre_samples:].copy()
        else:
            self.prev_tail = np.empty(0, dtype=np.complex64)
        self.sample_cursor += len(iq_samples)
        return segments


@dataclass
class SourceStats:
    samples: int = 0
    chunks: int = 0
    overflows: int = 0
    timeouts: int = 0
    dropped_chunks: int = 0
    errors: list[str] = field(default_factory=list)


class IQSource:
    def start(self) -> None:
        raise NotImplementedError

    def read(self, timeout_s: float) -> tuple[int, np.ndarray] | None:
        raise NotImplementedError

    def stop(self) -> None:
        return None

    @property
    def config(self) -> dict[str, Any]:
        return {}

    @property
    def stats(self) -> SourceStats:
        raise NotImplementedError


class QueueSource(IQSource):
    def __init__(self, *, chunk_size: int = 4096, queue_depth: int = 16, save_path: Path | None = None):
        self.chunk_size = int(chunk_size)
        self.queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=max(1, queue_depth))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = SourceStats()
        self._config: dict[str, Any] = {}
        self._save_path = save_path
        self._save_handle = None

    @property
    def stats(self) -> SourceStats:
        return self._stats

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def read(self, timeout_s: float) -> tuple[int, np.ndarray] | None:
        try:
            item = self.queue.get(timeout=timeout_s)
        except queue.Empty:
            self._stats.timeouts += 1
            return None
        if item is None:
            return None
        return item

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._save_handle is not None:
            self._save_handle.close()
            self._save_handle = None

    def _put(self, offset: int, samples: np.ndarray) -> None:
        self._stats.samples += len(samples)
        self._stats.chunks += 1
        if self._save_handle is not None:
            self._save_handle.write(np.asarray(samples, dtype=np.complex64).tobytes())
        try:
            self.queue.put((offset, samples), timeout=0.2)
        except queue.Full:
            self._stats.dropped_chunks += 1

    def _finish(self) -> None:
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            pass


class UhdB210Source(QueueSource):
    def __init__(self, args: argparse.Namespace, save_path: Path | None):
        super().__init__(chunk_size=args.chunk_samples, queue_depth=args.queue_depth, save_path=save_path)
        self.args = args
        self._config = {
            "source": "uhd_b210",
            "device": "B210",
            "usrp_args": args.usrp_args,
            "channel": BLE_ADV_CHANNEL,
            "center_frequency_hz": BLE_ADV_FREQUENCY_HZ,
            "sample_rate_sps": args.sample_rate,
            "bandwidth_hz": BLE_ADV_BANDWIDTH_HZ,
            "gain_db": args.gain,
            "antenna": args.antenna,
            "iq_format": "complex64",
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="b210-rx", daemon=True)
        self._thread.start()

    def _run(self) -> None:  # pragma: no cover - requires UHD hardware
        try:
            import uhd

            if self._save_path is not None:
                self._save_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_handle = self._save_path.open("wb")
            usrp = uhd.usrp.MultiUSRP(self.args.usrp_args)
            usrp.set_rx_rate(self.args.sample_rate, 0)
            usrp.set_rx_freq(uhd.types.TuneRequest(BLE_ADV_FREQUENCY_HZ), 0)
            usrp.set_rx_gain(self.args.gain, 0)
            usrp.set_rx_bandwidth(BLE_ADV_BANDWIDTH_HZ, 0)
            try:
                usrp.set_rx_antenna(self.args.antenna, 0)
            except RuntimeError as exc:
                self._stats.errors.append(f"antenna: {exc}")
            self._config.update(
                {
                    "actual_center_frequency_hz": usrp.get_rx_freq(0),
                    "actual_sample_rate_sps": usrp.get_rx_rate(0),
                    "actual_bandwidth_hz": usrp.get_rx_bandwidth(0),
                    "actual_gain_db": usrp.get_rx_gain(0),
                }
            )
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [0]
            streamer = usrp.get_rx_stream(stream_args)
            metadata = uhd.types.RXMetadata()
            buffer = np.zeros((1, self.chunk_size), dtype=np.complex64)
            command = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            command.stream_now = True
            streamer.issue_stream_cmd(command)
            offset = 0
            while not self._stop.is_set():
                count = streamer.recv(buffer, metadata, 1.0)
                error_code = metadata.error_code
                if error_code != uhd.types.RXMetadataErrorCode.none:
                    description = metadata.strerror()
                    if "overflow" in description.lower() or "overrun" in description.lower():
                        self._stats.overflows += 1
                    elif "timeout" in description.lower():
                        self._stats.timeouts += 1
                    else:
                        self._stats.errors.append(description)
                    continue
                if count:
                    samples = np.asarray(buffer[0, :count], dtype=np.complex64).copy()
                    self._put(offset, samples)
                    offset += count
            streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
        except Exception as exc:
            self._stats.errors.append(f"UHD: {type(exc).__name__}: {exc}")
        finally:
            self._finish()


class UdpIQSource(QueueSource):
    def __init__(self, args: argparse.Namespace, save_path: Path | None):
        super().__init__(chunk_size=args.chunk_samples, queue_depth=args.queue_depth, save_path=save_path)
        self.args = args
        self._config = {
            "source": "udp_fc32",
            "bind": f"{args.udp_host}:{args.udp_port}",
            "channel": BLE_ADV_CHANNEL,
            "center_frequency_hz": BLE_ADV_FREQUENCY_HZ,
            "sample_rate_sps": args.sample_rate,
            "bandwidth_hz": BLE_ADV_BANDWIDTH_HZ,
            "iq_format": "complex64",
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="udp-rx", daemon=True)
        self._thread.start()

    def _run(self) -> None:  # pragma: no cover - socket loop is integration-only
        offset = 0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.25)
        sock.bind((self.args.udp_host, self.args.udp_port))
        try:
            while not self._stop.is_set():
                try:
                    payload, _address = sock.recvfrom(self.args.udp_max_bytes)
                except socket.timeout:
                    continue
                if len(payload) % np.dtype(np.complex64).itemsize:
                    self._stats.errors.append(f"malformed UDP chunk: {len(payload)} bytes")
                    continue
                samples = np.frombuffer(payload, dtype=np.complex64).copy()
                self._put(offset, samples)
                offset += len(samples)
        except Exception as exc:
            self._stats.errors.append(f"UDP: {type(exc).__name__}: {exc}")
        finally:
            sock.close()
            self._finish()


class ReplayIQSource(IQSource):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stats = SourceStats()
        self._config = {
            "source": "replay",
            "input_iq": str(Path(args.input_iq).resolve()),
            "iq_format": args.iq_format,
            "channel": BLE_ADV_CHANNEL,
            "center_frequency_hz": BLE_ADV_FREQUENCY_HZ,
            "sample_rate_sps": args.sample_rate,
            "bandwidth_hz": BLE_ADV_BANDWIDTH_HZ,
        }
        self._chunks: list[tuple[int, np.ndarray]] = []
        self._index = 0

    @property
    def stats(self) -> SourceStats:
        return self._stats

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def start(self) -> None:
        path = Path(self.args.input_iq)
        if path.suffix == ".npy":
            iq = np.asarray(np.load(path), dtype=np.complex64)
        else:
            dtype = np.int16 if self.args.iq_format == "sc16" else np.complex64
            raw = np.fromfile(path, dtype=dtype)
            if dtype == np.int16:
                raw = raw[: raw.size - raw.size % 2]
                iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
                iq = (iq / 32768.0).astype(np.complex64)
            else:
                iq = raw.astype(np.complex64)
        for start in range(0, len(iq), self.args.chunk_samples):
            self._chunks.append((start, iq[start : start + self.args.chunk_samples]))

    def read(self, timeout_s: float) -> tuple[int, np.ndarray] | None:
        del timeout_s
        if self._index >= len(self._chunks):
            return None
        item = self._chunks[self._index]
        self._index += 1
        self._stats.chunks += 1
        self._stats.samples += len(item[1])
        return item


def _command_for(args: argparse.Namespace, payload: bytes) -> list[str]:
    if args.uart_command:
        data = payload.decode("utf-8", errors="strict") if "{data}" in args.uart_command else ""
        return [args.uart_command.format(hex=payload.hex(), data=data)]
    if args.platform == "efr32mg24":
        return ["setAdvTailHex", payload.hex()]
    if args.platform == "cc2340r5":
        return ["covert-hex", payload.hex()]
    return ["phantom", "covert-hex", payload.hex()]


def send_uart(args: argparse.Namespace, payload: bytes, run_root: Path) -> dict[str, Any]:
    command_parts = _command_for(args, payload)
    artifact = {
        "status": "skipped" if not args.uart else "failed",
        "platform": args.platform,
        "port": args.uart,
        "expected_len": len(payload),
        "expected_hex": payload.hex(),
        "command": " ".join(command_parts),
        "ack": None,
        "timestamp": now_iso(),
    }
    if not args.uart:
        json_dump(run_root / "uart_ack.json", artifact)
        return artifact
    transcript: list[str] = []
    try:
        import serial

        with serial.Serial(args.uart, baudrate=115200, timeout=0.2, write_timeout=3,
                           rtscts=False, dsrdtr=False) as device:
            device.reset_input_buffer()
            # Mynewt-based boards can still be printing their shell banner when
            # the B210 has finished starting.  Drain a short boot window before
            # issuing the runtime command; otherwise the first command can be
            # lost even though the UART and firmware are healthy.
            boot_deadline = time.monotonic() + 2.0
            while time.monotonic() < boot_deadline:
                line = device.readline().decode("utf-8", errors="replace")
                if line:
                    transcript.append(line)
            if args.platform == "efr32mg24" and not args.uart_command:
                commands = []
                device.write(b"status\r\n")
                device.flush()
                preflight = bytearray()
                deadline = time.monotonic() + args.ack_timeout_s
                while time.monotonic() < deadline:
                    line = device.readline()
                    if line:
                        preflight.extend(line)
                        transcript.append(line.decode("utf-8", errors="replace"))
                if b"{AppMode:ContinuousTx}" in preflight:
                    commands.append("tx 0\r\n")
                commands.extend(
                    [
                        "setBleAdvertising 39\r\n",
                        "setAdvName E\r\n",
                        f"setAdvTailHex {payload.hex()}\r\n",
                        "tx 0\r\n",
                        "status\r\n",
                    ]
                )
            else:
                commands = [" ".join(command_parts) + "\n"]
            artifact["command"] = [command.rstrip("\r\n") for command in commands]
            for command in commands:
                transcript.append(f"TX {command!r}\n")
                device.write(command.encode("ascii"))
                device.flush()
                deadline = time.monotonic() + args.ack_timeout_s
                while time.monotonic() < deadline:
                    line = device.readline().decode("utf-8", errors="replace")
                    if not line:
                        continue
                    transcript.append(line)
                    if args.platform != "efr32mg24" and (
                        ACK_JSON_RE.search(line) or CC_ACK_RE.search(line)
                    ):
                        break
            output = "".join(transcript)
            match = ACK_JSON_RE.search(output)
            if match:
                try:
                    artifact["ack"] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            cc_match = CC_ACK_RE.search(output)
            if cc_match:
                artifact["ack"] = {"status": 0, "len": int(cc_match.group(1))}
            if args.platform == "efr32mg24":
                covert_matches = re.findall(r"CovertLen:(\d+)", output)
                tx_matches = re.findall(r"TxLength:(\d+)", output)
                covert_match = covert_matches[-1] if covert_matches else None
                tx_match = tx_matches[-1] if tx_matches else None
                artifact["ack"] = {
                    "status": 0 if covert_match and int(covert_match) == len(payload) else 1,
                    "len": int(covert_match) if covert_match else None,
                    "tx_length": int(tx_match) if tx_match else None,
                    "continuous_tx": "{ContinuousTx:Enabled}" in output,
                }
            ack = artifact["ack"] or {}
            artifact["status"] = "passed" if (
                ack.get("status") == 0
                and int(ack.get("len", -1)) == len(payload)
                and (not ack.get("hex") or ack.get("hex", "").lower() == payload.hex())
            ) else "failed"
    except Exception as exc:
        transcript.append(f"SERIAL_ERROR: {type(exc).__name__}: {exc}\n")
        artifact["error"] = str(exc)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs/uart_transcript.log").write_text("".join(transcript), encoding="utf-8", errors="replace")
    artifact["log"] = str(run_root / "logs/uart_transcript.log")
    json_dump(run_root / "uart_ack.json", artifact)
    return artifact


def make_source(args: argparse.Namespace, run_root: Path) -> IQSource:
    capture_path = run_root / "capture.fc32" if args.save_iq else None
    if args.source == "uhd":
        return UhdB210Source(args, capture_path)
    if args.source == "udp":
        return UdpIQSource(args, capture_path)
    return ReplayIQSource(args)


def build_run_manifest(args: argparse.Namespace, payload: bytes, run_id: str, run_root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now_iso(),
        "platform": args.platform,
        "mode": "broadcast_only",
        "input": {"len_bytes": len(payload), "hex": payload.hex()},
        "usrp": {
            "device": "B210",
            "channel": BLE_ADV_CHANNEL,
            "center_frequency_hz": BLE_ADV_FREQUENCY_HZ,
            "sample_rate_sps": args.sample_rate,
            "bandwidth_hz": BLE_ADV_BANDWIDTH_HZ,
            "gain_db": args.gain,
            "antenna": args.antenna,
            "usrp_args": args.usrp_args,
            "source": args.source,
        },
        "parser": {
            "threshold": args.threshold,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "pre_samples": args.pre_samples,
            "freq_dev_hz": args.freq_dev,
            "aa_tolerance_bits": args.aa_tolerance,
            "drop_last_extra": args.drop_last_extra,
            "strip_hw_crc": args.strip_hw_crc,
        },
        "paths": {"run_root": str(run_root)},
        "scope_exclusions": [
            "connection_state",
            "PIP",
            "full_fragment_reassembly",
            "strict_TIFS",
        ],
    }


def run(args: argparse.Namespace) -> int:
    if args.observe:
        return run_observe(args)
    try:
        payload = parse_payload(covert_hex=args.covert_hex, covert_data=args.covert_data)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_id = args.run_id or f"adv-ch39-{args.platform}-{time.strftime('%Y%m%d-%H%M%S')}"
    root = Path(args.output_root).expanduser().resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    json_dump(root / "input.json", {"platform": args.platform, "len_bytes": len(payload), "hex": payload.hex()})
    json_dump(root / "run_manifest.json", build_run_manifest(args, payload, run_id, root))
    source = make_source(args, root)
    json_dump(root / "usrp_config.json", {**build_run_manifest(args, payload, run_id, root)["usrp"], **source.config})
    json_dump(root / "parser_config.json", build_run_manifest(args, payload, run_id, root)["parser"])
    if args.dry_run:
        json_dump(root / "uart_ack.json", {"status": "dry_run", "expected_len": len(payload), "expected_hex": payload.hex()})
        summary = {"run_id": run_id, "status": "dry_run", "platform": args.platform, "input_len": len(payload), "paths": {"run_root": str(root)}}
        json_dump(root / "result_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    source.start()
    time.sleep(max(0.0, args.uart_delay_s))
    json_dump(root / "usrp_config.json", source.config)
    uart = send_uart(args, payload, root) if not args.no_uart else {"status": "skipped", "reason": "--no-uart"}
    if args.no_uart:
        json_dump(root / "uart_ack.json", uart)
    segmenter = ThresholdSegmenter(args.threshold, args.min_len, args.max_len, args.pre_samples)
    parser_path = root / "realtime_parser.jsonl"
    candidate_count = 0
    segment_count = 0
    hit_count = 0
    partial_count = 0
    best: dict[str, Any] | None = None
    started = time.monotonic()
    with parser_path.open("w", encoding="utf-8") as output:
        while time.monotonic() - started < args.duration_s:
            item = source.read(min(0.5, max(0.01, args.duration_s - (time.monotonic() - started))))
            if item is None:
                if args.source == "replay":
                    break
                continue
            offset, samples = item
            for segment_offset, segment in segmenter.push(samples):
                segment_count += 1
                for decoded in decode_iq(
                    segment,
                    sample_rate=args.sample_rate,
                    channel=BLE_ADV_CHANNEL,
                    freq_dev=args.freq_dev,
                    aa_tolerance=args.aa_tolerance,
                    drop_last_extra=args.drop_last_extra,
                    strip_hw_crc=args.strip_hw_crc,
                ):
                    if args.address and decoded.get("advertiser_address", "").upper() != args.address.upper():
                        continue
                    candidate_count += 1
                    observed = bytes.fromhex(decoded["covert_hex"])
                    quality = compare_payload(payload, observed)
                    if quality["status"] == "exact":
                        hit_count += 1
                    elif quality["status"] == "partial":
                        partial_count += 1
                    event = {
                        "schema_version": 1,
                        "timestamp": now_iso(),
                        "run_id": run_id,
                        "platform": args.platform,
                        "channel": BLE_ADV_CHANNEL,
                        "frequency_hz": BLE_ADV_FREQUENCY_HZ,
                        "sample_offset": segment_offset,
                        "input_len": len(payload),
                        "input_hex": payload.hex(),
                        "recovered_len": len(observed),
                        "recovered_hex": observed.hex(),
                        "candidate_index": candidate_count,
                        "candidate_count": candidate_count,
                        "hit_count": hit_count,
                        "partial_hit_count": partial_count,
                        "quality": quality,
                        "mode": decoded["mode"],
                        "access_address": decoded["access_address"],
                        "advertiser_address": decoded["advertiser_address"],
                        "normal_payload_hex": decoded["normal_payload_hex"],
                        "hardware_crc_stripped": decoded["hardware_crc_stripped"],
                    }
                    output.write(json.dumps(event, sort_keys=True) + "\n")
                    output.flush()
                    if best is None or (quality["status"] == "exact" and best["quality"]["status"] != "exact"):
                        best = event
    source.stop()
    stats = source.stats
    capture_quality = {
        "samples": stats.samples,
        "chunks": stats.chunks,
        "overflows": stats.overflows,
        "timeouts": stats.timeouts,
        "dropped_chunks": stats.dropped_chunks,
        "errors": stats.errors,
        "overflow_is_quality_warning": True,
    }
    if hit_count:
        status = "passed"
    elif partial_count:
        status = "partial"
    else:
        status = "failed"
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": now_iso(),
        "status": status,
        "platform": args.platform,
        "channel": BLE_ADV_CHANNEL,
        "frequency_hz": BLE_ADV_FREQUENCY_HZ,
        "input_len": len(payload),
        "input_hex": payload.hex(),
        "candidate_count": candidate_count,
        "segment_count": segment_count,
        "hit_count": hit_count,
        "partial_hit_count": partial_count,
        "best_candidate": best,
        "uart": uart,
        "capture_quality": capture_quality,
        "paths": {
            "input": str(root / "input.json"),
            "uart_ack": str(root / "uart_ack.json"),
            "usrp_config": str(root / "usrp_config.json"),
            "parser_config": str(root / "parser_config.json"),
            "parser_output": str(parser_path),
            "summary": str(root / "result_summary.json"),
        },
        "scope_exclusions": build_run_manifest(args, payload, run_id, root)["scope_exclusions"],
    }
    json_dump(root / "result_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status in ("passed", "partial") else 2


def _display_covert(payload: bytes) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.hex()
    return text if text.isprintable() else payload.hex()


def run_observe(args: argparse.Namespace) -> int:
    """Continuously print validated PhantomChannel advertising packets."""
    source = make_source(args, Path(args.output_root).expanduser().resolve())
    segmenter = ThresholdSegmenter(args.threshold, args.min_len, args.max_len, args.pre_samples)
    deadline = time.monotonic() + args.duration_s if args.duration_s > 0 else None
    print("PhantomChannel USRP receiver", flush=True)
    print(
        f"Channel: {BLE_ADV_CHANNEL} ({BLE_ADV_FREQUENCY_HZ / 1e6:.0f} MHz), "
        f"sample rate: {args.sample_rate / 1e6:g} MS/s, gain: {args.gain:g} dB, "
        f"antenna: {args.antenna}",
        flush=True,
    )
    source.start()
    found = 0
    seen: tuple[str, str, str] | None = None
    try:
        while deadline is None or time.monotonic() < deadline:
            timeout = 0.5 if deadline is None else min(0.5, max(0.01, deadline - time.monotonic()))
            item = source.read(timeout)
            if item is None:
                if args.source == "replay":
                    break
                if source.stats.errors:
                    raise RuntimeError(source.stats.errors[-1])
                continue
            _offset, samples = item
            for _segment_offset, segment in segmenter.push(samples):
                for decoded in decode_iq(
                    segment,
                    sample_rate=args.sample_rate,
                    channel=BLE_ADV_CHANNEL,
                    freq_dev=args.freq_dev,
                    aa_tolerance=args.aa_tolerance,
                    drop_last_extra=args.drop_last_extra,
                    strip_hw_crc=args.strip_hw_crc,
                ):
                    if not decoded["pc_frame_valid"]:
                        continue
                    if args.address and decoded["advertiser_address"].upper() != args.address.upper():
                        continue
                    key = (
                        decoded["advertiser_address"],
                        decoded["normal_payload_hex"],
                        decoded["covert_hex"],
                    )
                    if key == seen and not args.show_duplicates:
                        continue
                    seen = key
                    found += 1
                    covert = bytes.fromhex(decoded["covert_hex"])
                    print(f"Address: {decoded['advertiser_address']}", flush=True)
                    print(f"Normal payload: {decoded['normal_payload_hex']}", flush=True)
                    print(f"Covert data: {_display_covert(covert)}", flush=True)
                    if _display_covert(covert) != covert.hex():
                        print(f"Covert data hex: {covert.hex()}", flush=True)
                    print(flush=True)
                    if args.once:
                        return 0
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(f"Receiver error: {exc}", file=sys.stderr)
        return 2
    finally:
        source.stop()
    return 0 if found else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("nrf52840", "da14695", "efr32mg24", "cc2340r5"), required=True)
    parser.add_argument("--covert-hex")
    parser.add_argument("--covert-data")
    parser.add_argument("--observe", action="store_true", help="receive without known covert data and print validated packets")
    parser.add_argument("--once", action="store_true", help="with --observe, exit after the first decoded packet")
    parser.add_argument("--show-duplicates", action="store_true", help="with --observe, print repeated identical packets")
    parser.add_argument("--uart", default="")
    parser.add_argument("--address", default="", help="optional advertiser address filter")
    parser.add_argument("--uart-command", help="custom command template using {hex} or {data}")
    parser.add_argument("--no-uart", action="store_true")
    parser.add_argument("--ack-timeout-s", type=float, default=2.0)
    parser.add_argument("--source", choices=("uhd", "udp", "replay"), default="uhd")
    parser.add_argument("--input-iq", default="")
    parser.add_argument("--iq-format", choices=("sc16", "complex64"), default="sc16")
    parser.add_argument("--usrp-args", default="type=b200,master_clock_rate=16e6")
    parser.add_argument("--sample-rate", type=float, default=BLE_SAMPLE_RATE_SPS)
    parser.add_argument("--gain", type=float, default=50.0)
    parser.add_argument("--antenna", default="RX2")
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--udp-max-bytes", type=int, default=1 << 20)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--uart-delay-s", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-len", type=int, default=DEFAULT_MIN_SEGMENT_LEN)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_SEGMENT_LEN)
    parser.add_argument("--pre-samples", type=int, default=80)
    parser.add_argument("--freq-dev", type=float, default=250000.0)
    parser.add_argument("--aa-tolerance", type=int, default=DEFAULT_AA_TOLERANCE)
    parser.add_argument("--drop-last-extra", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip-hw-crc", action=argparse.BooleanOptionalAction, default=DEFAULT_STRIP_HARDWARE_CRC)
    parser.add_argument("--chunk-samples", type=int, default=4096)
    parser.add_argument("--queue-depth", type=int, default=16)
    parser.add_argument("--save-iq", action="store_true")
    parser.add_argument("--output-root", default="artifacts/broadcast-runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.source == "replay" and not args.input_iq:
        raise SystemExit("--source replay requires --input-iq")
    if args.duration_s <= 0 and not args.observe:
        raise SystemExit("--duration-s must be positive (zero means continuous only with --observe)")
    if args.observe and (args.covert_hex is not None or args.covert_data is not None):
        raise SystemExit("--observe does not accept --covert-hex or --covert-data")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
