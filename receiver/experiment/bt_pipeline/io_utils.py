import json
import socket
import struct
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


BYTES_PER_PACKET = 65535
FRAMED_MAGIC = 0x49515548
FRAMED_VERSION = 1
FLAG_HAS_IQ_PAYLOAD = 0x00000001
FLAG_OVERFLOW = 0x00000002
FLAG_GAP_DETECTED = 0x00000004
FLAG_START_OF_CAPTURE = 0x00000008
FLAG_END_OF_CAPTURE = 0x00000010
FLAG_FRAGMENTED = 0x00000020
FLAG_UDP_DROPPED_BEFORE_SEND = 0x00000040
PAYLOAD_FORMATS = {"sc16": 1, "complex64": 2}
PAYLOAD_FORMAT_NAMES = {value: key for key, value in PAYLOAD_FORMATS.items()}
# UHD framed UDP header, little-endian.
# 40 MHz complex IQ note: rx_time can repair the time axis after overflow,
# but it cannot recover IQ samples lost during overflow. Packets spanning
# gaps or incomplete blocks should be treated as unreliable.
FRAMED_HEADER_FORMAT = "<IHHIQHHQQIIIQddddIIQqQQ"
FRAMED_HEADER_LEN = struct.calcsize(FRAMED_HEADER_FORMAT)
RX_TAG_FIELDS = [
    "block_seq",
    "file_sample_offset",
    "block_sample_offset",
    "nsamps",
    "rx_time_full_secs",
    "rx_time_frac_secs",
    "rx_time_sec",
    "sample_rate",
    "center_freq",
    "gain",
    "payload_format",
    "error_code",
    "overflow_count_total",
    "gap_samples_est",
    "flags",
    "fragment_count",
    "received_fragment_count",
    "complete",
    "udp_seq_gap",
    "local_receive_time",
]


@dataclass
class FramedUdpHeader:
    magic: int
    version: int
    header_len: int
    flags: int
    block_seq: int
    fragment_index: int
    fragment_count: int
    file_sample_offset: int
    block_sample_offset: int
    fragment_sample_offset: int
    fragment_nsamps: int
    block_nsamps: int
    rx_time_full_secs: int
    rx_time_frac_secs: float
    sample_rate: float
    center_freq: float
    gain: float
    payload_format: int
    error_code: int
    overflow_count_total: int
    gap_samples_est: int
    reserved0: int
    reserved1: int

    @property
    def rx_time_sec(self):
        return float(self.rx_time_full_secs) + float(self.rx_time_frac_secs)

    @property
    def payload_format_name(self):
        return PAYLOAD_FORMAT_NAMES.get(self.payload_format, f"unknown_{self.payload_format}")


@dataclass
class RxTimeTag:
    block_seq: int
    file_sample_offset: int
    block_sample_offset: int
    nsamps: int
    rx_time_sec: float
    sample_rate: float
    center_freq: float
    gain: float
    payload_format: str
    error_code: int
    overflow_count_total: int
    gap_samples_est: int
    flags: int
    complete: bool

    @property
    def end_sample(self):
        return self.file_sample_offset + self.nsamps


class RxTimeTagLookup:
    def __init__(self, tags):
        self.tags = [tag for tag in (tags or []) if tag.nsamps > 0]
        self.offsets = [tag.file_sample_offset for tag in self.tags]

    def __bool__(self):
        return bool(self.tags)

    def __iter__(self):
        return iter(self.tags)

    def __len__(self):
        return len(self.tags)

    def slice_for_range(self, start_sample, end_sample):
        if not self.tags:
            return self
        start = float(start_sample)
        end = float(end_sample)
        left = max(0, bisect_right(self.offsets, start) - 1)
        right = min(len(self.tags), bisect_right(self.offsets, end) + 1)
        return RxTimeTagLookup(self.tags[left:right])


def build_rx_time_tag_lookup(tags):
    if isinstance(tags, RxTimeTagLookup):
        return tags
    return RxTimeTagLookup(tags)


class PendingBlock:
    def __init__(self, header):
        self.header = header
        self.fragments = {}
        self.local_receive_time = datetime.now().isoformat(timespec="milliseconds")

    def add_fragment(self, header, payload):
        self.fragments[int(header.fragment_index)] = (header, payload)

    @property
    def expected_fragment_count(self):
        return max(1, int(self.header.fragment_count))

    @property
    def is_complete(self):
        return len(self.fragments) >= self.expected_fragment_count

    def ordered_payload(self):
        return b"".join(self.fragments[idx][1] for idx in sorted(self.fragments))

    def received_fragment_count(self):
        return len(self.fragments)


def utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sample_index_to_us(sample_index, sample_rate):
    # At 4 MHz, 1 sample = 0.25 us, so timestamp_us = sample_index / 4.
    return float(sample_index) * 1e6 / float(sample_rate)


def sample_index_to_s(sample_index, sample_rate):
    return float(sample_index) / float(sample_rate)


def capture_udp_to_bin(
    udp_ip,
    udp_port,
    sample_rate,
    duration,
    iq_format,
    output_bin,
    metadata_path,
    center_freq=None,
):
    output_bin = Path(output_bin)
    metadata_path = Path(metadata_path)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    start_wall = utc_now_iso()
    start_mono = time.monotonic()
    deadline = start_mono + float(duration)
    total_bytes = 0
    packet_count = 0
    interrupted = False

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
    sock.settimeout(0.2)
    sock.bind((udp_ip, int(udp_port)))

    print(f"Listening on UDP {udp_ip}:{udp_port} for {duration:.2f}s")
    try:
        with output_bin.open("wb") as f:
            while time.monotonic() < deadline:
                try:
                    packet, _ = sock.recvfrom(BYTES_PER_PACKET)
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    interrupted = True
                    break

                if not packet:
                    continue
                f.write(packet)
                total_bytes += len(packet)
                packet_count += 1
    finally:
        sock.close()

    end_mono = time.monotonic()
    end_wall = utc_now_iso()
    bytes_per_sample = 8 if iq_format == "complex64" else 4
    estimated_samples = total_bytes // bytes_per_sample
    metadata = {
        "sample_rate": float(sample_rate),
        "duration_requested_s": float(duration),
        "duration_actual_s": end_mono - start_mono,
        "center_freq": center_freq,
        "iq_format": iq_format,
        "file_name": str(output_bin),
        "metadata_file": str(metadata_path),
        "start_time_utc": start_wall,
        "end_time_utc": end_wall,
        "total_bytes": total_bytes,
        "packet_count": packet_count,
        "estimated_samples": int(estimated_samples),
        "interrupted": interrupted,
        "drop_detected": None,
        "note": "UDP has no built-in sequence counter here, so packet loss cannot be proven from raw payload only.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved IQ bin: {output_bin.resolve()}")
    print(f"Saved metadata: {metadata_path.resolve()}")
    return metadata


def parse_framed_udp_header(packet):
    if len(packet) < FRAMED_HEADER_LEN:
        return None, b"", "short_header"
    values = struct.unpack(FRAMED_HEADER_FORMAT, packet[:FRAMED_HEADER_LEN])
    header = FramedUdpHeader(*values)
    if header.magic != FRAMED_MAGIC:
        return header, b"", "bad_magic"
    if header.version != FRAMED_VERSION:
        return header, b"", "bad_version"
    if header.header_len < FRAMED_HEADER_LEN or header.header_len > len(packet):
        return header, b"", "bad_header_len"
    return header, packet[header.header_len:], None


def payload_bytes_per_sample(payload_format_name):
    if payload_format_name == "sc16":
        return 4
    if payload_format_name == "complex64":
        return 8
    raise ValueError(f"Unsupported payload format: {payload_format_name}")


def open_rx_tag_writer(path):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=RX_TAG_FIELDS)
    writer.writeheader()
    return f, writer


def rx_tag_row_from_header(
    header,
    file_sample_offset,
    nsamps,
    complete,
    received_fragment_count,
    udp_seq_gap,
    local_receive_time,
):
    return {
        "block_seq": int(header.block_seq),
        "file_sample_offset": int(file_sample_offset),
        "block_sample_offset": int(header.block_sample_offset),
        "nsamps": int(nsamps),
        "rx_time_full_secs": int(header.rx_time_full_secs),
        "rx_time_frac_secs": repr(float(header.rx_time_frac_secs)),
        "rx_time_sec": repr(float(header.rx_time_sec)),
        "sample_rate": repr(float(header.sample_rate)),
        "center_freq": repr(float(header.center_freq)),
        "gain": repr(float(header.gain)),
        "payload_format": header.payload_format_name,
        "error_code": int(header.error_code),
        "overflow_count_total": int(header.overflow_count_total),
        "gap_samples_est": int(header.gap_samples_est),
        "flags": int(header.flags),
        "fragment_count": int(header.fragment_count),
        "received_fragment_count": int(received_fragment_count),
        "complete": bool(complete),
        "udp_seq_gap": bool(udp_seq_gap),
        "local_receive_time": local_receive_time,
    }


def capture_framed_udp_to_bin(
    udp_ip,
    udp_port,
    sample_rate,
    duration,
    iq_format,
    output_bin,
    metadata_json,
    rx_tags_csv,
    center_freq,
    expected_payload_format="sc16",
    strict_fragments=False,
    socket_rcvbuf=256 * 1024 * 1024,
    max_packet_size=65535,
    bandwidth=None,
):
    output_bin = Path(output_bin)
    metadata_json = Path(metadata_json)
    rx_tags_csv = Path(rx_tags_csv)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    expected_payload_id = PAYLOAD_FORMATS[expected_payload_format]
    start_wall = utc_now_iso()
    start_mono = time.monotonic()
    deadline = start_mono + float(duration)
    next_report = start_mono + 1.0
    total_bytes = 0
    samples_written = 0
    blocks_written = 0
    metadata_only_blocks = 0
    incomplete_blocks = 0
    invalid_packets = 0
    payload_format_mismatch = 0
    udp_missing_blocks = 0
    gap_count = 0
    last_overflow_count = 0
    last_block_seq = None
    actual_sample_rate = None
    actual_center_freq = None
    pending = {}
    interrupted = False

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(socket_rcvbuf))
    sock.settimeout(0.2)
    sock.bind((udp_ip, int(udp_port)))

    tag_file, tag_writer = open_rx_tag_writer(rx_tags_csv)

    def flush_block(block, output_file, complete_override=None):
        nonlocal samples_written, blocks_written, incomplete_blocks, metadata_only_blocks, total_bytes
        header = block.header
        complete = block.is_complete if complete_override is None else bool(complete_override)
        payload = block.ordered_payload() if complete else b""
        has_payload = bool(header.flags & FLAG_HAS_IQ_PAYLOAD)
        bytes_per_sample = payload_bytes_per_sample(header.payload_format_name)
        actual_nsamps = len(payload) // bytes_per_sample if payload else 0
        expected_nsamps = int(header.block_nsamps)

        if has_payload and complete and payload:
            file_offset = samples_written
            output_file.write(payload)
            total_bytes += len(payload)
            samples_written += actual_nsamps
            blocks_written += 1
            tag_writer.writerow(
                rx_tag_row_from_header(
                    header,
                    file_offset,
                    actual_nsamps,
                    True,
                    block.received_fragment_count(),
                    False,
                    block.local_receive_time,
                )
            )
            return

        if has_payload:
            incomplete_blocks += 1
        else:
            metadata_only_blocks += 1

        tag_writer.writerow(
            rx_tag_row_from_header(
                header,
                samples_written,
                expected_nsamps if not has_payload else actual_nsamps,
                False,
                block.received_fragment_count(),
                False,
                block.local_receive_time,
            )
        )

    print(f"Listening for framed UDP {udp_ip}:{udp_port} for {duration:.2f}s")
    print(
        "Framed mode uses UHD rx_time tags. rx_time fixes the time axis after overflow, "
        "but it cannot recover IQ samples lost during overflow."
    )

    try:
        with output_bin.open("wb") as output_file:
            while time.monotonic() < deadline:
                try:
                    packet, _ = sock.recvfrom(max_packet_size)
                except socket.timeout:
                    now = time.monotonic()
                    if now >= next_report:
                        elapsed = max(now - start_mono, 1e-9)
                        print(
                            f"RX {total_bytes / elapsed / 1e6:.2f} MB/s, "
                            f"samples_written={samples_written}, blocks={blocks_written}, "
                            f"udp_block_gap={udp_missing_blocks}, incomplete_blocks={incomplete_blocks}, "
                            f"overflow_count={last_overflow_count}, gap_count={gap_count}"
                        )
                        next_report = now + 1.0
                    continue
                except KeyboardInterrupt:
                    interrupted = True
                    break

                header, payload, error = parse_framed_udp_header(packet)
                if error:
                    invalid_packets += 1
                    continue
                if header.payload_format != expected_payload_id:
                    payload_format_mismatch += 1
                    continue

                if header.sample_rate > 0:
                    actual_sample_rate = float(header.sample_rate)
                if header.center_freq > 0:
                    actual_center_freq = float(header.center_freq)

                if header.flags & FLAG_GAP_DETECTED or header.gap_samples_est:
                    gap_count += 1
                if header.flags & FLAG_OVERFLOW:
                    last_overflow_count = max(last_overflow_count, int(header.overflow_count_total))

                if last_block_seq is not None and header.block_seq > last_block_seq + 1:
                    udp_missing_blocks += int(header.block_seq - last_block_seq - 1)
                last_block_seq = max(int(header.block_seq), int(last_block_seq or header.block_seq))

                block = pending.get(header.block_seq)
                if block is None:
                    block = PendingBlock(header)
                    pending[header.block_seq] = block
                block.add_fragment(header, payload)

                if block.is_complete or header.fragment_count <= 1:
                    flush_block(block, output_file)
                    pending.pop(header.block_seq, None)

                if header.flags & FLAG_END_OF_CAPTURE:
                    break

                now = time.monotonic()
                if now >= next_report:
                    elapsed = max(now - start_mono, 1e-9)
                    print(
                        f"RX {total_bytes / elapsed / 1e6:.2f} MB/s, "
                        f"samples_written={samples_written}, blocks={blocks_written}, "
                        f"udp_block_gap={udp_missing_blocks}, incomplete_blocks={incomplete_blocks}, "
                        f"overflow_count={last_overflow_count}, gap_count={gap_count}"
                    )
                    next_report = now + 1.0

            for seq in sorted(pending):
                flush_block(pending[seq], output_file, complete_override=False)
    finally:
        tag_file.close()
        sock.close()

    end_mono = time.monotonic()
    metadata_sample_rate = float(actual_sample_rate if actual_sample_rate is not None else sample_rate)
    metadata_center_freq = float(actual_center_freq if actual_center_freq is not None else center_freq)
    metadata = {
        "sample_rate": metadata_sample_rate,
        "center_freq": metadata_center_freq,
        "requested_sample_rate": float(sample_rate),
        "requested_center_freq": float(center_freq),
        "bandwidth": None if bandwidth is None else float(bandwidth),
        "duration_requested_s": float(duration),
        "duration_actual_s": end_mono - start_mono,
        "iq_format": iq_format,
        "payload_format": expected_payload_format,
        "wideband_mode": "40m_capture_framed_udp",
        "file_name": str(output_bin),
        "metadata_file": str(metadata_json),
        "rx_tags_csv": str(rx_tags_csv),
        "timestamp_mode": "uhd_rx_time",
        "timestamp_note": (
            "Use rx_time_tags.csv to convert file sample index to UHD hardware time. "
            "Do not use sample_index/sample_rate as real receive time after overflow."
        ),
        "start_time_utc": start_wall,
        "end_time_utc": utc_now_iso(),
        "total_bytes": int(total_bytes),
        "total_samples_written": int(samples_written),
        "total_blocks": int(blocks_written),
        "metadata_only_blocks": int(metadata_only_blocks),
        "overflow_count_total": int(last_overflow_count),
        "gap_count": int(gap_count),
        "udp_missing_blocks": int(udp_missing_blocks),
        "incomplete_blocks": int(incomplete_blocks),
        "invalid_packets": int(invalid_packets),
        "payload_format_mismatch": int(payload_format_mismatch),
        "strict_fragments": bool(strict_fragments),
        "interrupted": interrupted,
        "important_note": (
            "rx_time can repair the time axis after overflow, but missing IQ is not recovered. "
            "Packets spanning overflow or incomplete blocks are unreliable."
        ),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved IQ bin: {output_bin.resolve()}")
    print(f"Saved rx time tags: {rx_tags_csv.resolve()}")
    print(f"Saved metadata: {metadata_json.resolve()}")
    return metadata


def load_metadata(metadata_path):
    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_rx_time_tags(path):
    import csv

    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    tags = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                fragment_count = int(float(row.get("fragment_count", 0) or 0))
                received_fragment_count = int(float(row.get("received_fragment_count", 0) or 0))
                nsamps = int(float(row.get("nsamps", 0)))
                complete_text = str(row.get("complete", "")).lower()
                complete = complete_text == "true"
                if not complete and nsamps > 0 and fragment_count > 0:
                    complete = fragment_count == received_fragment_count
                # Direct UHD capture rows are complete blocks but have no UDP
                # fragmentation columns.  Do not mark all of their timestamps
                # as being near an incomplete block.
                if "complete" not in row and nsamps > 0:
                    complete = int(float(row.get("error_code", 0) or 0)) == 0
                tags.append(
                    RxTimeTag(
                        block_seq=int(row.get("block_seq", 0)),
                        file_sample_offset=int(float(row.get("file_sample_offset", 0))),
                        block_sample_offset=int(float(row.get("block_sample_offset", 0))),
                        nsamps=nsamps,
                        rx_time_sec=float(row.get("rx_time_sec", 0.0)),
                        sample_rate=float(row.get("sample_rate", 0.0)),
                        center_freq=float(row.get("center_freq", 0.0)),
                        gain=float(row.get("gain", 0.0)),
                        payload_format=row.get("payload_format", ""),
                        error_code=int(float(row.get("error_code", 0))),
                        overflow_count_total=int(float(row.get("overflow_count_total", 0))),
                        gap_samples_est=int(float(row.get("gap_samples_est", 0))),
                        flags=int(float(row.get("flags", 0))),
                        complete=complete,
                    )
                )
            except (TypeError, ValueError):
                continue
    tags = _repair_degenerate_rx_time_file_offsets(tags)
    tags.sort(key=lambda tag: tag.file_sample_offset)
    return tags


def _repair_degenerate_rx_time_file_offsets(tags):
    if not tags:
        return tags

    payload_tags = [tag for tag in tags if tag.nsamps > 0]
    if not payload_tags:
        return tags

    file_offsets = [tag.file_sample_offset for tag in payload_tags]
    if len(set(file_offsets)) > 1:
        return tags

    block_offsets = [tag.block_sample_offset for tag in payload_tags]
    if len(set(block_offsets)) <= 1:
        return tags
    if any(b < a for a, b in zip(block_offsets, block_offsets[1:])):
        return tags

    repaired = []
    for tag in tags:
        repaired.append(
            RxTimeTag(
                block_seq=tag.block_seq,
                file_sample_offset=tag.block_sample_offset,
                block_sample_offset=tag.block_sample_offset,
                nsamps=tag.nsamps,
                rx_time_sec=tag.rx_time_sec,
                sample_rate=tag.sample_rate,
                center_freq=tag.center_freq,
                gain=tag.gain,
                payload_format=tag.payload_format,
                error_code=tag.error_code,
                overflow_count_total=tag.overflow_count_total,
                gap_samples_est=tag.gap_samples_est,
                flags=tag.flags,
                complete=tag.complete,
            )
        )
    return repaired


def sample_index_to_hw_time(sample_index, tags, sample_rate):
    if not tags:
        return None, "no_rx_time_tags", {}

    sample_index = float(sample_index)
    if isinstance(tags, RxTimeTagLookup):
        tag_values = tags.tags
        offsets = tags.offsets
    else:
        tag_values = [tag for tag in tags if tag.nsamps > 0]
        if not tag_values:
            return None, "no_rx_time_tags", {}
        offsets = [tag.file_sample_offset for tag in tag_values]
    pos = bisect_right(offsets, sample_index) - 1
    if pos < 0:
        tag = tag_values[0]
        status = "extrapolated"
    else:
        tag = tag_values[pos]
        status = "ok" if sample_index < tag.end_sample and tag.complete else "extrapolated"

    if (
        tag.flags & (FLAG_OVERFLOW | FLAG_GAP_DETECTED | FLAG_UDP_DROPPED_BEFORE_SEND)
        or tag.gap_samples_est
        or not tag.complete
    ):
        status = "gap_or_overflow_nearby" if status == "ok" else f"{status};gap_or_overflow_nearby"

    tag_rate = tag.sample_rate if tag.sample_rate > 0 else float(sample_rate)
    hw_time_sec = tag.rx_time_sec + (sample_index - tag.file_sample_offset) / tag_rate
    info = {
        "rx_time_tag_block_seq": tag.block_seq,
        "rx_time_tag_offset": sample_index - tag.file_sample_offset,
        "gap_samples_est": tag.gap_samples_est,
        "overflow_count_total": tag.overflow_count_total,
    }
    return hw_time_sec, status, info


def load_iq_bin(input_bin, iq_format="complex64", remove_dc=False, normalize=False):
    path = Path(input_bin)
    data = path.read_bytes()
    if iq_format == "complex64":
        usable = len(data) - (len(data) % np.dtype(np.complex64).itemsize)
        iq = np.frombuffer(data[:usable], dtype=np.complex64).astype(np.complex64, copy=False)
    elif iq_format == "int16":
        usable = len(data) - (len(data) % (2 * np.dtype(np.int16).itemsize))
        raw = np.frombuffer(data[:usable], dtype=np.int16)
        if raw.size < 2:
            iq = np.empty(0, dtype=np.complex64)
        else:
            pairs = raw[: raw.size - (raw.size % 2)].reshape(-1, 2)
            iq = (pairs[:, 0].astype(np.float32) + 1j * pairs[:, 1].astype(np.float32)) / 32768.0
            iq = iq.astype(np.complex64, copy=False)
    else:
        raise ValueError(f"Unsupported iq_format: {iq_format}")

    if remove_dc:
        iq = remove_dc_offset(iq)
    if normalize:
        iq = normalize_iq(iq)
    return iq


def remove_dc_offset(iq):
    if iq.size == 0:
        return iq
    return (iq - np.mean(iq)).astype(np.complex64, copy=False)


def normalize_iq(iq):
    if iq.size == 0:
        return iq
    scale = np.percentile(np.abs(iq), 95)
    if not np.isfinite(scale) or scale <= 0:
        return iq
    return (iq / scale).astype(np.complex64, copy=False)
