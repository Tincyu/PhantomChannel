from dataclasses import dataclass
import socket
import sys
import threading
from datetime import datetime
from time import perf_counter, sleep

import numpy as np

from . import cuda_backend
from .io_utils import (
    FLAG_END_OF_CAPTURE,
    FLAG_HAS_IQ_PAYLOAD,
    PendingBlock,
    RxTimeTag,
    build_rx_time_tag_lookup,
    parse_framed_udp_header,
)
from .wideband_channelizer import iter_iq_chunks


@dataclass
class RealtimeChunk:
    chunk_seq: int
    chunk_start: int
    core_start: int
    core_end: int
    raw_iq: object
    rx_time: float | None = None


class RealtimeIQSource:
    def read_chunk(self):
        raise NotImplementedError

    def close(self):
        return None


class FileReplayIQSource(RealtimeIQSource):
    def __init__(
        self,
        input_bin,
        iq_format,
        chunk_samples,
        overlap_samples,
        decode_int16=True,
    ):
        self._chunk_seq = 0
        self._iterator = iter_iq_chunks(
            input_bin,
            iq_format,
            chunk_samples,
            overlap_samples,
            decode_int16=decode_int16,
        )

    def read_chunk(self):
        try:
            chunk_start, core_start, core_end, chunk = next(self._iterator)
        except StopIteration:
            return None
        item = RealtimeChunk(
            chunk_seq=self._chunk_seq,
            chunk_start=chunk_start,
            core_start=core_start,
            core_end=core_end,
            raw_iq=chunk,
        )
        self._chunk_seq += 1
        return item


class RawPipeIQSource(RealtimeIQSource):
    def __init__(
        self,
        stream=None,
        iq_format="int16",
        chunk_samples=16_000_000,
        overlap_samples=0,
        decode_int16=True,
        pinned_host_buffer=False,
        pinned_slots=1,
    ):
        self.stream = stream if stream is not None else sys.stdin.buffer
        self.iq_format = iq_format
        self.chunk_samples = int(chunk_samples)
        self.overlap_samples = int(overlap_samples)
        self.decode_int16 = bool(decode_int16)
        self.pinned_host_buffer = bool(pinned_host_buffer)
        self.pinned_slots = max(1, int(pinned_slots))
        self._chunk_seq = 0
        self._core_start = 0
        self._tail = None
        self._pinned_slots = [
            {"memory": None, "view": None, "capacity": 0}
            for _slot_index in range(self.pinned_slots)
        ]
        self._next_pinned_slot = 0

    def read_chunk(self):
        core = self._read_core()
        if core is None:
            return None

        tail_samples = self._array_sample_count(self._tail) if self._tail is not None else 0
        raw_iq = self._concat_tail(core)
        chunk_start = self._core_start - tail_samples
        core_start = self._core_start
        core_samples = self._array_sample_count(core)
        core_end = core_start + core_samples

        self._tail = self._take_tail(raw_iq)
        self._core_start = core_end
        item = RealtimeChunk(
            chunk_seq=self._chunk_seq,
            chunk_start=chunk_start,
            core_start=core_start,
            core_end=core_end,
            raw_iq=raw_iq,
        )
        self._chunk_seq += 1
        return item

    def _read_core(self):
        if self.iq_format == "int16":
            itemsize = np.dtype(np.int16).itemsize
            byte_count = self.chunk_samples * 2 * itemsize
            raw, raw_size = self._read_exact_or_partial(byte_count)
            if raw_size <= 0:
                return None
            arr = np.frombuffer(raw, dtype=np.int16, count=raw_size // itemsize)
            arr = arr[: arr.size - (arr.size % 2)]
            if arr.size == 0:
                return None
            if not self.decode_int16:
                if self.pinned_host_buffer:
                    return arr
                return arr.copy()
            iq = np.empty(arr.size // 2, dtype=np.complex64)
            iq.real = arr[0::2]
            iq.imag = arr[1::2]
            iq.real *= np.float32(1.0 / 32768.0)
            iq.imag *= np.float32(1.0 / 32768.0)
            return iq
        if self.iq_format == "complex64":
            itemsize = np.dtype(np.complex64).itemsize
            raw, raw_size = self._read_exact_or_partial(self.chunk_samples * itemsize)
            if raw_size <= 0:
                return None
            usable = raw_size - (raw_size % itemsize)
            if usable <= 0:
                return None
            arr = np.frombuffer(raw, dtype=np.complex64, count=usable // itemsize)
            if self.pinned_host_buffer:
                return arr
            return arr.copy()
        raise ValueError(f"Unsupported iq_format: {self.iq_format}")

    def _read_exact_or_partial(self, byte_count):
        if self.pinned_host_buffer:
            return self._read_exact_or_partial_pinned(byte_count)
        chunks = []
        remaining = int(byte_count)
        total = 0
        while remaining > 0:
            data = self.stream.read(remaining)
            if not data:
                break
            chunks.append(data)
            total += len(data)
            remaining -= len(data)
        return b"".join(chunks), total

    def _read_exact_or_partial_pinned(self, byte_count):
        byte_count = int(byte_count)
        slot_index = self._next_pinned_slot
        self._next_pinned_slot = (self._next_pinned_slot + 1) % self.pinned_slots
        self._ensure_pinned_capacity(slot_index, byte_count)
        view = self._pinned_slots[slot_index]["view"]
        total = 0
        while total < byte_count:
            nread = self.stream.readinto(view[total:byte_count])
            if not nread:
                break
            total += int(nread)
        return view[:total], total

    def _ensure_pinned_capacity(self, slot_index, byte_count):
        slot = self._pinned_slots[slot_index]
        if slot["capacity"] >= byte_count:
            return
        cp = cuda_backend.require_cuda()
        slot["memory"] = cp.cuda.alloc_pinned_memory(int(byte_count))
        slot["view"] = memoryview(slot["memory"])
        slot["capacity"] = int(byte_count)

    def _concat_tail(self, core):
        if self._tail is None or self._array_sample_count(self._tail) == 0:
            return core
        return np.concatenate((self._tail, core))

    def _take_tail(self, raw_iq):
        keep = min(max(0, self.overlap_samples), self._array_sample_count(raw_iq))
        if keep <= 0:
            return None
        if self.iq_format == "int16" and not self.decode_int16:
            return raw_iq[-keep * 2 :].copy()
        return raw_iq[-keep:].copy()

    def _array_sample_count(self, array):
        if array is None:
            return 0
        if self.iq_format == "int16" and not self.decode_int16:
            return int(array.size // 2)
        return int(array.size)


class UdpFramedTagReceiver:
    def __init__(
        self,
        udp_ip="0.0.0.0",
        udp_port=9002,
        socket_rcvbuf=64 * 1024 * 1024,
        max_packet_size=65535,
        timeout_s=0.05,
        expected_payload_format="sc16",
    ):
        if expected_payload_format != "sc16":
            raise ValueError("UDP tag side-channel currently supports sc16 payload metadata")
        self.max_packet_size = int(max_packet_size)
        self.timeout_s = float(timeout_s)
        self._expected_payload_format = expected_payload_format
        self._closed = False
        self._lock = threading.Lock()
        self._thread = None
        self._tags = []
        self._last_block_seq = None
        self._packets_rx = 0
        self._metadata_blocks_rx = 0
        self._payload_meta_blocks_rx = 0
        self._invalid_packets = 0
        self._payload_format_mismatch = 0
        self._udp_missing_blocks = 0
        self._socket_timeouts = 0
        self._last_tag_end = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(socket_rcvbuf))
        self.sock.settimeout(self.timeout_s)
        self.sock.bind((udp_ip, int(udp_port)))

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="udp-framed-tag-receiver", daemon=True)
        self._thread.start()

    def close(self):
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def lookup(self):
        with self._lock:
            return build_rx_time_tag_lookup(list(self._tags))

    def wait_until_covered(self, sample_end, timeout_s=0.2):
        deadline = perf_counter() + max(0.0, float(timeout_s))
        while perf_counter() < deadline:
            with self._lock:
                if self._last_tag_end >= int(sample_end):
                    return True
            if self._closed:
                return False
            sleep(0.001)
        with self._lock:
            return self._last_tag_end >= int(sample_end)

    def stats_detail(self):
        with self._lock:
            return (
                f"tag_pkts={self._packets_rx},"
                f"tag_blocks={self._metadata_blocks_rx},"
                f"tag_payload_blocks={self._payload_meta_blocks_rx},"
                f"tag_count={len(self._tags)},"
                f"tag_last_end={self._last_tag_end},"
                f"tag_gap={self._udp_missing_blocks},"
                f"tag_bad={self._invalid_packets},"
                f"tag_fmt_bad={self._payload_format_mismatch},"
                f"tag_timeouts={self._socket_timeouts}"
            )

    def _run(self):
        while not self._closed:
            try:
                packet, _addr = self.sock.recvfrom(self.max_packet_size)
            except socket.timeout:
                with self._lock:
                    self._socket_timeouts += 1
                continue
            except OSError:
                break
            self._handle_packet(packet)

    def _handle_packet(self, packet):
        header, _payload, error = parse_framed_udp_header(packet)
        with self._lock:
            self._packets_rx += 1
            if error:
                self._invalid_packets += 1
                return
            if header.payload_format_name != self._expected_payload_format:
                self._payload_format_mismatch += 1
                return
            if self._last_block_seq is not None and header.block_seq > self._last_block_seq + 1:
                self._udp_missing_blocks += int(header.block_seq - self._last_block_seq - 1)
            self._last_block_seq = max(int(header.block_seq), int(self._last_block_seq or header.block_seq))

            nsamps = int(header.block_nsamps or header.fragment_nsamps)
            complete = bool(nsamps > 0)
            tag = RxTimeTag(
                block_seq=int(header.block_seq),
                file_sample_offset=int(header.file_sample_offset),
                block_sample_offset=int(header.block_sample_offset),
                nsamps=nsamps,
                rx_time_sec=float(header.rx_time_sec),
                sample_rate=float(header.sample_rate),
                center_freq=float(header.center_freq),
                gain=float(header.gain),
                payload_format=header.payload_format_name,
                error_code=int(header.error_code),
                overflow_count_total=int(header.overflow_count_total),
                gap_samples_est=int(header.gap_samples_est),
                flags=int(header.flags),
                complete=complete,
            )
            self._tags.append(tag)
            self._metadata_blocks_rx += 1
            if header.flags & FLAG_HAS_IQ_PAYLOAD:
                self._payload_meta_blocks_rx += 1
            if tag.end_sample > self._last_tag_end:
                self._last_tag_end = tag.end_sample


class RawPipeWithUdpTagsIQSource(RealtimeIQSource):
    def __init__(
        self,
        stream=None,
        iq_format="int16",
        chunk_samples=16_000_000,
        overlap_samples=0,
        decode_int16=True,
        pinned_host_buffer=False,
        pinned_slots=1,
        udp_ip="0.0.0.0",
        udp_port=9002,
        udp_timeout_s=0.05,
        udp_socket_rcvbuf=64 * 1024 * 1024,
        udp_max_packet_size=65535,
        tag_wait_s=0.2,
    ):
        self.pipe = RawPipeIQSource(
            stream=stream,
            iq_format=iq_format,
            chunk_samples=chunk_samples,
            overlap_samples=overlap_samples,
            decode_int16=decode_int16,
            pinned_host_buffer=pinned_host_buffer,
            pinned_slots=pinned_slots,
        )
        self.tags = UdpFramedTagReceiver(
            udp_ip=udp_ip,
            udp_port=udp_port,
            socket_rcvbuf=udp_socket_rcvbuf,
            max_packet_size=udp_max_packet_size,
            timeout_s=udp_timeout_s,
        )
        self.tag_wait_s = float(tag_wait_s)
        self._chunks = 0
        self._tag_wait_misses = 0
        self.tags.start()

    def read_chunk(self):
        chunk = self.pipe.read_chunk()
        if chunk is None:
            return None
        if not self.tags.wait_until_covered(chunk.core_end, self.tag_wait_s):
            self._tag_wait_misses += 1
        self._chunks += 1
        return chunk

    def rx_time_lookup(self):
        return self.tags.lookup()

    def stats_detail(self):
        return f"pipe_chunks={self._chunks},tag_wait_miss={self._tag_wait_misses},{self.tags.stats_detail()}"

    def close(self):
        self.tags.close()
        self.pipe.close()


class FramedUdpIQSource(RealtimeIQSource):
    def __init__(
        self,
        udp_ip="0.0.0.0",
        udp_port=9002,
        iq_format="int16",
        chunk_samples=16_000_000,
        overlap_samples=0,
        decode_int16=True,
        socket_rcvbuf=256 * 1024 * 1024,
        max_packet_size=65535,
        timeout_s=0.2,
        expected_payload_format="sc16",
    ):
        if iq_format != "int16":
            raise ValueError("--source udp-framed currently requires --iq-format int16")
        if expected_payload_format != "sc16":
            raise ValueError("--source udp-framed currently supports sc16 payloads")
        self.iq_format = iq_format
        self.chunk_samples = int(chunk_samples)
        self.overlap_samples = int(overlap_samples)
        self.decode_int16 = bool(decode_int16)
        self.max_packet_size = int(max_packet_size)
        self._chunk_seq = 0
        self._core_start = 0
        self._tail = None
        self._pending_blocks = {}
        self._core_parts = []
        self._core_buffer = None
        self._core_samples = 0
        self._rx_time_tags = []
        self._last_block_seq = None
        self._closed = False
        self._saw_end = False
        self._invalid_packets = 0
        self._udp_missing_blocks = 0
        self._incomplete_blocks = 0
        self._metadata_only_blocks = 0
        self._payload_format_mismatch = 0
        self._packets_rx = 0
        self._payload_blocks_rx = 0
        self._samples_rx = 0
        self._bytes_rx = 0
        self._socket_timeouts = 0
        self._chunk_datagrams = 0
        self._chunk_payload_blocks = 0
        self._chunk_samples = 0
        self._chunk_bytes = 0
        self._last_chunk_stats = {}

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(socket_rcvbuf))
        self.sock.settimeout(float(timeout_s))
        self.sock.bind((udp_ip, int(udp_port)))

    def read_chunk(self):
        chunk_start_time = perf_counter()
        self._chunk_datagrams = 0
        self._chunk_payload_blocks = 0
        self._chunk_samples = 0
        self._chunk_bytes = 0
        while self._core_samples < self.chunk_samples and not self._saw_end:
            self._recv_one_packet()

        if self._core_samples <= 0:
            return None

        core = self._consume_core()
        tail_samples = self._array_sample_count(self._tail) if self._tail is not None else 0
        raw_iq = self._concat_tail(core)
        chunk_start = self._core_start - tail_samples
        core_start = self._core_start
        core_samples = self._array_sample_count(core)
        core_end = core_start + core_samples

        self._tail = self._take_tail(raw_iq)
        self._core_start = core_end
        item = RealtimeChunk(
            chunk_seq=self._chunk_seq,
            chunk_start=chunk_start,
            core_start=core_start,
            core_end=core_end,
            raw_iq=raw_iq,
        )
        assemble_s = perf_counter() - chunk_start_time
        self._last_chunk_stats = {
            "assemble_s": assemble_s,
            "datagrams": self._chunk_datagrams,
            "payload_blocks": self._chunk_payload_blocks,
            "samples": self._chunk_samples,
            "bytes": self._chunk_bytes,
            "msps": self._chunk_samples / assemble_s / 1e6 if assemble_s > 0 else 0.0,
            "mbps": self._chunk_bytes / assemble_s / 1e6 if assemble_s > 0 else 0.0,
        }
        self._chunk_seq += 1
        return item

    def rx_time_lookup(self):
        return build_rx_time_tag_lookup(self._rx_time_tags)

    def stats_detail(self):
        chunk = self._last_chunk_stats
        if not chunk:
            return (
                "src_pkts=0,src_blocks=0,src_msps=0.000,src_MBps=0.000,"
                "src_gap=0,src_bad=0,src_incomplete=0,src_meta=0,src_timeouts=0"
            )
        return (
            f"src_pkts={self._packets_rx},"
            f"src_blocks={self._payload_blocks_rx},"
            f"src_samples={self._samples_rx},"
            f"src_tags={len(self._rx_time_tags)},"
            f"src_last_pkts={chunk.get('datagrams', 0)},"
            f"src_last_blocks={chunk.get('payload_blocks', 0)},"
            f"src_last_samples={chunk.get('samples', 0)},"
            f"src_assemble={chunk.get('assemble_s', 0.0):.3f}s,"
            f"src_msps={chunk.get('msps', 0.0):.3f},"
            f"src_MBps={chunk.get('mbps', 0.0):.3f},"
            f"src_gap={self._udp_missing_blocks},"
            f"src_bad={self._invalid_packets},"
            f"src_fmt_bad={self._payload_format_mismatch},"
            f"src_incomplete={self._incomplete_blocks},"
            f"src_meta={self._metadata_only_blocks},"
            f"src_timeouts={self._socket_timeouts}"
        )

    def close(self):
        self._closed = True
        self.sock.close()

    def _recv_one_packet(self):
        while True:
            try:
                packet, _addr = self.sock.recvfrom(self.max_packet_size)
                break
            except socket.timeout:
                self._socket_timeouts += 1
                if self._core_samples > 0:
                    return
                continue
        self._packets_rx += 1
        self._chunk_datagrams += 1
        self._bytes_rx += len(packet)
        self._chunk_bytes += len(packet)
        header, payload, error = parse_framed_udp_header(packet)
        if error:
            self._invalid_packets += 1
            return
        if header.payload_format_name != "sc16":
            self._payload_format_mismatch += 1
            return

        if self._last_block_seq is not None and header.block_seq > self._last_block_seq + 1:
            self._udp_missing_blocks += int(header.block_seq - self._last_block_seq - 1)
        self._last_block_seq = max(int(header.block_seq), int(self._last_block_seq or header.block_seq))

        if header.fragment_count <= 1:
            self._flush_single_packet_block(header, payload)
            if header.flags & FLAG_END_OF_CAPTURE:
                self._saw_end = True
            return

        block = self._pending_blocks.get(header.block_seq)
        if block is None:
            block = PendingBlock(header)
            block.local_receive_time = datetime.now().isoformat(timespec="milliseconds")
            self._pending_blocks[header.block_seq] = block
        block.add_fragment(header, payload)

        if block.is_complete or header.fragment_count <= 1:
            self._flush_block(block)
            self._pending_blocks.pop(header.block_seq, None)

        if header.flags & FLAG_END_OF_CAPTURE:
            self._saw_end = True

    def _flush_single_packet_block(self, header, payload):
        has_payload = bool(header.flags & FLAG_HAS_IQ_PAYLOAD)
        nsamps = len(payload) // 4 if payload else 0
        file_offset = self._core_start + self._core_samples
        if has_payload and payload:
            arr = np.frombuffer(payload, dtype=np.int16)
            arr = arr[: arr.size - (arr.size % 2)]
            nsamps = int(arr.size // 2)
            if arr.size > 0:
                self._append_core_int16(arr)
            self._payload_blocks_rx += 1
            self._samples_rx += nsamps
            self._chunk_payload_blocks += 1
            self._chunk_samples += nsamps
            self._rx_time_tags.append(self._tag_from_header(header, file_offset, nsamps, True))
            return

        if has_payload:
            self._incomplete_blocks += 1
        else:
            self._metadata_only_blocks += 1
        self._rx_time_tags.append(self._tag_from_header(header, file_offset, nsamps, False))

    def _flush_block(self, block):
        header = block.header
        complete = block.is_complete
        has_payload = bool(header.flags & FLAG_HAS_IQ_PAYLOAD)
        payload = block.ordered_payload() if complete else b""
        nsamps = len(payload) // 4 if payload else 0
        file_offset = self._core_start + self._core_samples
        if has_payload and complete and payload:
            arr = np.frombuffer(payload, dtype=np.int16)
            arr = arr[: arr.size - (arr.size % 2)]
            if arr.size > 0:
                self._append_core_int16(arr)
            self._payload_blocks_rx += 1
            self._samples_rx += int(nsamps)
            self._chunk_payload_blocks += 1
            self._chunk_samples += int(nsamps)
            self._rx_time_tags.append(self._tag_from_header(header, file_offset, nsamps, True))
            return

        if has_payload:
            self._incomplete_blocks += 1
        else:
            self._metadata_only_blocks += 1
        self._rx_time_tags.append(self._tag_from_header(header, file_offset, nsamps, False))

    def _tag_from_header(self, header, file_offset, nsamps, complete):
        return RxTimeTag(
            block_seq=int(header.block_seq),
            file_sample_offset=int(file_offset),
            block_sample_offset=int(header.block_sample_offset),
            nsamps=int(nsamps),
            rx_time_sec=float(header.rx_time_sec),
            sample_rate=float(header.sample_rate),
            center_freq=float(header.center_freq),
            gain=float(header.gain),
            payload_format=header.payload_format_name,
            error_code=int(header.error_code),
            overflow_count_total=int(header.overflow_count_total),
            gap_samples_est=int(header.gap_samples_est),
            flags=int(header.flags),
            complete=bool(complete),
        )

    def _append_core_int16(self, arr):
        if self.decode_int16:
            iq = np.empty(arr.size // 2, dtype=np.complex64)
            iq.real = arr[0::2]
            iq.imag = arr[1::2]
            iq.real *= np.float32(1.0 / 32768.0)
            iq.imag *= np.float32(1.0 / 32768.0)
            self._core_parts.append(iq)
            self._core_samples += int(iq.size)
            return
        samples = int(arr.size // 2)
        self._ensure_core_buffer(samples)
        start = self._core_samples * 2
        end = start + samples * 2
        self._core_buffer[start:end] = arr[: samples * 2]
        self._core_samples += samples

    def _ensure_core_buffer(self, additional_samples):
        needed_iq = self._core_samples + int(additional_samples)
        needed_i16 = needed_iq * 2
        if self._core_buffer is not None and self._core_buffer.size >= needed_i16:
            return
        block_margin_samples = max(4096, self.max_packet_size // 4)
        capacity_samples = max(
            needed_iq,
            self.chunk_samples + self.overlap_samples + block_margin_samples,
        )
        new_buffer = np.empty(capacity_samples * 2, dtype=np.int16)
        if self._core_buffer is not None and self._core_samples > 0:
            used = self._core_samples * 2
            new_buffer[:used] = self._core_buffer[:used]
        self._core_buffer = new_buffer

    def _consume_core(self):
        if self.decode_int16:
            if not self._core_parts:
                return None
            if len(self._core_parts) == 1:
                core = self._core_parts[0]
            else:
                core = np.concatenate(self._core_parts)
            self._core_parts = []
            self._core_samples = 0
            return core
        if self._core_buffer is None or self._core_samples <= 0:
            return None
        used = self._core_samples * 2
        core = self._core_buffer[:used].copy()
        self._core_samples = 0
        return core

    def _concat_tail(self, core):
        if self._tail is None or self._array_sample_count(self._tail) == 0:
            return core
        return np.concatenate((self._tail, core))

    def _take_tail(self, raw_iq):
        keep = min(max(0, self.overlap_samples), self._array_sample_count(raw_iq))
        if keep <= 0:
            return None
        if not self.decode_int16:
            return raw_iq[-keep * 2 :].copy()
        return raw_iq[-keep:].copy()

    def _array_sample_count(self, array):
        if array is None:
            return 0
        if not self.decode_int16:
            return int(array.size // 2)
        return int(array.size)
