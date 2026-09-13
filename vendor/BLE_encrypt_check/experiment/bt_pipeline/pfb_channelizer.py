from dataclasses import dataclass
from time import perf_counter as _perf_counter

import numpy as np
from scipy.signal import firwin, lfilter

from . import cuda_backend


def _add_optional_timing(timings, key, seconds):
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + seconds


def _add_batch_or_per_target_timing(timings, per_target_timings, key, seconds, count):
    if per_target_timings is not None and count:
        per_target_seconds = seconds / max(1, int(count))
        for target_timing in per_target_timings:
            _add_optional_timing(target_timing, key, per_target_seconds)
    else:
        _add_optional_timing(timings, key, seconds)


@dataclass
class HostSegmentBatch:
    """Compact host representation for CUDA threshold segments.

    buffers hold merged copy-back arrays. Descriptor arrays identify each segment
    as a view into one buffer without materializing per-segment Python tuples.
    """

    buffers: list
    buffer_indices: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    segment_indices: np.ndarray
    core_starts: np.ndarray
    core_ends: np.ndarray

    def __len__(self):
        return int(self.lengths.size)

    @property
    def descriptor_count(self):
        return len(self)

    @property
    def buffer_count(self):
        return len(self.buffers)

    def segment_view(self, index):
        buffer_index = int(self.buffer_indices[index])
        offset = int(self.offsets[index])
        length = int(self.lengths[index])
        return self.buffers[buffer_index][offset : offset + length]

    def segment_tuple(self, index):
        return (
            int(self.starts[index]),
            int(self.ends[index]),
            self.segment_view(index),
            int(self.segment_indices[index]),
            int(self.core_starts[index]),
            int(self.core_ends[index]),
        )

    def to_segments(self):
        return [self.segment_tuple(index) for index in range(len(self))]

    def __iter__(self):
        return iter(self.to_segments())

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = np.arange(len(self), dtype=np.int64)[item]
            return HostSegmentBatch.from_segments(
                [self.segment_tuple(int(index)) for index in indices]
            )
        return self.segment_tuple(item)

    @classmethod
    def empty(cls):
        empty = np.empty(0, dtype=np.int64)
        return cls([], empty, empty, empty, empty, empty, empty, empty, empty)

    @classmethod
    def from_segments(cls, segments):
        buffers = []
        buffer_indices = []
        starts = []
        ends = []
        offsets = []
        lengths = []
        segment_indices = []
        core_starts = []
        core_ends = []
        for fallback_index, segment_item in enumerate(segments):
            if len(segment_item) >= 4:
                start, end, segment, segment_index = segment_item[:4]
            else:
                start, end, segment = segment_item
                segment_index = fallback_index
            if len(segment_item) >= 6:
                core_start, core_end = segment_item[4:6]
            else:
                core_start, core_end = start, end
            segment_array = np.ascontiguousarray(segment, dtype=np.complex64)
            buffers.append(segment_array)
            buffer_indices.append(len(buffers) - 1)
            starts.append(int(start))
            ends.append(int(end))
            offsets.append(0)
            lengths.append(int(segment_array.size))
            segment_indices.append(int(segment_index))
            core_starts.append(int(core_start))
            core_ends.append(int(core_end))
        return cls(
            buffers,
            np.asarray(buffer_indices, dtype=np.int64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(lengths, dtype=np.int64),
            np.asarray(segment_indices, dtype=np.int64),
            np.asarray(core_starts, dtype=np.int64),
            np.asarray(core_ends, dtype=np.int64),
        )


def _iq_to_cuda_complex64(
    iq,
    cp,
    raw_sample_start=None,
    overlap_cache=None,
    cache_tail_samples=0,
    timing_context=None,
):
    host = np.asarray(iq)
    if host.dtype == np.int16:
        host = host[: host.size - (host.size % 2)]
        num_samples = int(host.size // 2)
        iq_gpu = cp.empty(num_samples, dtype=cp.complex64)
        prefix_samples = 0

        if (
            overlap_cache is not None
            and raw_sample_start is not None
            and overlap_cache.get("start") == raw_sample_start
        ):
            cached = overlap_cache.get("iq_gpu")
            if cached is not None:
                prefix_samples = min(int(cached.size), num_samples)
                if prefix_samples:
                    iq_gpu[:prefix_samples] = cached[:prefix_samples]
                    _add_optional_timing(
                        timing_context, "overlap_reuse_samples", prefix_samples
                    )

        decode_samples = num_samples - prefix_samples
        if decode_samples:
            raw_start = prefix_samples * 2
            raw_gpu = cp.ascontiguousarray(
                cuda_backend.to_device(host[raw_start:], dtype=cp.int16)
            )
            threads = 256
            blocks = (decode_samples + threads - 1) // threads
            _get_int16_iq_decode_kernel()(
                (blocks,),
                (threads,),
                (raw_gpu, iq_gpu, np.int64(decode_samples), np.int64(prefix_samples)),
            )

        if overlap_cache is not None and raw_sample_start is not None:
            tail_samples = min(max(0, int(cache_tail_samples)), num_samples)
            if tail_samples:
                overlap_cache["start"] = raw_sample_start + num_samples - tail_samples
                overlap_cache["iq_gpu"] = iq_gpu[-tail_samples:].copy()
                overlap_cache["samples"] = tail_samples
        return iq_gpu
    return cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))


def design_oversampled_pfb(sample_rate, subband_sample_rate, numtaps, cutoff_hz, oversample=2):
    decim = _integer_ratio(sample_rate, subband_sample_rate)
    num_channels = decim * int(oversample)
    if oversample != 2:
        raise ValueError("Only 2x oversampled PFB channelization is supported.")
    if numtaps < num_channels:
        raise ValueError("--pfb-numtaps must be at least the number of PFB channels.")
    if not 0 < cutoff_hz < subband_sample_rate / 2:
        raise ValueError("--pfb-cutoff must be between 0 and half the subband sample rate.")
    return firwin(numtaps, cutoff_hz, fs=sample_rate), decim, num_channels


def channelize_oversampled_pfb(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    use_cuda=False,
    device_id=None,
    return_host=False,
    cuda_pfb_backend="cupyx",
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    if use_cuda:
        return _channelize_oversampled_pfb_cuda(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            cuda_pfb_backend,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )

    num_channels = decim * 2
    num_outputs = (iq.size + decim - 1) // decim
    channels = np.empty((num_channels, num_outputs), dtype=np.complex64)
    bin_indices = np.arange(num_channels, dtype=np.float64)

    for output_parity, alignment in enumerate((0, decim)):
        output_indices = np.arange(output_parity, num_outputs, 2)
        num_aligned_outputs = output_indices.size
        polyphase_outputs = np.zeros((num_channels, num_aligned_outputs), dtype=np.complex64)
        aligned_indices = np.arange(num_aligned_outputs)

        for phase in range(num_channels):
            phase_taps = prototype[phase::num_channels]
            source = iq[(alignment - phase) % num_channels :: num_channels]
            filtered = lfilter(phase_taps, 1.0, source).astype(np.complex64, copy=False)
            source_indices = aligned_indices + (alignment - phase) // num_channels
            valid = (source_indices >= 0) & (source_indices < filtered.size)
            polyphase_outputs[phase, valid] = filtered[source_indices[valid]]

        aligned_channels = np.fft.ifft(polyphase_outputs, axis=0) * num_channels
        alignment_phase = np.exp(-1j * 2 * np.pi * bin_indices * alignment / num_channels)
        channels[:, output_indices] = (aligned_channels * alignment_phase[:, None]).astype(
            np.complex64, copy=False
        )

    signed_bins = _signed_bin_indices(num_channels)
    chunk_phase = np.exp(
        -1j * 2 * np.pi * signed_bins * (sample_rate / num_channels) * chunk_start_index / sample_rate
    )
    channels *= chunk_phase[:, None].astype(np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    cuda_pfb_backend,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    if cuda_pfb_backend == "cupyx":
        return _channelize_oversampled_pfb_cuda_cupyx(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel":
        return _channelize_oversampled_pfb_cuda_kernel(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel_multi":
        return _channelize_oversampled_pfb_cuda_kernel_multi(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel_multi_float":
        return _channelize_oversampled_pfb_cuda_kernel_multi_float(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel_multi_float_phase":
        return _channelize_oversampled_pfb_cuda_kernel_multi_float_phase(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel_multi_float_phase_t":
        return _channelize_oversampled_pfb_cuda_kernel_multi_float_phase_t(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel_phase":
        return _channelize_oversampled_pfb_cuda_kernel_phase(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel_precomp":
        return _channelize_oversampled_pfb_cuda_kernel_precomp(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    if cuda_pfb_backend == "kernel_const":
        return _channelize_oversampled_pfb_cuda_kernel_const(
            iq,
            chunk_start_index,
            sample_rate,
            decim,
            prototype,
            device_id,
            return_host,
            timing_context,
            cuda_iq_overlap_cache,
            cuda_iq_overlap_cache_samples,
        )
    raise ValueError(
        f"Unknown CUDA PFB backend: {cuda_pfb_backend}. "
        "Expected 'cupyx', 'kernel', 'kernel_multi', 'kernel_multi_float', "
        "'kernel_multi_float_phase', 'kernel_multi_float_phase_t', "
        "'kernel_phase', 'kernel_precomp', "
        "or 'kernel_const'."
    )


def _channelize_oversampled_pfb_cuda_cupyx(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    gpu_lfilter = _get_gpu_lfilter()

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        iq_gpu = _iq_to_cuda_complex64(
            iq,
            cp,
            raw_sample_start=chunk_start_index,
            overlap_cache=cuda_iq_overlap_cache,
            cache_tail_samples=cuda_iq_overlap_cache_samples,
            timing_context=timing_context,
        )
        prototype_gpu = cuda_backend.to_device(prototype)
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_gpu.size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        bin_indices = cp.arange(num_channels, dtype=cp.float64)
        lfilter_den = cp.asarray([1.0], dtype=prototype_gpu.dtype)

        for output_parity, alignment in enumerate((0, decim)):
            output_indices = cp.arange(output_parity, num_outputs, 2)
            num_aligned_outputs = int(output_indices.size)
            polyphase_outputs = cp.zeros(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            aligned_indices = cp.arange(num_aligned_outputs)

            for phase in range(num_channels):
                phase_taps = prototype_gpu[phase::num_channels]
                source = iq_gpu[(alignment - phase) % num_channels :: num_channels]
                filtered = gpu_lfilter(phase_taps, lfilter_den, source).astype(
                    cp.complex64, copy=False
                )
                source_indices = aligned_indices + (alignment - phase) // num_channels
                valid = (source_indices >= 0) & (source_indices < filtered.size)
                polyphase_outputs[phase, valid] = filtered[source_indices[valid]]

            aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
            alignment_phase = cp.exp(-1j * 2 * cp.pi * bin_indices * alignment / num_channels)
            channels[:, output_indices] = (
                aligned_channels * alignment_phase[:, None]
            ).astype(cp.complex64, copy=False)

        signed_bins = _signed_bin_indices(num_channels, cp)
        chunk_phase = cp.exp(
            -1j
            * 2
            * cp.pi
            * signed_bins
            * (sample_rate / num_channels)
            * chunk_start_index
            / sample_rate
        )
        channels *= chunk_phase[:, None].astype(cp.complex64)

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    iq_host = np.asarray(iq)
    use_int16_direct = iq_host.dtype == np.int16
    kernel = (
        _get_pfb_polyphase_i16_kernel()
        if use_int16_direct
        else _get_pfb_polyphase_kernel()
    )

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        if use_int16_direct:
            iq_host = iq_host[: iq_host.size - (iq_host.size % 2)]
            iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq_host, dtype=cp.int16))
            iq_size = int(iq_gpu.size // 2)
        else:
            iq_gpu = _iq_to_cuda_complex64(
                iq,
                cp,
                raw_sample_start=chunk_start_index,
                overlap_cache=cuda_iq_overlap_cache,
                cache_tail_samples=cuda_iq_overlap_cache_samples,
                timing_context=timing_context,
            )
            iq_size = int(iq_gpu.size)
        prototype_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(prototype, dtype=cp.float64)
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        bin_indices = cp.arange(num_channels, dtype=cp.float64)

        for output_parity, alignment in enumerate((0, decim)):
            output_indices = cp.arange(output_parity, num_outputs, 2)
            num_aligned_outputs = int(output_indices.size)
            polyphase_outputs = cp.empty(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            if num_aligned_outputs:
                threads = 256
                blocks_x = (num_aligned_outputs + threads - 1) // threads
                kernel(
                    (blocks_x, num_channels),
                    (threads,),
                    (
                        iq_gpu,
                        prototype_gpu,
                        polyphase_outputs,
                        np.int64(iq_size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_gpu.size),
                    ),
                )

            aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
            alignment_phase = cp.exp(-1j * 2 * cp.pi * bin_indices * alignment / num_channels)
            channels[:, output_indices] = (
                aligned_channels * alignment_phase[:, None]
            ).astype(cp.complex64, copy=False)

        signed_bins = _signed_bin_indices(num_channels, cp)
        chunk_phase = cp.exp(
            -1j
            * 2
            * cp.pi
            * signed_bins
            * (sample_rate / num_channels)
            * chunk_start_index
            / sample_rate
        )
        channels *= chunk_phase[:, None].astype(cp.complex64)

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel_multi(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    iq_host = np.asarray(iq)
    use_int16_direct = iq_host.dtype == np.int16
    kernel = (
        _get_pfb_polyphase_multi_i16_kernel()
        if use_int16_direct
        else _get_pfb_polyphase_multi_kernel()
    )

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        if use_int16_direct:
            iq_host = iq_host[: iq_host.size - (iq_host.size % 2)]
            iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq_host, dtype=cp.int16))
            iq_size = int(iq_gpu.size // 2)
        else:
            iq_gpu = _iq_to_cuda_complex64(
                iq,
                cp,
                raw_sample_start=chunk_start_index,
                overlap_cache=cuda_iq_overlap_cache,
                cache_tail_samples=cuda_iq_overlap_cache_samples,
                timing_context=timing_context,
            )
            iq_size = int(iq_gpu.size)
        prototype_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(prototype, dtype=cp.float64)
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        bin_indices = cp.arange(num_channels, dtype=cp.float64)
        taps_per_phase = (int(prototype_gpu.size) + num_channels - 1) // num_channels
        threads = 128
        outputs_per_thread = 4
        block_outputs = threads * outputs_per_thread
        shared_samples = block_outputs + taps_per_phase - 1
        shared_mem = shared_samples * 8

        for output_parity, alignment in enumerate((0, decim)):
            output_indices = cp.arange(output_parity, num_outputs, 2)
            num_aligned_outputs = int(output_indices.size)
            polyphase_outputs = cp.empty(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            if num_aligned_outputs:
                blocks_x = (num_aligned_outputs + block_outputs - 1) // block_outputs
                kernel(
                    (blocks_x, num_channels),
                    (threads,),
                    (
                        iq_gpu,
                        prototype_gpu,
                        polyphase_outputs,
                        np.int64(iq_size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_gpu.size),
                        np.int64(taps_per_phase),
                    ),
                    shared_mem=shared_mem,
                )

            aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
            alignment_phase = cp.exp(-1j * 2 * cp.pi * bin_indices * alignment / num_channels)
            channels[:, output_indices] = (
                aligned_channels * alignment_phase[:, None]
            ).astype(cp.complex64, copy=False)

        signed_bins = _signed_bin_indices(num_channels, cp)
        chunk_phase = cp.exp(
            -1j
            * 2
            * cp.pi
            * signed_bins
            * (sample_rate / num_channels)
            * chunk_start_index
            / sample_rate
        )
        channels *= chunk_phase[:, None].astype(cp.complex64)

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel_multi_float(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    iq_host = np.asarray(iq)
    use_int16_direct = iq_host.dtype == np.int16
    kernel = (
        _get_pfb_polyphase_multi_float_i16_kernel()
        if use_int16_direct
        else _get_pfb_polyphase_multi_float_kernel()
    )

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        if use_int16_direct:
            iq_host = iq_host[: iq_host.size - (iq_host.size % 2)]
            iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq_host, dtype=cp.int16))
            iq_size = int(iq_gpu.size // 2)
        else:
            iq_gpu = _iq_to_cuda_complex64(
                iq,
                cp,
                raw_sample_start=chunk_start_index,
                overlap_cache=cuda_iq_overlap_cache,
                cache_tail_samples=cuda_iq_overlap_cache_samples,
                timing_context=timing_context,
            )
            iq_size = int(iq_gpu.size)
        prototype_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(prototype, dtype=cp.float32)
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        bin_indices = cp.arange(num_channels, dtype=cp.float64)
        taps_per_phase = (int(prototype_gpu.size) + num_channels - 1) // num_channels
        threads = 128
        outputs_per_thread = 4
        block_outputs = threads * outputs_per_thread
        shared_samples = block_outputs + taps_per_phase - 1
        shared_mem = shared_samples * 8

        for output_parity, alignment in enumerate((0, decim)):
            output_indices = cp.arange(output_parity, num_outputs, 2)
            num_aligned_outputs = int(output_indices.size)
            polyphase_outputs = cp.empty(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            if num_aligned_outputs:
                blocks_x = (num_aligned_outputs + block_outputs - 1) // block_outputs
                kernel(
                    (blocks_x, num_channels),
                    (threads,),
                    (
                        iq_gpu,
                        prototype_gpu,
                        polyphase_outputs,
                        np.int64(iq_size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_gpu.size),
                        np.int64(taps_per_phase),
                    ),
                    shared_mem=shared_mem,
                )

            aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
            alignment_phase = cp.exp(-1j * 2 * cp.pi * bin_indices * alignment / num_channels)
            channels[:, output_indices] = (
                aligned_channels * alignment_phase[:, None]
            ).astype(cp.complex64, copy=False)

        signed_bins = _signed_bin_indices(num_channels, cp)
        chunk_phase = cp.exp(
            -1j
            * 2
            * cp.pi
            * signed_bins
            * (sample_rate / num_channels)
            * chunk_start_index
            / sample_rate
        )
        channels *= chunk_phase[:, None].astype(cp.complex64)

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel_multi_float_phase(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    iq_host = np.asarray(iq)
    use_int16_direct = iq_host.dtype == np.int16
    polyphase_kernel = (
        _get_pfb_polyphase_multi_float_i16_kernel()
        if use_int16_direct
        else _get_pfb_polyphase_multi_float_kernel()
    )
    phase_kernel = _get_pfb_phase_apply_kernel()

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        if use_int16_direct:
            iq_host = iq_host[: iq_host.size - (iq_host.size % 2)]
            iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq_host, dtype=cp.int16))
            iq_size = int(iq_gpu.size // 2)
        else:
            iq_gpu = _iq_to_cuda_complex64(
                iq,
                cp,
                raw_sample_start=chunk_start_index,
                overlap_cache=cuda_iq_overlap_cache,
                cache_tail_samples=cuda_iq_overlap_cache_samples,
                timing_context=timing_context,
            )
            iq_size = int(iq_gpu.size)
        prototype_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(prototype, dtype=cp.float32)
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        taps_per_phase = (int(prototype_gpu.size) + num_channels - 1) // num_channels
        polyphase_threads = 128
        outputs_per_thread = 4
        block_outputs = polyphase_threads * outputs_per_thread
        shared_samples = block_outputs + taps_per_phase - 1
        shared_mem = shared_samples * 8
        phase_threads = 256
        chunk_mod = int(chunk_start_index % num_channels)
        phase_table = _get_pfb_combined_phase_table(num_channels, decim, device_id)

        for output_parity, alignment in enumerate((0, decim)):
            num_aligned_outputs = (num_outputs + 1 - output_parity) // 2
            polyphase_outputs = cp.empty(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            if num_aligned_outputs:
                polyphase_blocks_x = (
                    num_aligned_outputs + block_outputs - 1
                ) // block_outputs
                polyphase_kernel(
                    (polyphase_blocks_x, num_channels),
                    (polyphase_threads,),
                    (
                        iq_gpu,
                        prototype_gpu,
                        polyphase_outputs,
                        np.int64(iq_size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_gpu.size),
                        np.int64(taps_per_phase),
                    ),
                    shared_mem=shared_mem,
                )

                aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
                phase_blocks_x = (num_aligned_outputs + phase_threads - 1) // phase_threads
                phase_kernel(
                    (phase_blocks_x, num_channels),
                    (phase_threads,),
                    (
                        aligned_channels,
                        phase_table[output_parity, chunk_mod],
                        channels,
                        np.int64(num_channels),
                        np.int64(num_outputs),
                        np.int64(num_aligned_outputs),
                        np.int64(output_parity),
                    ),
                )

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel_multi_float_phase_t(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    iq_host = np.asarray(iq)
    use_int16_direct = iq_host.dtype == np.int16
    polyphase_kernel = (
        _get_pfb_polyphase_multi_float_i16_transposed_kernel()
        if use_int16_direct
        else _get_pfb_polyphase_multi_float_transposed_kernel()
    )
    phase_kernel = _get_pfb_phase_apply_transposed_kernel()

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        if use_int16_direct:
            iq_host = iq_host[: iq_host.size - (iq_host.size % 2)]
            iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq_host, dtype=cp.int16))
            iq_size = int(iq_gpu.size // 2)
        else:
            iq_gpu = _iq_to_cuda_complex64(
                iq,
                cp,
                raw_sample_start=chunk_start_index,
                overlap_cache=cuda_iq_overlap_cache,
                cache_tail_samples=cuda_iq_overlap_cache_samples,
                timing_context=timing_context,
            )
            iq_size = int(iq_gpu.size)
        prototype_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(prototype, dtype=cp.float32)
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        taps_per_phase = (int(prototype_gpu.size) + num_channels - 1) // num_channels
        polyphase_threads = 128
        outputs_per_thread = 4
        block_outputs = polyphase_threads * outputs_per_thread
        shared_samples = block_outputs + taps_per_phase - 1
        shared_mem = shared_samples * 8
        phase_threads = 256
        chunk_mod = int(chunk_start_index % num_channels)
        phase_table = _get_pfb_combined_phase_table(num_channels, decim, device_id)

        for output_parity, alignment in enumerate((0, decim)):
            num_aligned_outputs = (num_outputs + 1 - output_parity) // 2
            polyphase_outputs = cp.empty(
                (num_aligned_outputs, num_channels), dtype=cp.complex64
            )
            if num_aligned_outputs:
                polyphase_blocks_x = (
                    num_aligned_outputs + block_outputs - 1
                ) // block_outputs
                polyphase_kernel(
                    (polyphase_blocks_x, num_channels),
                    (polyphase_threads,),
                    (
                        iq_gpu,
                        prototype_gpu,
                        polyphase_outputs,
                        np.int64(iq_size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_gpu.size),
                        np.int64(taps_per_phase),
                    ),
                    shared_mem=shared_mem,
                )

                aligned_channels = cp.fft.ifft(polyphase_outputs, axis=1) * num_channels
                phase_blocks_x = (num_aligned_outputs + phase_threads - 1) // phase_threads
                phase_kernel(
                    (phase_blocks_x, num_channels),
                    (phase_threads,),
                    (
                        aligned_channels,
                        phase_table[output_parity, chunk_mod],
                        channels,
                        np.int64(num_channels),
                        np.int64(num_outputs),
                        np.int64(num_aligned_outputs),
                        np.int64(output_parity),
                    ),
                )

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel_phase(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    polyphase_kernel = _get_pfb_polyphase_kernel()
    phase_kernel = _get_pfb_phase_apply_kernel()

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        iq_gpu = _iq_to_cuda_complex64(
            iq,
            cp,
            raw_sample_start=chunk_start_index,
            overlap_cache=cuda_iq_overlap_cache,
            cache_tail_samples=cuda_iq_overlap_cache_samples,
            timing_context=timing_context,
        )
        prototype_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(prototype, dtype=cp.float64)
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_gpu.size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        chunk_mod = int(chunk_start_index % num_channels)
        phase_table = _get_pfb_combined_phase_table(num_channels, decim, device_id)

        for output_parity, alignment in enumerate((0, decim)):
            output_indices = cp.arange(output_parity, num_outputs, 2)
            num_aligned_outputs = int(output_indices.size)
            polyphase_outputs = cp.empty(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            if num_aligned_outputs:
                threads = 256
                blocks_x = (num_aligned_outputs + threads - 1) // threads
                polyphase_kernel(
                    (blocks_x, num_channels),
                    (threads,),
                    (
                        iq_gpu,
                        prototype_gpu,
                        polyphase_outputs,
                        np.int64(iq_gpu.size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_gpu.size),
                    ),
                )

                aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
                phase_kernel(
                    (blocks_x, num_channels),
                    (threads,),
                    (
                        aligned_channels,
                        phase_table[output_parity, chunk_mod],
                        channels,
                        np.int64(num_channels),
                        np.int64(num_outputs),
                        np.int64(num_aligned_outputs),
                        np.int64(output_parity),
                    ),
                )

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel_precomp(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    kernel = _get_pfb_polyphase_kernel()

    with cuda_backend.use_device(device_id):
        stage_start = _perf_counter()
        iq_gpu = _iq_to_cuda_complex64(
            iq,
            cp,
            raw_sample_start=chunk_start_index,
            overlap_cache=cuda_iq_overlap_cache,
            cache_tail_samples=cuda_iq_overlap_cache_samples,
            timing_context=timing_context,
        )
        prototype_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(prototype, dtype=cp.float64)
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_gpu.size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        alignment_phase_table, chunk_phase_table = _get_pfb_phase_tables(
            num_channels, decim, device_id
        )

        for output_parity, alignment in enumerate((0, decim)):
            output_indices = cp.arange(output_parity, num_outputs, 2)
            num_aligned_outputs = int(output_indices.size)
            polyphase_outputs = cp.empty(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            if num_aligned_outputs:
                threads = 256
                blocks_x = (num_aligned_outputs + threads - 1) // threads
                kernel(
                    (blocks_x, num_channels),
                    (threads,),
                    (
                        iq_gpu,
                        prototype_gpu,
                        polyphase_outputs,
                        np.int64(iq_gpu.size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_gpu.size),
                    ),
                )

            aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
            channels[:, output_indices] = (
                aligned_channels * alignment_phase_table[output_parity][:, None]
            ).astype(cp.complex64, copy=False)

        chunk_phase = chunk_phase_table[int(chunk_start_index % num_channels)]
        channels *= chunk_phase[:, None]

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def _channelize_oversampled_pfb_cuda_kernel_const(
    iq,
    chunk_start_index,
    sample_rate,
    decim,
    prototype,
    device_id,
    return_host,
    timing_context=None,
    cuda_iq_overlap_cache=None,
    cuda_iq_overlap_cache_samples=0,
):
    cp = cuda_backend.require_cuda()
    prototype_host = np.ascontiguousarray(prototype, dtype=np.float64)
    if prototype_host.size > _MAX_PFB_CONST_TAPS:
        raise ValueError(
            f"PFB prototype has {prototype_host.size} taps, which exceeds the "
            f"kernel_const limit of {_MAX_PFB_CONST_TAPS}."
        )

    kernel = _get_pfb_polyphase_const_kernel()

    with cuda_backend.use_device(device_id):
        _load_pfb_const_taps(prototype_host)
        stage_start = _perf_counter()
        iq_gpu = _iq_to_cuda_complex64(
            iq,
            cp,
            raw_sample_start=chunk_start_index,
            overlap_cache=cuda_iq_overlap_cache,
            cache_tail_samples=cuda_iq_overlap_cache_samples,
            timing_context=timing_context,
        )
        _add_optional_timing(timing_context, "host_to_gpu_s", _perf_counter() - stage_start)
        num_channels = decim * 2
        num_outputs = (iq_gpu.size + decim - 1) // decim
        channels = cp.empty((num_channels, num_outputs), dtype=cp.complex64)
        bin_indices = cp.arange(num_channels, dtype=cp.float64)

        for output_parity, alignment in enumerate((0, decim)):
            output_indices = cp.arange(output_parity, num_outputs, 2)
            num_aligned_outputs = int(output_indices.size)
            polyphase_outputs = cp.empty(
                (num_channels, num_aligned_outputs), dtype=cp.complex64
            )
            if num_aligned_outputs:
                threads = 256
                blocks_x = (num_aligned_outputs + threads - 1) // threads
                kernel(
                    (blocks_x, num_channels),
                    (threads,),
                    (
                        iq_gpu,
                        polyphase_outputs,
                        np.int64(iq_gpu.size),
                        np.int64(num_channels),
                        np.int64(alignment),
                        np.int64(num_aligned_outputs),
                        np.int64(prototype_host.size),
                    ),
                )

            aligned_channels = cp.fft.ifft(polyphase_outputs, axis=0) * num_channels
            alignment_phase = cp.exp(-1j * 2 * cp.pi * bin_indices * alignment / num_channels)
            channels[:, output_indices] = (
                aligned_channels * alignment_phase[:, None]
            ).astype(cp.complex64, copy=False)

        signed_bins = _signed_bin_indices(num_channels, cp)
        chunk_phase = cp.exp(
            -1j
            * 2
            * cp.pi
            * signed_bins
            * (sample_rate / num_channels)
            * chunk_start_index
            / sample_rate
        )
        channels *= chunk_phase[:, None].astype(cp.complex64)

    if return_host:
        return cuda_backend.to_host(channels, dtype=np.complex64)
    return channels


def select_coarse_subband(channel_bank, target_freq_mhz, center_freq_hz, sample_rate):
    num_channels = channel_bank.shape[0]
    bin_spacing_hz = sample_rate / num_channels
    target_offset_hz = target_freq_mhz * 1e6 - center_freq_hz
    signed_bin = int(np.floor(target_offset_hz / bin_spacing_hz + 0.5))
    min_bin = -(num_channels // 2)
    max_bin = num_channels // 2
    if not min_bin <= signed_bin <= max_bin:
        raise ValueError(
            f"Target {target_freq_mhz:.3f} MHz is outside the PFB bin range around "
            f"{center_freq_hz / 1e6:.3f} MHz."
        )
    coarse_offset_hz = signed_bin * bin_spacing_hz
    residual_hz = target_offset_hz - coarse_offset_hz
    return channel_bank[signed_bin % num_channels], coarse_offset_hz, residual_hz


def shift_within_subband(
    iq,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz,
    use_cuda=False,
    device_id=None,
    return_host=False,
):
    if use_cuda or cuda_backend.is_device_array(iq):
        return _shift_within_subband_cuda(
            iq,
            chunk_start_index,
            decim,
            sample_rate,
            residual_hz,
            device_id,
            return_host,
        )

    if abs(residual_hz) < 1e-9:
        return iq
    wideband_indices = chunk_start_index + np.arange(iq.size, dtype=np.float64) * decim
    mixer = np.exp(-1j * 2 * np.pi * residual_hz * wideband_indices / sample_rate)
    return (iq * mixer).astype(np.complex64, copy=False)


def shift_within_subband_batch(
    iq,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz_values,
    use_cuda=False,
    device_id=None,
    return_host=False,
):
    if use_cuda or cuda_backend.is_device_array(iq):
        return _shift_within_subband_batch_cuda(
            iq,
            chunk_start_index,
            decim,
            sample_rate,
            residual_hz_values,
            device_id,
            return_host,
        )

    residual_hz_values = np.asarray(residual_hz_values, dtype=np.float64)
    if residual_hz_values.size == 0:
        return np.empty((0, iq.size), dtype=np.complex64)
    wideband_indices = chunk_start_index + np.arange(iq.size, dtype=np.float64) * decim
    mixer = np.exp(
        -1j
        * 2
        * np.pi
        * residual_hz_values[:, None]
        * wideband_indices[None, :]
        / sample_rate
    )
    return (np.asarray(iq, dtype=np.complex64)[None, :] * mixer).astype(
        np.complex64, copy=False
    )


def shift_cleanup_threshold_segments(
    iq,
    cleanup_lpf,
    threshold,
    min_len,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz,
    use_cuda=False,
    device_id=None,
    timing_context=None,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    known_candidate_protocol=None,
    known_ble_aas=None,
    known_bredr_lap_sync_words=None,
    segment_return_mode="segments",
    target_dsp_materialization="full",
):
    if not (use_cuda or cuda_backend.is_device_array(iq)):
        shifted = shift_within_subband(
            iq,
            chunk_start_index,
            decim,
            sample_rate,
            residual_hz,
        )
        cleaned = apply_cleanup_lpf(shifted, cleanup_lpf)
        return threshold_segments_with_iq(cleaned, threshold, min_len)
    return _shift_cleanup_threshold_segments_cuda(
        iq,
        cleanup_lpf,
        threshold,
        min_len,
        chunk_start_index,
        decim,
        sample_rate,
        residual_hz,
        device_id,
        timing_context,
        segment_copy_merge_gap_samples,
        segment_copy_max_merged_samples,
        known_candidate_protocol,
        known_ble_aas,
        known_bredr_lap_sync_words,
        segment_return_mode,
        target_dsp_materialization,
    )


def shift_cleanup_threshold_segments_batch(
    iq,
    cleanup_lpf,
    threshold,
    min_len,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz_values,
    use_cuda=False,
    device_id=None,
    timing_context=None,
    candidate_detector="fixed",
    envelope_window_samples=8,
    start_noise_multiplier=3.5,
    hold_noise_multiplier=1.8,
    candidate_gap_tolerance_samples=0,
    candidate_prepad_samples=0,
    candidate_postpad_samples=0,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    known_candidate_protocol=None,
    known_ble_aas=None,
    known_bredr_lap_sync_words=None,
    per_target_timing_contexts=None,
    segment_return_mode="segments",
    target_dsp_materialization="full",
):
    if not residual_hz_values:
        return []
    if not (use_cuda or cuda_backend.is_device_array(iq)):
        if candidate_detector != "fixed":
            raise NotImplementedError("adaptive hysteresis candidate detection currently requires CUDA")
        return [
            shift_cleanup_threshold_segments(
                iq,
                cleanup_lpf,
                threshold,
                min_len,
                chunk_start_index,
                decim,
                sample_rate,
                residual_hz,
                segment_copy_merge_gap_samples=segment_copy_merge_gap_samples,
                segment_copy_max_merged_samples=segment_copy_max_merged_samples,
                known_candidate_protocol=known_candidate_protocol,
                known_ble_aas=known_ble_aas,
                known_bredr_lap_sync_words=known_bredr_lap_sync_words,
                segment_return_mode=segment_return_mode,
                target_dsp_materialization=target_dsp_materialization,
            )
            for residual_hz in residual_hz_values
        ]
    return _shift_cleanup_threshold_segments_batch_cuda(
        iq,
        cleanup_lpf,
        threshold,
        min_len,
        chunk_start_index,
        decim,
        sample_rate,
        residual_hz_values,
        device_id,
        timing_context,
        candidate_detector,
        envelope_window_samples,
        start_noise_multiplier,
        hold_noise_multiplier,
        candidate_gap_tolerance_samples,
        candidate_prepad_samples,
        candidate_postpad_samples,
        segment_copy_merge_gap_samples,
        segment_copy_max_merged_samples,
        known_candidate_protocol,
        known_ble_aas,
        known_bredr_lap_sync_words,
        per_target_timing_contexts,
        segment_return_mode,
        target_dsp_materialization,
    )


def _shift_within_subband_cuda(
    iq,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz,
    device_id,
    return_host,
):
    cp = cuda_backend.require_cuda()
    with cuda_backend.use_device(device_id):
        iq_gpu = cuda_backend.to_device(iq, dtype=cp.complex64)
        if abs(residual_hz) < 1e-9:
            shifted = iq_gpu
        else:
            wideband_indices = chunk_start_index + cp.arange(
                iq_gpu.size, dtype=cp.float64
            ) * decim
            mixer = cp.exp(-1j * 2 * cp.pi * residual_hz * wideband_indices / sample_rate)
            shifted = (iq_gpu * mixer).astype(cp.complex64, copy=False)
    if return_host:
        return cuda_backend.to_host(shifted, dtype=np.complex64)
    return shifted


def _shift_within_subband_batch_cuda(
    iq,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz_values,
    device_id,
    return_host,
):
    cp = cuda_backend.require_cuda()
    with cuda_backend.use_device(device_id):
        iq_gpu = cuda_backend.to_device(iq, dtype=cp.complex64)
        residuals_gpu = cuda_backend.to_device(residual_hz_values, dtype=cp.float64)
        if residuals_gpu.size == 0:
            shifted = cp.empty((0, iq_gpu.size), dtype=cp.complex64)
        else:
            wideband_indices = chunk_start_index + cp.arange(
                iq_gpu.size, dtype=cp.float64
            ) * decim
            mixer = cp.exp(
                -1j
                * 2
                * cp.pi
                * residuals_gpu[:, None]
                * wideband_indices[None, :]
                / sample_rate
            )
            shifted = (iq_gpu[None, :] * mixer).astype(cp.complex64, copy=False)
    if return_host:
        return cuda_backend.to_host(shifted, dtype=np.complex64)
    return shifted


def design_cleanup_lpf(subband_sample_rate, cutoff_hz, numtaps):
    if numtaps < 1 or numtaps % 2 == 0:
        raise ValueError("--cleanup-numtaps must be a positive odd integer.")
    if not 0 < cutoff_hz < subband_sample_rate / 2:
        raise ValueError("Cleanup FIR cutoff must be between 0 and half the subband sample rate.")
    return firwin(numtaps, cutoff_hz, fs=subband_sample_rate)


def apply_cleanup_lpf(
    iq,
    cleanup_lpf,
    use_cuda=False,
    device_id=None,
    return_host=False,
    cuda_fir_backend="cupyx",
):
    if use_cuda or cuda_backend.is_device_array(iq):
        return _apply_cleanup_lpf_cuda(
            iq, cleanup_lpf, device_id, return_host, cuda_fir_backend
        )
    return lfilter(cleanup_lpf, 1.0, iq).astype(np.complex64, copy=False)


def threshold_segments_with_iq(
    iq,
    threshold,
    min_len,
    use_cuda=False,
    device_id=None,
    known_candidate_protocol=None,
    known_ble_aas=None,
    known_bredr_lap_sync_words=None,
    sample_rate=4e6,
    timing_context=None,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    segment_return_mode="segments",
):
    if use_cuda or cuda_backend.is_device_array(iq):
        return _threshold_segments_with_iq_cuda(
            iq,
            threshold,
            min_len,
            device_id,
            known_candidate_protocol,
            known_ble_aas,
            known_bredr_lap_sync_words,
            sample_rate,
            timing_context,
            segment_copy_merge_gap_samples,
            segment_copy_max_merged_samples,
            segment_return_mode,
        )

    if len(iq) == 0:
        return []
    amp = np.abs(iq)
    above_th = amp > threshold
    edges = np.diff(above_th.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above_th[0]:
        starts = np.r_[0, starts]
    if above_th[-1]:
        ends = np.r_[ends, len(above_th)]
    return [
        (int(start), int(end), iq[start:end])
        for start, end in zip(starts, ends)
        if end - start >= min_len
    ]


def _threshold_segments_with_iq_cuda(
    iq,
    threshold,
    min_len,
    device_id,
    known_candidate_protocol=None,
    known_ble_aas=None,
    known_bredr_lap_sync_words=None,
    sample_rate=4e6,
    timing_context=None,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    segment_return_mode="segments",
):
    cp = cuda_backend.require_cuda()
    with cuda_backend.use_device(device_id):
        iq_gpu = cuda_backend.to_device(iq, dtype=cp.complex64)
        if iq_gpu.size == 0:
            return []
        above_th = cp.abs(iq_gpu) > threshold
        edges = cp.diff(above_th.astype(cp.int8))
        starts = cp.where(edges == 1)[0] + 1
        ends = cp.where(edges == -1)[0] + 1
        if bool(above_th[0].item()):
            starts = cp.r_[cp.asarray([0], dtype=starts.dtype), starts]
        if bool(above_th[-1].item()):
            ends = cp.r_[ends, cp.asarray([iq_gpu.size], dtype=ends.dtype)]
        lengths = ends - starts
        valid = lengths >= min_len
        starts_valid = starts[valid]
        ends_valid = ends[valid]
        segment_ids = None
        if _has_known_candidate_patterns(
            known_candidate_protocol, known_ble_aas, known_bredr_lap_sync_words
        ):
            segment_ids = cp.arange(starts_valid.size, dtype=cp.int64)
            starts_valid, ends_valid, segment_ids = _filter_cuda_known_candidates(
                cp,
                iq_gpu,
                starts_valid,
                ends_valid,
                segment_ids,
                known_candidate_protocol,
                known_ble_aas,
                known_bredr_lap_sync_words,
                sample_rate,
            )
        stage_start = _perf_counter()
        starts_host = cp.asnumpy(starts_valid).astype(np.int64, copy=False)
        ends_host = cp.asnumpy(ends_valid).astype(np.int64, copy=False)
        segment_ids_host = (
            cp.asnumpy(segment_ids).astype(np.int64, copy=False)
            if segment_ids is not None
            else None
        )
        _add_optional_timing(timing_context, "segment_copy_back_s", _perf_counter() - stage_start)
        segments = _copy_cuda_threshold_segments_merged(
            cp,
            iq_gpu,
            starts_host,
            ends_host,
            segment_ids_host,
            max_gap_samples=segment_copy_merge_gap_samples,
            max_merged_samples=segment_copy_max_merged_samples,
            timing_context=timing_context,
            segment_return_mode=segment_return_mode,
        )
    return segments


def _shift_cleanup_threshold_segments_cuda(
    iq,
    cleanup_lpf,
    threshold,
    min_len,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz,
    device_id,
    timing_context=None,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    known_candidate_protocol=None,
    known_ble_aas=None,
    known_bredr_lap_sync_words=None,
    segment_return_mode="segments",
    target_dsp_materialization="full",
):
    cp = cuda_backend.require_cuda()
    cleanup_lpf_host = np.ascontiguousarray(cleanup_lpf, dtype=np.float32)
    if cleanup_lpf_host.size > _MAX_CLEANUP_CONST_TAPS:
        raise ValueError(
            f"cleanup FIR has {cleanup_lpf_host.size} taps, which exceeds the "
            f"fused kernel limit of {_MAX_CLEANUP_CONST_TAPS}."
        )
    use_known_filter = _has_known_candidate_patterns(
        known_candidate_protocol, known_ble_aas, known_bredr_lap_sync_words
    )
    use_threshold_then_segments = (
        target_dsp_materialization == "threshold_then_segments"
        and not use_known_filter
    )
    use_zero_shift = abs(float(residual_hz)) < 1e-9
    kernel = (
        _get_shift_cleanup_threshold_multi_float_zero_mask_kernel()
        if use_threshold_then_segments and use_zero_shift
        else (
            _get_shift_cleanup_threshold_multi_float_mask_kernel()
            if use_threshold_then_segments
            else (
                _get_shift_cleanup_threshold_multi_float_zero_kernel()
                if use_zero_shift
                else _get_shift_cleanup_threshold_multi_float_kernel()
            )
        )
    )
    with cuda_backend.use_device(device_id):
        _load_shift_cleanup_threshold_multi_float_taps(cleanup_lpf_host)
        iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))
        if iq_gpu.size == 0:
            return []
        cleaned = None if use_threshold_then_segments else cp.empty_like(iq_gpu, dtype=cp.complex64)
        above_th = cp.empty(iq_gpu.size, dtype=cp.int8)
        threads = 256
        outputs_per_thread = _CLEANUP_MULTI_OUTPUTS_PER_THREAD
        blocks = (int(iq_gpu.size) + threads * outputs_per_thread - 1) // (
            threads * outputs_per_thread
        )
        shared_len = threads * outputs_per_thread + int(cleanup_lpf_host.size) - 1
        if use_zero_shift:
            args = (
                iq_gpu,
                above_th,
                np.int64(iq_gpu.size),
                np.int64(cleanup_lpf_host.size),
                np.float32(threshold),
            )
            if not use_threshold_then_segments:
                args = (iq_gpu, cleaned) + args[1:]
            stage_start = _perf_counter()
            kernel((blocks,), (threads,), args, shared_mem=shared_len * 8)
            if use_threshold_then_segments:
                _add_optional_timing(timing_context, "threshold_mask_s", _perf_counter() - stage_start)
        else:
            args = (
                iq_gpu,
                above_th,
                np.int64(iq_gpu.size),
                np.int64(cleanup_lpf_host.size),
                np.float32(threshold),
                np.float64(chunk_start_index),
                np.float64(decim),
                np.float64(sample_rate),
                np.float64(residual_hz),
            )
            if not use_threshold_then_segments:
                args = (iq_gpu, cleaned) + args[1:]
            stage_start = _perf_counter()
            kernel((blocks,), (threads,), args, shared_mem=shared_len * 8)
            if use_threshold_then_segments:
                _add_optional_timing(timing_context, "threshold_mask_s", _perf_counter() - stage_start)
        if use_threshold_then_segments:
            segments = _threshold_then_materialize_segments_from_cuda_mask(
                cp,
                iq_gpu,
                cleanup_lpf_host,
                above_th,
                min_len,
                chunk_start_index,
                decim,
                sample_rate,
                residual_hz,
                timing_context=timing_context,
                segment_copy_merge_gap_samples=segment_copy_merge_gap_samples,
                segment_copy_max_merged_samples=segment_copy_max_merged_samples,
                segment_return_mode=segment_return_mode,
            )
        else:
            segments = _threshold_segments_from_cuda_mask(
                cp,
                cleaned,
                above_th,
                min_len,
                known_candidate_protocol=known_candidate_protocol,
                known_ble_aas=known_ble_aas,
                known_bredr_lap_sync_words=known_bredr_lap_sync_words,
                sample_rate=sample_rate / decim,
                timing_context=timing_context,
                segment_copy_merge_gap_samples=segment_copy_merge_gap_samples,
                segment_copy_max_merged_samples=segment_copy_max_merged_samples,
                segment_return_mode=segment_return_mode,
            )
    return segments


def _shift_cleanup_threshold_segments_batch_cuda(
    iq,
    cleanup_lpf,
    threshold,
    min_len,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz_values,
    device_id,
    timing_context=None,
    candidate_detector="fixed",
    envelope_window_samples=8,
    start_noise_multiplier=3.5,
    hold_noise_multiplier=1.8,
    candidate_gap_tolerance_samples=0,
    candidate_prepad_samples=0,
    candidate_postpad_samples=0,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    known_candidate_protocol=None,
    known_ble_aas=None,
    known_bredr_lap_sync_words=None,
    per_target_timing_contexts=None,
    segment_return_mode="segments",
    target_dsp_materialization="full",
):
    cp = cuda_backend.require_cuda()
    cleanup_lpf_host = np.ascontiguousarray(cleanup_lpf, dtype=np.float32)
    if cleanup_lpf_host.size > _MAX_CLEANUP_CONST_TAPS:
        raise ValueError(
            f"cleanup FIR has {cleanup_lpf_host.size} taps, which exceeds the "
            f"fused kernel limit of {_MAX_CLEANUP_CONST_TAPS}."
        )
    residuals_host = np.ascontiguousarray(residual_hz_values, dtype=np.float64)
    use_known_filter = _has_known_candidate_patterns(
        known_candidate_protocol, known_ble_aas, known_bredr_lap_sync_words
    )
    use_threshold_then_segments = (
        target_dsp_materialization == "threshold_then_segments"
        and not use_known_filter
    )
    if candidate_detector not in ("fixed", "adaptive_hysteresis"):
        raise ValueError(f"unsupported candidate detector: {candidate_detector}")
    if candidate_detector == "adaptive_hysteresis" and use_threshold_then_segments:
        raise ValueError("adaptive hysteresis requires full target DSP materialization")
    if candidate_detector == "adaptive_hysteresis" and use_known_filter:
        raise ValueError("adaptive hysteresis must remain blind and cannot use known-candidate filtering")
    with cuda_backend.use_device(device_id):
        _load_shift_cleanup_threshold_multi_float_taps(cleanup_lpf_host)
        iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))
        residuals_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(residuals_host, dtype=cp.float64)
        )
        target_count = int(residuals_gpu.size)
        if iq_gpu.size == 0 or target_count == 0:
            return [[] for _ in range(target_count)]
        cleaned = None if use_threshold_then_segments else cp.empty((target_count, iq_gpu.size), dtype=cp.complex64)
        above_th = cp.empty((target_count, iq_gpu.size), dtype=cp.int8)
        threads = 256
        outputs_per_thread = _CLEANUP_MULTI_OUTPUTS_PER_THREAD
        blocks_x = (int(iq_gpu.size) + threads * outputs_per_thread - 1) // (
            threads * outputs_per_thread
        )
        shared_len = threads * outputs_per_thread + int(cleanup_lpf_host.size) - 1
        if target_count > 1:
            kernel = (
                _get_shift_cleanup_threshold_multi_float_batch_shared_input_mask_kernel()
                if use_threshold_then_segments
                else _get_shift_cleanup_threshold_multi_float_batch_shared_input_kernel()
            )
            args = (
                iq_gpu,
                residuals_gpu,
                above_th,
                np.int64(iq_gpu.size),
                np.int64(target_count),
                np.int64(cleanup_lpf_host.size),
                np.float32(threshold),
                np.float64(chunk_start_index),
                np.float64(decim),
                np.float64(sample_rate),
            )
            if not use_threshold_then_segments:
                args = (iq_gpu, residuals_gpu, cleaned) + args[2:]
            stage_start = _perf_counter()
            kernel((blocks_x,), (threads,), args, shared_mem=shared_len * 16)
            if use_threshold_then_segments:
                _add_batch_or_per_target_timing(
                    timing_context,
                    per_target_timing_contexts,
                    "threshold_mask_s",
                    _perf_counter() - stage_start,
                    target_count,
                )
        else:
            kernel = (
                _get_shift_cleanup_threshold_multi_float_batch_mask_kernel()
                if use_threshold_then_segments
                else _get_shift_cleanup_threshold_multi_float_batch_kernel()
            )
            args = (
                iq_gpu,
                residuals_gpu,
                above_th,
                np.int64(iq_gpu.size),
                np.int64(cleanup_lpf_host.size),
                np.float32(threshold),
                np.float64(chunk_start_index),
                np.float64(decim),
                np.float64(sample_rate),
            )
            if not use_threshold_then_segments:
                args = (iq_gpu, residuals_gpu, cleaned) + args[2:]
            stage_start = _perf_counter()
            kernel((blocks_x, target_count), (threads,), args, shared_mem=shared_len * 8)
            if use_threshold_then_segments:
                _add_batch_or_per_target_timing(
                    timing_context,
                    per_target_timing_contexts,
                    "threshold_mask_s",
                    _perf_counter() - stage_start,
                    target_count,
                )
        segment_batches = []
        for target_index in range(target_count):
            target_timing = (
                per_target_timing_contexts[target_index]
                if per_target_timing_contexts is not None
                else timing_context
            )
            if use_threshold_then_segments:
                segment_batches.append(
                    _threshold_then_materialize_segments_from_cuda_mask(
                        cp,
                        iq_gpu,
                        cleanup_lpf_host,
                        above_th[target_index],
                        min_len,
                        chunk_start_index,
                        decim,
                        sample_rate,
                        float(residuals_host[target_index]),
                        timing_context=target_timing,
                        segment_copy_merge_gap_samples=segment_copy_merge_gap_samples,
                        segment_copy_max_merged_samples=segment_copy_max_merged_samples,
                        segment_return_mode=segment_return_mode,
                    )
                )
            else:
                if candidate_detector == "adaptive_hysteresis":
                    segments = _adaptive_hysteresis_segments_cuda(
                        cp,
                        cleaned[target_index],
                        min_len,
                        envelope_window_samples=envelope_window_samples,
                        start_noise_multiplier=start_noise_multiplier,
                        hold_noise_multiplier=hold_noise_multiplier,
                        gap_tolerance_samples=candidate_gap_tolerance_samples,
                        prepad_samples=candidate_prepad_samples,
                        postpad_samples=candidate_postpad_samples,
                        timing_context=target_timing,
                        segment_copy_merge_gap_samples=segment_copy_merge_gap_samples,
                        segment_copy_max_merged_samples=segment_copy_max_merged_samples,
                        segment_return_mode=segment_return_mode,
                    )
                else:
                    segments = _threshold_segments_from_cuda_mask(
                        cp,
                        cleaned[target_index],
                        above_th[target_index],
                        min_len,
                        known_candidate_protocol=known_candidate_protocol,
                        known_ble_aas=known_ble_aas,
                        known_bredr_lap_sync_words=known_bredr_lap_sync_words,
                        sample_rate=sample_rate / decim,
                        timing_context=target_timing,
                        segment_copy_merge_gap_samples=segment_copy_merge_gap_samples,
                        segment_copy_max_merged_samples=segment_copy_max_merged_samples,
                        segment_return_mode=segment_return_mode,
                    )
                segment_batches.append(segments)
        return segment_batches


def _adaptive_hysteresis_segments_cuda(
    cp,
    iq_gpu,
    min_len,
    envelope_window_samples=8,
    start_noise_multiplier=3.5,
    hold_noise_multiplier=1.8,
    gap_tolerance_samples=0,
    prepad_samples=0,
    postpad_samples=0,
    timing_context=None,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    segment_return_mode="segments",
):
    if iq_gpu.size == 0:
        return HostSegmentBatch.empty() if segment_return_mode == "compact" else []
    window = int(envelope_window_samples)
    if window < 1:
        raise ValueError("envelope_window_samples must be positive")
    if not 1.0 <= hold_noise_multiplier < start_noise_multiplier:
        raise ValueError("noise multipliers must satisfy 1 <= hold < start")
    if gap_tolerance_samples < 0 or prepad_samples < 0 or postpad_samples < 0:
        raise ValueError("candidate gap tolerance and padding must be non-negative")

    stage_start = _perf_counter()
    power = cp.abs(iq_gpu).astype(cp.float32) ** 2
    if window > 1:
        kernel = cp.full(window, 1.0 / window, dtype=cp.float32)
        envelope = cp.convolve(power, kernel, mode="same")
    else:
        envelope = power
    stride = max(1, int(envelope.size) // 65536)
    noise_power = float(cp.median(envelope[::stride]).item())
    high_threshold = noise_power * float(start_noise_multiplier) ** 2
    low_threshold = noise_power * float(hold_noise_multiplier) ** 2
    low_mask = envelope >= low_threshold
    high_mask = envelope >= high_threshold
    edges = cp.diff(low_mask.astype(cp.int8))
    starts = cp.where(edges == 1)[0] + 1
    ends = cp.where(edges == -1)[0] + 1
    if bool(low_mask[0].item()):
        starts = cp.r_[cp.asarray([0], dtype=starts.dtype), starts]
    if bool(low_mask[-1].item()):
        ends = cp.r_[ends, cp.asarray([iq_gpu.size], dtype=ends.dtype)]
    if gap_tolerance_samples and starts.size > 1:
        starts_host = cp.asnumpy(starts).astype(np.int64, copy=False)
        ends_host = cp.asnumpy(ends).astype(np.int64, copy=False)
        merged_starts = [int(starts_host[0])]
        merged_ends = [int(ends_host[0])]
        for start, end in zip(starts_host[1:], ends_host[1:]):
            if int(start) - merged_ends[-1] <= int(gap_tolerance_samples):
                merged_ends[-1] = int(end)
            else:
                merged_starts.append(int(start))
                merged_ends.append(int(end))
        starts = cp.asarray(merged_starts, dtype=cp.int64)
        ends = cp.asarray(merged_ends, dtype=cp.int64)
    high_prefix = cp.empty(high_mask.size + 1, dtype=cp.int64)
    high_prefix[0] = 0
    cp.cumsum(high_mask, dtype=cp.int64, out=high_prefix[1:])
    seeded = (high_prefix[ends] - high_prefix[starts]) > 0
    long_enough = (ends - starts) >= int(min_len)
    valid = seeded & long_enough
    core_starts = starts[valid]
    core_ends = ends[valid]
    window_starts = cp.maximum(core_starts - int(prepad_samples), 0)
    window_ends = cp.minimum(core_ends + int(postpad_samples), iq_gpu.size)
    starts_host = cp.asnumpy(window_starts).astype(np.int64, copy=False)
    ends_host = cp.asnumpy(window_ends).astype(np.int64, copy=False)
    core_starts_host = cp.asnumpy(core_starts).astype(np.int64, copy=False)
    core_ends_host = cp.asnumpy(core_ends).astype(np.int64, copy=False)
    _add_optional_timing(timing_context, "adaptive_hysteresis_s", _perf_counter() - stage_start)
    _add_optional_timing(timing_context, "adaptive_noise_power", noise_power)
    return _copy_cuda_threshold_segments_merged(
        cp,
        iq_gpu,
        starts_host,
        ends_host,
        core_starts=core_starts_host,
        core_ends=core_ends_host,
        max_gap_samples=segment_copy_merge_gap_samples,
        max_merged_samples=segment_copy_max_merged_samples,
        timing_context=timing_context,
        segment_return_mode=segment_return_mode,
    )


def _threshold_segments_from_cuda_mask(
    cp,
    iq_gpu,
    above_th,
    min_len,
    known_candidate_protocol=None,
    known_ble_aas=None,
    known_bredr_lap_sync_words=None,
    sample_rate=4e6,
    timing_context=None,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    segment_return_mode="segments",
):
    if iq_gpu.size == 0:
        return []
    edges = cp.diff(above_th)
    starts = cp.where(edges == 1)[0] + 1
    ends = cp.where(edges == -1)[0] + 1
    if bool(above_th[0].item()):
        starts = cp.r_[cp.asarray([0], dtype=starts.dtype), starts]
    if bool(above_th[-1].item()):
        ends = cp.r_[ends, cp.asarray([iq_gpu.size], dtype=ends.dtype)]
    lengths = ends - starts
    valid = lengths >= min_len
    starts_valid = starts[valid]
    ends_valid = ends[valid]
    segment_ids = None
    if _has_known_candidate_patterns(
        known_candidate_protocol, known_ble_aas, known_bredr_lap_sync_words
    ):
        segment_ids = cp.arange(starts_valid.size, dtype=cp.int64)
        starts_valid, ends_valid, segment_ids = _filter_cuda_known_candidates(
            cp,
            iq_gpu,
            starts_valid,
            ends_valid,
            segment_ids,
            known_candidate_protocol,
            known_ble_aas,
            known_bredr_lap_sync_words,
            sample_rate,
        )
    stage_start = _perf_counter()
    starts_host = cp.asnumpy(starts_valid).astype(np.int64, copy=False)
    ends_host = cp.asnumpy(ends_valid).astype(np.int64, copy=False)
    segment_ids_host = (
        cp.asnumpy(segment_ids).astype(np.int64, copy=False)
        if segment_ids is not None
        else None
    )
    _add_optional_timing(timing_context, "segment_copy_back_s", _perf_counter() - stage_start)
    return _copy_cuda_threshold_segments_merged(
        cp,
        iq_gpu,
        starts_host,
        ends_host,
        segment_ids_host,
        max_gap_samples=segment_copy_merge_gap_samples,
        max_merged_samples=segment_copy_max_merged_samples,
        timing_context=timing_context,
        segment_return_mode=segment_return_mode,
    )


def _cuda_threshold_starts_ends(cp, above_th, min_len):
    if above_th.size == 0:
        return None, None
    edges = cp.diff(above_th)
    starts = cp.where(edges == 1)[0] + 1
    ends = cp.where(edges == -1)[0] + 1
    if bool(above_th[0].item()):
        starts = cp.r_[cp.asarray([0], dtype=starts.dtype), starts]
    if bool(above_th[-1].item()):
        ends = cp.r_[ends, cp.asarray([above_th.size], dtype=ends.dtype)]
    lengths = ends - starts
    valid = lengths >= min_len
    return starts[valid], ends[valid]


def _merged_cuda_copy_ranges(starts, ends, max_gap_samples=4096, max_merged_samples=262144):
    if starts.size == 0:
        return []
    ranges = []
    group_start_index = 0
    group_start = int(starts[0])
    group_end = int(ends[0])
    for index in range(1, starts.size):
        start = int(starts[index])
        end = int(ends[index])
        gap = start - group_end
        merged_span = end - group_start
        if gap <= max_gap_samples and merged_span <= max_merged_samples:
            group_end = end
            continue
        ranges.append((group_start_index, index, group_start, group_end))
        group_start_index = index
        group_start = start
        group_end = end
    ranges.append((group_start_index, int(starts.size), group_start, group_end))
    return ranges


def _threshold_then_materialize_segments_from_cuda_mask(
    cp,
    iq_gpu,
    cleanup_lpf_host,
    above_th,
    min_len,
    chunk_start_index,
    decim,
    sample_rate,
    residual_hz,
    timing_context=None,
    segment_copy_merge_gap_samples=4096,
    segment_copy_max_merged_samples=262144,
    segment_return_mode="segments",
):
    starts_valid, ends_valid = _cuda_threshold_starts_ends(cp, above_th, min_len)
    if starts_valid is None:
        return HostSegmentBatch.empty() if segment_return_mode == "compact" else []
    stage_start = _perf_counter()
    starts_host = cp.asnumpy(starts_valid).astype(np.int64, copy=False)
    ends_host = cp.asnumpy(ends_valid).astype(np.int64, copy=False)
    _add_optional_timing(timing_context, "segment_copy_back_s", _perf_counter() - stage_start)
    _add_optional_timing(timing_context, "segment_count", int(starts_host.size))
    if starts_host.size == 0:
        return HostSegmentBatch.empty() if segment_return_mode == "compact" else []

    ranges = _merged_cuda_copy_ranges(
        starts_host,
        ends_host,
        max_gap_samples=segment_copy_merge_gap_samples,
        max_merged_samples=segment_copy_max_merged_samples,
    )
    range_starts = np.asarray([copy_start for _a, _b, copy_start, _copy_end in ranges], dtype=np.int64)
    range_lengths = np.asarray([copy_end - copy_start for _a, _b, copy_start, copy_end in ranges], dtype=np.int64)
    output_offsets = np.empty(range_lengths.size, dtype=np.int64)
    cursor = 0
    for index, length in enumerate(range_lengths):
        output_offsets[index] = cursor
        cursor += int(length)
    total_samples = int(cursor)
    if total_samples == 0:
        return HostSegmentBatch.empty() if segment_return_mode == "compact" else []

    stage_start = _perf_counter()
    output_gpu = cp.empty(total_samples, dtype=cp.complex64)
    threads = 256
    blocks = (total_samples + threads - 1) // threads
    _get_shift_cleanup_materialize_ranges_kernel()(
        (blocks,),
        (threads,),
        (
            iq_gpu,
            output_gpu,
            cp.asarray(range_starts, dtype=cp.int64),
            cp.asarray(range_lengths, dtype=cp.int64),
            cp.asarray(output_offsets, dtype=cp.int64),
            np.int64(range_lengths.size),
            np.int64(total_samples),
            np.int64(iq_gpu.size),
            np.int64(cleanup_lpf_host.size),
            np.float64(chunk_start_index),
            np.float64(decim),
            np.float64(sample_rate),
            np.float64(residual_hz),
        ),
    )
    _add_optional_timing(timing_context, "segment_materialize_s", _perf_counter() - stage_start)
    _add_optional_timing(timing_context, "segment_materialize_calls", int(range_lengths.size))
    _add_optional_timing(timing_context, "segment_materialize_samples", total_samples)

    stage_start = _perf_counter()
    materialized_host = cp.asnumpy(output_gpu).astype(np.complex64, copy=False)
    _add_optional_timing(timing_context, "segment_copy_back_s", _perf_counter() - stage_start)
    _add_optional_timing(timing_context, "segment_copy_calls", 1)
    _add_optional_timing(timing_context, "segment_copy_samples", total_samples)

    segments = []
    buffer_indices = []
    descriptor_starts = []
    descriptor_ends = []
    descriptor_offsets = []
    descriptor_lengths = []
    descriptor_segment_indices = []
    for range_index, (group_start_index, end_index, copy_start, _copy_end) in enumerate(ranges):
        base_offset = int(output_offsets[range_index])
        for segment_index in range(group_start_index, end_index):
            start = int(starts_host[segment_index])
            end = int(ends_host[segment_index])
            offset_start = base_offset + start - copy_start
            length = end - start
            if segment_return_mode == "compact":
                buffer_indices.append(0)
                descriptor_starts.append(start)
                descriptor_ends.append(end)
                descriptor_offsets.append(offset_start)
                descriptor_lengths.append(length)
                descriptor_segment_indices.append(segment_index)
            else:
                segments.append(
                    (
                        start,
                        end,
                        materialized_host[offset_start : offset_start + length],
                    )
                )

    if segment_return_mode == "compact":
        _add_optional_timing(timing_context, "segment_buffers", 1)
        _add_optional_timing(timing_context, "segment_descriptor_count", len(descriptor_lengths))
        return HostSegmentBatch(
            [materialized_host],
            np.asarray(buffer_indices, dtype=np.int64),
            np.asarray(descriptor_starts, dtype=np.int64),
            np.asarray(descriptor_ends, dtype=np.int64),
            np.asarray(descriptor_offsets, dtype=np.int64),
            np.asarray(descriptor_lengths, dtype=np.int64),
            np.asarray(descriptor_segment_indices, dtype=np.int64),
            np.asarray(descriptor_starts, dtype=np.int64),
            np.asarray(descriptor_ends, dtype=np.int64),
        )
    return segments


def _filter_cuda_known_candidates(
    cp,
    iq_gpu,
    starts,
    ends,
    segment_ids,
    protocol,
    known_ble_aas,
    known_bredr_lap_sync_words,
    sample_rate,
):
    if starts.size == 0 or not protocol:
        return starts, ends, segment_ids

    protocol_id = 0
    ble_patterns = cp.empty(0, dtype=cp.uint32)
    bredr_patterns = cp.empty(0, dtype=cp.uint64)
    if protocol == "ble":
        patterns = _ble_aa_patterns(known_ble_aas)
        if not patterns:
            return starts, ends, segment_ids
        protocol_id = 1
        ble_patterns = cp.asarray(patterns, dtype=cp.uint32)
    elif protocol == "bredr":
        patterns = _bredr_sync_patterns(known_bredr_lap_sync_words)
        if not patterns:
            return starts, ends, segment_ids
        protocol_id = 2
        bredr_patterns = cp.asarray(patterns, dtype=cp.uint64)
    else:
        return starts, ends, segment_ids

    keep = cp.zeros(starts.size, dtype=cp.int8)
    kernel = _get_known_candidate_filter_kernel()
    threads = 128
    kernel(
        (int(starts.size),),
        (threads,),
        (
            iq_gpu,
            starts,
            ends,
            keep,
            np.int64(starts.size),
            np.int32(protocol_id),
            np.int32(round(sample_rate / 1e6)),
            np.int32(round(sample_rate / 2e6)),
            ble_patterns,
            np.int32(ble_patterns.size),
            bredr_patterns,
            np.int32(bredr_patterns.size),
        ),
    )
    valid = keep != 0
    return starts[valid], ends[valid], segment_ids[valid]


def _has_known_candidate_patterns(protocol, known_ble_aas, known_bredr_lap_sync_words):
    if protocol == "ble":
        return bool(known_ble_aas)
    if protocol == "bredr":
        return bool(known_bredr_lap_sync_words)
    return False


def _ble_aa_patterns(known_ble_aas):
    if not known_ble_aas:
        return []
    patterns = []
    for value in known_ble_aas:
        text = str(value).strip().replace("0x", "").replace("0X", "").upper()
        if not text:
            continue
        text = text.zfill(8)[-8:]
        try:
            patterns.append(int.from_bytes(bytes.fromhex(text), "little"))
        except ValueError:
            continue
    return sorted(set(patterns))


def _bredr_sync_patterns(known_bredr_lap_sync_words):
    if not known_bredr_lap_sync_words:
        return []
    patterns = []
    for value in known_bredr_lap_sync_words:
        try:
            patterns.append(int(value) & ((1 << 64) - 1))
        except (TypeError, ValueError):
            continue
    return sorted(set(patterns))


def _copy_cuda_threshold_segments_merged(
    cp,
    iq_gpu,
    starts,
    ends,
    segment_ids=None,
    core_starts=None,
    core_ends=None,
    max_gap_samples=4096,
    max_merged_samples=262144,
    timing_context=None,
    segment_return_mode="segments",
):
    if segment_return_mode not in ("segments", "compact"):
        raise ValueError("segment_return_mode must be 'segments' or 'compact'.")
    _add_optional_timing(timing_context, "segment_count", int(starts.size))
    if starts.size == 0:
        return HostSegmentBatch.empty() if segment_return_mode == "compact" else []

    segments = []
    buffers = []
    buffer_indices = []
    descriptor_starts = []
    descriptor_ends = []
    descriptor_offsets = []
    descriptor_lengths = []
    descriptor_segment_indices = []
    descriptor_core_starts = []
    descriptor_core_ends = []
    if core_starts is None:
        core_starts = starts
    if core_ends is None:
        core_ends = ends
    group_start_index = 0
    group_start = int(starts[0])
    group_end = int(ends[0])

    def flush_group(end_index, copy_start, copy_end):
        stage_start = _perf_counter()
        merged_host = cp.asnumpy(iq_gpu[copy_start:copy_end]).astype(
            np.complex64, copy=False
        )
        _add_optional_timing(timing_context, "segment_copy_back_s", _perf_counter() - stage_start)
        _add_optional_timing(timing_context, "segment_copy_calls", 1)
        _add_optional_timing(timing_context, "segment_copy_samples", copy_end - copy_start)
        buffer_index = None
        if segment_return_mode == "compact":
            buffer_index = len(buffers)
            buffers.append(merged_host)
            _add_optional_timing(timing_context, "segment_buffers", 1)
        for segment_index in range(group_start_index, end_index):
            start = int(starts[segment_index])
            end = int(ends[segment_index])
            core_start = int(core_starts[segment_index])
            core_end = int(core_ends[segment_index])
            offset_start = start - copy_start
            offset_end = end - copy_start
            output_segment_index = (
                segment_index
                if segment_ids is None
                else int(segment_ids[segment_index])
            )
            if segment_return_mode == "compact":
                buffer_indices.append(buffer_index)
                descriptor_starts.append(start)
                descriptor_ends.append(end)
                descriptor_offsets.append(offset_start)
                descriptor_lengths.append(offset_end - offset_start)
                descriptor_segment_indices.append(output_segment_index)
                descriptor_core_starts.append(core_start)
                descriptor_core_ends.append(core_end)
            else:
                segment = merged_host[offset_start:offset_end]
                if core_start == start and core_end == end:
                    if segment_ids is None:
                        segments.append((start, end, segment))
                    else:
                        segments.append((start, end, segment, output_segment_index))
                elif segment_ids is None:
                    segments.append((start, end, segment, segment_index, core_start, core_end))
                else:
                    segments.append((start, end, segment, output_segment_index, core_start, core_end))

    for index in range(1, starts.size):
        start = int(starts[index])
        end = int(ends[index])
        gap = start - group_end
        merged_span = end - group_start
        if gap <= max_gap_samples and merged_span <= max_merged_samples:
            group_end = end
            continue

        flush_group(index, group_start, group_end)
        group_start_index = index
        group_start = start
        group_end = end

    flush_group(starts.size, group_start, group_end)
    if segment_return_mode == "compact":
        _add_optional_timing(
            timing_context, "segment_descriptor_count", len(descriptor_lengths)
        )
        return HostSegmentBatch(
            buffers,
            np.asarray(buffer_indices, dtype=np.int64),
            np.asarray(descriptor_starts, dtype=np.int64),
            np.asarray(descriptor_ends, dtype=np.int64),
            np.asarray(descriptor_offsets, dtype=np.int64),
            np.asarray(descriptor_lengths, dtype=np.int64),
            np.asarray(descriptor_segment_indices, dtype=np.int64),
            np.asarray(descriptor_core_starts, dtype=np.int64),
            np.asarray(descriptor_core_ends, dtype=np.int64),
        )
    return segments


def _apply_cleanup_lpf_cuda(iq, cleanup_lpf, device_id, return_host, cuda_fir_backend):
    if cuda_fir_backend == "kernel":
        filtered = _apply_cleanup_lpf_cuda_kernel(iq, cleanup_lpf, device_id)
    elif cuda_fir_backend == "kernel_shared":
        filtered = _apply_cleanup_lpf_cuda_kernel_shared(iq, cleanup_lpf, device_id)
    elif cuda_fir_backend == "kernel_multi":
        filtered = _apply_cleanup_lpf_cuda_kernel_multi(iq, cleanup_lpf, device_id)
    elif cuda_fir_backend == "kernel_multi_float":
        filtered = _apply_cleanup_lpf_cuda_kernel_multi_float(
            iq, cleanup_lpf, device_id
        )
    elif cuda_fir_backend == "kernel_const":
        filtered = _apply_cleanup_lpf_cuda_kernel_const(iq, cleanup_lpf, device_id)
    elif cuda_fir_backend == "cupyx":
        filtered = _apply_cleanup_lpf_cuda_cupyx(iq, cleanup_lpf, device_id)
    else:
        raise ValueError(
            f"Unknown CUDA cleanup FIR backend: {cuda_fir_backend}. "
            "Expected 'cupyx', 'kernel', 'kernel_shared', 'kernel_multi', "
            "'kernel_multi_float', or 'kernel_const'."
        )
    if return_host:
        return cuda_backend.to_host(filtered, dtype=np.complex64)
    return filtered


def _apply_cleanup_lpf_cuda_cupyx(iq, cleanup_lpf, device_id):
    cp = cuda_backend.require_cuda()
    gpu_lfilter = _get_gpu_lfilter()
    with cuda_backend.use_device(device_id):
        iq_gpu = cuda_backend.to_device(iq, dtype=cp.complex64)
        cleanup_lpf_gpu = cuda_backend.to_device(cleanup_lpf)
        lfilter_den = cp.asarray([1.0], dtype=cleanup_lpf_gpu.dtype)
        filtered = gpu_lfilter(cleanup_lpf_gpu, lfilter_den, iq_gpu).astype(
            cp.complex64, copy=False
        )
    return filtered


def _apply_cleanup_lpf_cuda_kernel(iq, cleanup_lpf, device_id):
    cp = cuda_backend.require_cuda()
    kernel = _get_cleanup_fir_kernel()

    with cuda_backend.use_device(device_id):
        iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))
        cleanup_lpf_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(cleanup_lpf, dtype=cp.float64)
        )
        output = cp.empty_like(iq_gpu, dtype=cp.complex64)

        cols = int(iq_gpu.shape[-1]) if iq_gpu.ndim else int(iq_gpu.size)
        rows = int(iq_gpu.size // cols) if cols else 0
        total = int(iq_gpu.size)
        if total:
            threads = 256
            blocks = (total + threads - 1) // threads
            kernel(
                (blocks,),
                (threads,),
                (
                    iq_gpu,
                    cleanup_lpf_gpu,
                    output,
                    np.int64(rows),
                    np.int64(cols),
                    np.int64(cleanup_lpf_gpu.size),
                ),
            )
    return output


def _apply_cleanup_lpf_cuda_kernel_shared(iq, cleanup_lpf, device_id):
    cp = cuda_backend.require_cuda()
    kernel = _get_cleanup_fir_shared_kernel()

    with cuda_backend.use_device(device_id):
        iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))
        cleanup_lpf_gpu = cp.ascontiguousarray(
            cuda_backend.to_device(cleanup_lpf, dtype=cp.float64)
        )
        output = cp.empty_like(iq_gpu, dtype=cp.complex64)

        cols = int(iq_gpu.shape[-1]) if iq_gpu.ndim else int(iq_gpu.size)
        rows = int(iq_gpu.size // cols) if cols else 0
        if rows and cols:
            threads = 256
            blocks_x = (cols + threads - 1) // threads
            shared_bytes = (threads + int(cleanup_lpf_gpu.size) - 1) * 8
            kernel(
                (blocks_x, rows),
                (threads,),
                (
                    iq_gpu,
                    cleanup_lpf_gpu,
                    output,
                    np.int64(cols),
                    np.int64(cleanup_lpf_gpu.size),
                ),
                shared_mem=shared_bytes,
            )
    return output


def _apply_cleanup_lpf_cuda_kernel_const(iq, cleanup_lpf, device_id):
    cp = cuda_backend.require_cuda()
    cleanup_lpf_host = np.ascontiguousarray(cleanup_lpf, dtype=np.float64)
    if cleanup_lpf_host.size > _MAX_CLEANUP_CONST_TAPS:
        raise ValueError(
            f"cleanup FIR has {cleanup_lpf_host.size} taps, which exceeds the "
            f"kernel_const limit of {_MAX_CLEANUP_CONST_TAPS}."
        )

    kernel = _get_cleanup_fir_const_kernel()

    with cuda_backend.use_device(device_id):
        _load_cleanup_fir_const_taps(cleanup_lpf_host)
        iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))
        output = cp.empty_like(iq_gpu, dtype=cp.complex64)

        cols = int(iq_gpu.shape[-1]) if iq_gpu.ndim else int(iq_gpu.size)
        rows = int(iq_gpu.size // cols) if cols else 0
        total = int(iq_gpu.size)
        if total:
            threads = 256
            blocks = (total + threads - 1) // threads
            kernel(
                (blocks,),
                (threads,),
                (
                    iq_gpu,
                    output,
                    np.int64(rows),
                    np.int64(cols),
                    np.int64(cleanup_lpf_host.size),
                ),
            )
    return output


def _apply_cleanup_lpf_cuda_kernel_multi(iq, cleanup_lpf, device_id):
    cp = cuda_backend.require_cuda()
    cleanup_lpf_host = np.ascontiguousarray(cleanup_lpf, dtype=np.float64)
    if cleanup_lpf_host.size > _MAX_CLEANUP_CONST_TAPS:
        raise ValueError(
            f"cleanup FIR has {cleanup_lpf_host.size} taps, which exceeds the "
            f"kernel_multi limit of {_MAX_CLEANUP_CONST_TAPS}."
        )

    kernel = _get_cleanup_fir_multi_kernel()

    with cuda_backend.use_device(device_id):
        _load_cleanup_fir_multi_taps(cleanup_lpf_host)
        iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))
        output = cp.empty_like(iq_gpu, dtype=cp.complex64)

        cols = int(iq_gpu.shape[-1]) if iq_gpu.ndim else int(iq_gpu.size)
        rows = int(iq_gpu.size // cols) if cols else 0
        if rows and cols:
            threads = 256
            outputs_per_thread = _CLEANUP_MULTI_OUTPUTS_PER_THREAD
            blocks_x = (cols + threads * outputs_per_thread - 1) // (
                threads * outputs_per_thread
            )
            shared_len = threads * outputs_per_thread + int(cleanup_lpf_host.size) - 1
            kernel(
                (blocks_x, rows),
                (threads,),
                (
                    iq_gpu,
                    output,
                    np.int64(cols),
                    np.int64(cleanup_lpf_host.size),
                ),
                shared_mem=shared_len * 8,
            )
    return output


def _apply_cleanup_lpf_cuda_kernel_multi_float(iq, cleanup_lpf, device_id):
    cp = cuda_backend.require_cuda()
    cleanup_lpf_host = np.ascontiguousarray(cleanup_lpf, dtype=np.float32)
    if cleanup_lpf_host.size > _MAX_CLEANUP_CONST_TAPS:
        raise ValueError(
            f"cleanup FIR has {cleanup_lpf_host.size} taps, which exceeds the "
            f"kernel_multi_float limit of {_MAX_CLEANUP_CONST_TAPS}."
        )

    kernel = _get_cleanup_fir_multi_float_kernel()

    with cuda_backend.use_device(device_id):
        _load_cleanup_fir_multi_float_taps(cleanup_lpf_host)
        iq_gpu = cp.ascontiguousarray(cuda_backend.to_device(iq, dtype=cp.complex64))
        output = cp.empty_like(iq_gpu, dtype=cp.complex64)

        cols = int(iq_gpu.shape[-1]) if iq_gpu.ndim else int(iq_gpu.size)
        rows = int(iq_gpu.size // cols) if cols else 0
        if rows and cols:
            threads = 256
            outputs_per_thread = _CLEANUP_MULTI_OUTPUTS_PER_THREAD
            blocks_x = (cols + threads * outputs_per_thread - 1) // (
                threads * outputs_per_thread
            )
            shared_len = threads * outputs_per_thread + int(cleanup_lpf_host.size) - 1
            kernel(
                (blocks_x, rows),
                (threads,),
                (
                    iq_gpu,
                    output,
                    np.int64(cols),
                    np.int64(cleanup_lpf_host.size),
                ),
                shared_mem=shared_len * 8,
            )
    return output


def describe_target_mapping(target_freqs_mhz, center_freq_hz, sample_rate, num_channels):
    bin_spacing_hz = sample_rate / num_channels
    mappings = []
    for target_freq_mhz in target_freqs_mhz:
        target_offset_hz = target_freq_mhz * 1e6 - center_freq_hz
        signed_bin = int(np.floor(target_offset_hz / bin_spacing_hz + 0.5))
        coarse_freq_mhz = (center_freq_hz + signed_bin * bin_spacing_hz) / 1e6
        residual_hz = target_offset_hz - signed_bin * bin_spacing_hz
        mappings.append((target_freq_mhz, coarse_freq_mhz, residual_hz))
    return mappings


def _integer_ratio(sample_rate, subband_sample_rate):
    ratio = sample_rate / subband_sample_rate
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1e-6:
        raise ValueError("Wideband sample rate must be an integer multiple of subband sample rate.")
    return rounded


def _get_gpu_lfilter():
    try:
        from cupyx.scipy.signal import lfilter as gpu_lfilter
    except Exception as exc:
        raise cuda_backend.CudaUnavailableError(
            f"CuPy signal lfilter is unavailable: {exc}"
        ) from exc
    return gpu_lfilter


_CLEANUP_FIR_KERNEL = None
_CLEANUP_FIR_SHARED_KERNEL = None
_CLEANUP_FIR_CONST_MODULE = None
_CLEANUP_FIR_CONST_KERNEL = None
_CLEANUP_FIR_CONST_CACHE = {}
_CLEANUP_FIR_MULTI_MODULE = None
_CLEANUP_FIR_MULTI_KERNEL = None
_CLEANUP_FIR_MULTI_CACHE = {}
_CLEANUP_FIR_MULTI_FLOAT_MODULE = None
_CLEANUP_FIR_MULTI_FLOAT_KERNEL = None
_CLEANUP_FIR_MULTI_FLOAT_CACHE = {}
_PFB_POLYPHASE_KERNEL = None
_PFB_POLYPHASE_I16_KERNEL = None
_PFB_POLYPHASE_MULTI_KERNEL = None
_PFB_POLYPHASE_MULTI_I16_KERNEL = None
_PFB_POLYPHASE_MULTI_FLOAT_KERNEL = None
_PFB_POLYPHASE_MULTI_FLOAT_I16_KERNEL = None
_PFB_POLYPHASE_MULTI_FLOAT_TRANSPOSED_KERNEL = None
_PFB_POLYPHASE_MULTI_FLOAT_I16_TRANSPOSED_KERNEL = None
_PFB_POLYPHASE_CONST_MODULE = None
_PFB_POLYPHASE_CONST_KERNEL = None
_PFB_POLYPHASE_CONST_CACHE = {}
_PFB_PHASE_APPLY_KERNEL = None
_PFB_PHASE_APPLY_TRANSPOSED_KERNEL = None
_PFB_PHASE_TABLE_CACHE = {}
_PFB_COMBINED_PHASE_TABLE_CACHE = {}
_INT16_IQ_DECODE_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MASK_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_MASK_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_MASK_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_MASK_KERNEL = None
_SHIFT_CLEANUP_MATERIALIZE_RANGES_KERNEL = None
_SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_CACHE = {}
_KNOWN_CANDIDATE_FILTER_KERNEL = None
_MAX_CLEANUP_CONST_TAPS = 256
_CLEANUP_MULTI_OUTPUTS_PER_THREAD = 4
_MAX_PFB_CONST_TAPS = 1024


def _get_int16_iq_decode_kernel():
    global _INT16_IQ_DECODE_KERNEL
    if _INT16_IQ_DECODE_KERNEL is not None:
        return _INT16_IQ_DECODE_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void int16_iq_to_complex64(
    const short* raw,
    float2* iq,
    const long long num_samples,
    const long long output_offset
) {
    const long long sample_index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sample_index >= num_samples) {
        return;
    }
    const long long raw_index = sample_index * 2;
    iq[output_offset + sample_index] = make_float2(
        static_cast<float>(raw[raw_index]) * (1.0f / 32768.0f),
        static_cast<float>(raw[raw_index + 1]) * (1.0f / 32768.0f)
    );
}
"""
    _INT16_IQ_DECODE_KERNEL = cp.RawKernel(kernel_code, "int16_iq_to_complex64")
    return _INT16_IQ_DECODE_KERNEL


def _get_shift_cleanup_threshold_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void shift_cleanup_threshold(
    const float2* x,
    const double* taps,
    float2* y,
    signed char* above,
    const long long n,
    const long long ntaps,
    const double threshold,
    const double chunk_start,
    const double decim,
    const double sample_rate,
    const double residual_hz
) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    double acc_re = 0.0;
    double acc_im = 0.0;
    const long long max_k = idx + 1 < ntaps ? idx + 1 : ntaps;

    if (residual_hz == 0.0) {
        for (long long k = 0; k < max_k; ++k) {
            const float2 sample = x[idx - k];
            const double tap = taps[k];
            acc_re += static_cast<double>(sample.x) * tap;
            acc_im += static_cast<double>(sample.y) * tap;
        }
    } else {
        const double phase_scale = -6.283185307179586476925286766559 * residual_hz / sample_rate;
        const double phase = phase_scale * (chunk_start + static_cast<double>(idx) * decim);
        const double step_phase = -phase_scale * decim;
        double sine;
        double cosine;
        double step_sine;
        double step_cosine;
        sincos(phase, &sine, &cosine);
        sincos(step_phase, &step_sine, &step_cosine);

        for (long long k = 0; k < max_k; ++k) {
            const float2 sample = x[idx - k];
            const double shifted_re = static_cast<double>(sample.x) * cosine -
                                      static_cast<double>(sample.y) * sine;
            const double shifted_im = static_cast<double>(sample.x) * sine +
                                      static_cast<double>(sample.y) * cosine;
            const double tap = taps[k];
            acc_re += shifted_re * tap;
            acc_im += shifted_im * tap;

            const double next_cosine = cosine * step_cosine - sine * step_sine;
            sine = sine * step_cosine + cosine * step_sine;
            cosine = next_cosine;
        }
    }

    y[idx] = make_float2(static_cast<float>(acc_re), static_cast<float>(acc_im));
    above[idx] = (acc_re * acc_re + acc_im * acc_im) > (threshold * threshold) ? 1 : 0;
}
"""
    _SHIFT_CLEANUP_THRESHOLD_KERNEL = cp.RawKernel(
        kernel_code, "shift_cleanup_threshold"
    )
    return _SHIFT_CLEANUP_THRESHOLD_KERNEL


def _get_shift_cleanup_threshold_multi_float_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = rf"""
__constant__ float shift_cleanup_taps_multi_float_const[{_MAX_CLEANUP_CONST_TAPS}];

__device__ __forceinline__
float2 shift_sample(
    const float2 sample,
    const double sample_col,
    const double chunk_start,
    const double decim,
    const double sample_rate,
    const double residual_hz
) {{
    if (residual_hz == 0.0) {{
        return sample;
    }}
    const double phase =
        -6.283185307179586476925286766559 * residual_hz *
        (chunk_start + sample_col * decim) / sample_rate;
    double sine;
    double cosine;
    sincos(phase, &sine, &cosine);
    return make_float2(
        static_cast<float>(static_cast<double>(sample.x) * cosine -
                           static_cast<double>(sample.y) * sine),
        static_cast<float>(static_cast<double>(sample.x) * sine +
                           static_cast<double>(sample.y) * cosine)
    );
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float(
    const float2* x,
    float2* y,
    signed char* above,
    const long long n,
    const long long ntaps,
    const float threshold,
    const double chunk_start,
    const double decim,
    const double sample_rate,
    const double residual_hz
) {{
    extern __shared__ float2 tile[];

    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            tile[tile_idx] = shift_sample(
                x[source_col],
                static_cast<double>(source_col),
                chunk_start,
                decim,
                sample_rate,
                residual_hz
            );
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    float acc0_re = 0.0f;
    float acc0_im = 0.0f;
    float acc1_re = 0.0f;
    float acc1_im = 0.0f;
    float acc2_re = 0.0f;
    float acc2_im = 0.0f;
    float acc3_re = 0.0f;
    float acc3_im = 0.0f;

    for (long long k = 0; k < ntaps; ++k) {{
        const float tap = shift_cleanup_taps_multi_float_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += sample0.x * tap;
        acc0_im += sample0.y * tap;
        acc1_re += sample1.x * tap;
        acc1_im += sample1.y * tap;
        acc2_re += sample2.x * tap;
        acc2_im += sample2.y * tap;
        acc3_re += sample3.x * tap;
        acc3_im += sample3.y * tap;
    }}

    const float threshold_sq = threshold * threshold;
    if (first_col < n) {{
        y[first_col] = make_float2(acc0_re, acc0_im);
        above[first_col] = (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 1 < n) {{
        y[first_col + 1] = make_float2(acc1_re, acc1_im);
        above[first_col + 1] = (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 2 < n) {{
        y[first_col + 2] = make_float2(acc2_re, acc2_im);
        above[first_col + 2] = (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 3 < n) {{
        y[first_col + 3] = make_float2(acc3_re, acc3_im);
        above[first_col + 3] = (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
    }}
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float_batch(
    const float2* x,
    const double* residual_hz_values,
    float2* y,
    signed char* above,
    const long long n,
    const long long ntaps,
    const float threshold,
    const double chunk_start,
    const double decim,
    const double sample_rate
) {{
    extern __shared__ float2 tile[];

    const long long tid = threadIdx.x;
    const long long target_index = blockIdx.y;
    const double residual_hz = residual_hz_values[target_index];
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            tile[tile_idx] = shift_sample(
                x[source_col],
                static_cast<double>(source_col),
                chunk_start,
                decim,
                sample_rate,
                residual_hz
            );
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    float acc0_re = 0.0f;
    float acc0_im = 0.0f;
    float acc1_re = 0.0f;
    float acc1_im = 0.0f;
    float acc2_re = 0.0f;
    float acc2_im = 0.0f;
    float acc3_re = 0.0f;
    float acc3_im = 0.0f;

    for (long long k = 0; k < ntaps; ++k) {{
        const float tap = shift_cleanup_taps_multi_float_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += sample0.x * tap;
        acc0_im += sample0.y * tap;
        acc1_re += sample1.x * tap;
        acc1_im += sample1.y * tap;
        acc2_re += sample2.x * tap;
        acc2_im += sample2.y * tap;
        acc3_re += sample3.x * tap;
        acc3_im += sample3.y * tap;
    }}

    const float threshold_sq = threshold * threshold;
    const long long row_offset = target_index * n;
    if (first_col < n) {{
        y[row_offset + first_col] = make_float2(acc0_re, acc0_im);
        above[row_offset + first_col] =
            (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 1 < n) {{
        y[row_offset + first_col + 1] = make_float2(acc1_re, acc1_im);
        above[row_offset + first_col + 1] =
            (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 2 < n) {{
        y[row_offset + first_col + 2] = make_float2(acc2_re, acc2_im);
        above[row_offset + first_col + 2] =
            (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 3 < n) {{
        y[row_offset + first_col + 3] = make_float2(acc3_re, acc3_im);
        above[row_offset + first_col + 3] =
            (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
    }}
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float_batch_shared_input(
    const float2* x,
    const double* residual_hz_values,
    float2* y,
    signed char* above,
    const long long n,
    const long long target_count,
    const long long ntaps,
    const float threshold,
    const double chunk_start,
    const double decim,
    const double sample_rate
) {{
    extern __shared__ float2 shared[];

    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;
    float2* raw_tile = shared;
    float2* shifted_tile = shared + tile_len;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            raw_tile[tile_idx] = x[source_col];
        }} else {{
            raw_tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;
    const float threshold_sq = threshold * threshold;

    for (long long target_index = 0; target_index < target_count; ++target_index) {{
        const double residual_hz = residual_hz_values[target_index];
        for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
            const long long source_col = col_base + tile_idx - halo;
            if (source_col >= 0 && source_col < n) {{
                shifted_tile[tile_idx] = shift_sample(
                    raw_tile[tile_idx],
                    static_cast<double>(source_col),
                    chunk_start,
                    decim,
                    sample_rate,
                    residual_hz
                );
            }} else {{
                shifted_tile[tile_idx] = make_float2(0.0f, 0.0f);
            }}
        }}
        __syncthreads();

        float acc0_re = 0.0f;
        float acc0_im = 0.0f;
        float acc1_re = 0.0f;
        float acc1_im = 0.0f;
        float acc2_re = 0.0f;
        float acc2_im = 0.0f;
        float acc3_re = 0.0f;
        float acc3_im = 0.0f;

        for (long long k = 0; k < ntaps; ++k) {{
            const float tap = shift_cleanup_taps_multi_float_const[k];
            const float2 sample0 = shifted_tile[first_sample_offset - k];
            const float2 sample1 = shifted_tile[first_sample_offset + 1 - k];
            const float2 sample2 = shifted_tile[first_sample_offset + 2 - k];
            const float2 sample3 = shifted_tile[first_sample_offset + 3 - k];
            acc0_re += sample0.x * tap;
            acc0_im += sample0.y * tap;
            acc1_re += sample1.x * tap;
            acc1_im += sample1.y * tap;
            acc2_re += sample2.x * tap;
            acc2_im += sample2.y * tap;
            acc3_re += sample3.x * tap;
            acc3_im += sample3.y * tap;
        }}

        const long long row_offset = target_index * n;
        if (first_col < n) {{
            y[row_offset + first_col] = make_float2(acc0_re, acc0_im);
            above[row_offset + first_col] =
                (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
        }}
        if (first_col + 1 < n) {{
            y[row_offset + first_col + 1] = make_float2(acc1_re, acc1_im);
            above[row_offset + first_col + 1] =
                (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
        }}
        if (first_col + 2 < n) {{
            y[row_offset + first_col + 2] = make_float2(acc2_re, acc2_im);
            above[row_offset + first_col + 2] =
                (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
        }}
        if (first_col + 3 < n) {{
            y[row_offset + first_col + 3] = make_float2(acc3_re, acc3_im);
            above[row_offset + first_col + 3] =
                (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
        }}
        __syncthreads();
    }}
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float_zero(
    const float2* x,
    float2* y,
    signed char* above,
    const long long n,
    const long long ntaps,
    const float threshold
) {{
    extern __shared__ float2 tile[];

    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            tile[tile_idx] = x[source_col];
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    float acc0_re = 0.0f;
    float acc0_im = 0.0f;
    float acc1_re = 0.0f;
    float acc1_im = 0.0f;
    float acc2_re = 0.0f;
    float acc2_im = 0.0f;
    float acc3_re = 0.0f;
    float acc3_im = 0.0f;

    for (long long k = 0; k < ntaps; ++k) {{
        const float tap = shift_cleanup_taps_multi_float_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += sample0.x * tap;
        acc0_im += sample0.y * tap;
        acc1_re += sample1.x * tap;
        acc1_im += sample1.y * tap;
        acc2_re += sample2.x * tap;
        acc2_im += sample2.y * tap;
        acc3_re += sample3.x * tap;
        acc3_im += sample3.y * tap;
    }}

    const float threshold_sq = threshold * threshold;
    if (first_col < n) {{
        y[first_col] = make_float2(acc0_re, acc0_im);
        above[first_col] = (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 1 < n) {{
        y[first_col + 1] = make_float2(acc1_re, acc1_im);
        above[first_col + 1] = (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 2 < n) {{
        y[first_col + 2] = make_float2(acc2_re, acc2_im);
        above[first_col + 2] = (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 3 < n) {{
        y[first_col + 3] = make_float2(acc3_re, acc3_im);
        above[first_col + 3] = (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
    }}
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float_mask(
    const float2* x,
    signed char* above,
    const long long n,
    const long long ntaps,
    const float threshold,
    const double chunk_start,
    const double decim,
    const double sample_rate,
    const double residual_hz
) {{
    extern __shared__ float2 tile[];

    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            tile[tile_idx] = shift_sample(
                x[source_col],
                static_cast<double>(source_col),
                chunk_start,
                decim,
                sample_rate,
                residual_hz
            );
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    float acc0_re = 0.0f;
    float acc0_im = 0.0f;
    float acc1_re = 0.0f;
    float acc1_im = 0.0f;
    float acc2_re = 0.0f;
    float acc2_im = 0.0f;
    float acc3_re = 0.0f;
    float acc3_im = 0.0f;

    for (long long k = 0; k < ntaps; ++k) {{
        const float tap = shift_cleanup_taps_multi_float_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += sample0.x * tap;
        acc0_im += sample0.y * tap;
        acc1_re += sample1.x * tap;
        acc1_im += sample1.y * tap;
        acc2_re += sample2.x * tap;
        acc2_im += sample2.y * tap;
        acc3_re += sample3.x * tap;
        acc3_im += sample3.y * tap;
    }}

    const float threshold_sq = threshold * threshold;
    if (first_col < n) {{
        above[first_col] = (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 1 < n) {{
        above[first_col + 1] = (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 2 < n) {{
        above[first_col + 2] = (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 3 < n) {{
        above[first_col + 3] = (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
    }}
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float_zero_mask(
    const float2* x,
    signed char* above,
    const long long n,
    const long long ntaps,
    const float threshold
) {{
    extern __shared__ float2 tile[];

    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            tile[tile_idx] = x[source_col];
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    float acc0_re = 0.0f;
    float acc0_im = 0.0f;
    float acc1_re = 0.0f;
    float acc1_im = 0.0f;
    float acc2_re = 0.0f;
    float acc2_im = 0.0f;
    float acc3_re = 0.0f;
    float acc3_im = 0.0f;

    for (long long k = 0; k < ntaps; ++k) {{
        const float tap = shift_cleanup_taps_multi_float_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += sample0.x * tap;
        acc0_im += sample0.y * tap;
        acc1_re += sample1.x * tap;
        acc1_im += sample1.y * tap;
        acc2_re += sample2.x * tap;
        acc2_im += sample2.y * tap;
        acc3_re += sample3.x * tap;
        acc3_im += sample3.y * tap;
    }}

    const float threshold_sq = threshold * threshold;
    if (first_col < n) {{
        above[first_col] = (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 1 < n) {{
        above[first_col + 1] = (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 2 < n) {{
        above[first_col + 2] = (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 3 < n) {{
        above[first_col + 3] = (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
    }}
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float_batch_mask(
    const float2* x,
    const double* residual_hz_values,
    signed char* above,
    const long long n,
    const long long ntaps,
    const float threshold,
    const double chunk_start,
    const double decim,
    const double sample_rate
) {{
    extern __shared__ float2 tile[];

    const long long tid = threadIdx.x;
    const long long target_index = blockIdx.y;
    const double residual_hz = residual_hz_values[target_index];
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            tile[tile_idx] = shift_sample(
                x[source_col],
                static_cast<double>(source_col),
                chunk_start,
                decim,
                sample_rate,
                residual_hz
            );
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    float acc0_re = 0.0f;
    float acc0_im = 0.0f;
    float acc1_re = 0.0f;
    float acc1_im = 0.0f;
    float acc2_re = 0.0f;
    float acc2_im = 0.0f;
    float acc3_re = 0.0f;
    float acc3_im = 0.0f;

    for (long long k = 0; k < ntaps; ++k) {{
        const float tap = shift_cleanup_taps_multi_float_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += sample0.x * tap;
        acc0_im += sample0.y * tap;
        acc1_re += sample1.x * tap;
        acc1_im += sample1.y * tap;
        acc2_re += sample2.x * tap;
        acc2_im += sample2.y * tap;
        acc3_re += sample3.x * tap;
        acc3_im += sample3.y * tap;
    }}

    const float threshold_sq = threshold * threshold;
    const long long row_offset = target_index * n;
    if (first_col < n) {{
        above[row_offset + first_col] =
            (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 1 < n) {{
        above[row_offset + first_col + 1] =
            (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 2 < n) {{
        above[row_offset + first_col + 2] =
            (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
    }}
    if (first_col + 3 < n) {{
        above[row_offset + first_col + 3] =
            (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
    }}
}}

extern "C" __global__
void shift_cleanup_threshold_multi_float_batch_shared_input_mask(
    const float2* x,
    const double* residual_hz_values,
    signed char* above,
    const long long n,
    const long long target_count,
    const long long ntaps,
    const float threshold,
    const double chunk_start,
    const double decim,
    const double sample_rate
) {{
    extern __shared__ float2 shared[];

    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;
    float2* raw_tile = shared;
    float2* shifted_tile = shared + tile_len;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < n) {{
            raw_tile[tile_idx] = x[source_col];
        }} else {{
            raw_tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;
    const float threshold_sq = threshold * threshold;

    for (long long target_index = 0; target_index < target_count; ++target_index) {{
        const double residual_hz = residual_hz_values[target_index];
        for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
            const long long source_col = col_base + tile_idx - halo;
            if (source_col >= 0 && source_col < n) {{
                shifted_tile[tile_idx] = shift_sample(
                    raw_tile[tile_idx],
                    static_cast<double>(source_col),
                    chunk_start,
                    decim,
                    sample_rate,
                    residual_hz
                );
            }} else {{
                shifted_tile[tile_idx] = make_float2(0.0f, 0.0f);
            }}
        }}
        __syncthreads();

        float acc0_re = 0.0f;
        float acc0_im = 0.0f;
        float acc1_re = 0.0f;
        float acc1_im = 0.0f;
        float acc2_re = 0.0f;
        float acc2_im = 0.0f;
        float acc3_re = 0.0f;
        float acc3_im = 0.0f;

        for (long long k = 0; k < ntaps; ++k) {{
            const float tap = shift_cleanup_taps_multi_float_const[k];
            const float2 sample0 = shifted_tile[first_sample_offset - k];
            const float2 sample1 = shifted_tile[first_sample_offset + 1 - k];
            const float2 sample2 = shifted_tile[first_sample_offset + 2 - k];
            const float2 sample3 = shifted_tile[first_sample_offset + 3 - k];
            acc0_re += sample0.x * tap;
            acc0_im += sample0.y * tap;
            acc1_re += sample1.x * tap;
            acc1_im += sample1.y * tap;
            acc2_re += sample2.x * tap;
            acc2_im += sample2.y * tap;
            acc3_re += sample3.x * tap;
            acc3_im += sample3.y * tap;
        }}

        const long long row_offset = target_index * n;
        if (first_col < n) {{
            above[row_offset + first_col] =
                (acc0_re * acc0_re + acc0_im * acc0_im) > threshold_sq ? 1 : 0;
        }}
        if (first_col + 1 < n) {{
            above[row_offset + first_col + 1] =
                (acc1_re * acc1_re + acc1_im * acc1_im) > threshold_sq ? 1 : 0;
        }}
        if (first_col + 2 < n) {{
            above[row_offset + first_col + 2] =
                (acc2_re * acc2_re + acc2_im * acc2_im) > threshold_sq ? 1 : 0;
        }}
        if (first_col + 3 < n) {{
            above[row_offset + first_col + 3] =
                (acc3_re * acc3_re + acc3_im * acc3_im) > threshold_sq ? 1 : 0;
        }}
        __syncthreads();
    }}
}}

extern "C" __global__
void shift_cleanup_materialize_ranges(
    const float2* x,
    float2* out,
    const long long* range_starts,
    const long long* range_lengths,
    const long long* output_offsets,
    const long long range_count,
    const long long total_samples,
    const long long n,
    const long long ntaps,
    const double chunk_start,
    const double decim,
    const double sample_rate,
    const double residual_hz
) {{
    const long long out_idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total_samples) {{
        return;
    }}

    long long lo = 0;
    long long hi = range_count - 1;
    while (lo <= hi) {{
        const long long mid = (lo + hi) / 2;
        const long long begin = output_offsets[mid];
        const long long end = begin + range_lengths[mid];
        if (out_idx < begin) {{
            hi = mid - 1;
        }} else if (out_idx >= end) {{
            lo = mid + 1;
        }} else {{
            const long long sample_col = range_starts[mid] + (out_idx - begin);
            float acc_re = 0.0f;
            float acc_im = 0.0f;
            const long long max_k = sample_col + 1 < ntaps ? sample_col + 1 : ntaps;
            for (long long k = 0; k < max_k; ++k) {{
                const long long source_col = sample_col - k;
                if (source_col < 0 || source_col >= n) {{
                    continue;
                }}
                const float2 shifted = shift_sample(
                    x[source_col],
                    static_cast<double>(source_col),
                    chunk_start,
                    decim,
                    sample_rate,
                    residual_hz
                );
                const float tap = shift_cleanup_taps_multi_float_const[k];
                acc_re += shifted.x * tap;
                acc_im += shifted.y * tap;
            }}
            out[out_idx] = make_float2(acc_re, acc_im);
            return;
        }}
    }}
}}
"""
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE = cp.RawModule(
        code=kernel_code,
        options=("-std=c++11",),
        name_expressions=(
            "shift_cleanup_threshold_multi_float",
            "shift_cleanup_threshold_multi_float_batch",
            "shift_cleanup_threshold_multi_float_batch_shared_input",
            "shift_cleanup_threshold_multi_float_zero",
            "shift_cleanup_threshold_multi_float_mask",
            "shift_cleanup_threshold_multi_float_zero_mask",
            "shift_cleanup_threshold_multi_float_batch_mask",
            "shift_cleanup_threshold_multi_float_batch_shared_input_mask",
            "shift_cleanup_materialize_ranges",
        ),
    )
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_KERNEL


def _get_shift_cleanup_threshold_multi_float_zero_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float_zero"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_KERNEL


def _get_shift_cleanup_threshold_multi_float_batch_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float_batch"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_KERNEL


def _get_shift_cleanup_threshold_multi_float_batch_shared_input_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float_batch_shared_input"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_KERNEL


def _get_shift_cleanup_threshold_multi_float_mask_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MASK_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MASK_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MASK_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MASK_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float_mask"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MASK_KERNEL


def _get_shift_cleanup_threshold_multi_float_zero_mask_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_MASK_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_MASK_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_MASK_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_MASK_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float_zero_mask"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_ZERO_MASK_KERNEL


def _get_shift_cleanup_threshold_multi_float_batch_mask_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_MASK_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_MASK_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_MASK_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_MASK_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float_batch_mask"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_MASK_KERNEL


def _get_shift_cleanup_threshold_multi_float_batch_shared_input_mask_kernel():
    global _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_MASK_KERNEL
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_MASK_KERNEL is not None:
        return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_MASK_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_MASK_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_threshold_multi_float_batch_shared_input_mask"
        )
    )
    return _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_BATCH_SHARED_INPUT_MASK_KERNEL


def _get_shift_cleanup_materialize_ranges_kernel():
    global _SHIFT_CLEANUP_MATERIALIZE_RANGES_KERNEL
    if _SHIFT_CLEANUP_MATERIALIZE_RANGES_KERNEL is not None:
        return _SHIFT_CLEANUP_MATERIALIZE_RANGES_KERNEL

    _get_shift_cleanup_threshold_multi_float_kernel()
    _SHIFT_CLEANUP_MATERIALIZE_RANGES_KERNEL = (
        _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_function(
            "shift_cleanup_materialize_ranges"
        )
    )
    return _SHIFT_CLEANUP_MATERIALIZE_RANGES_KERNEL


def _load_shift_cleanup_threshold_multi_float_taps(cleanup_lpf_host):
    cp = cuda_backend.require_cuda()
    _get_shift_cleanup_threshold_multi_float_kernel()

    device_id = int(cp.cuda.runtime.getDevice())
    taps_bytes = cleanup_lpf_host.tobytes()
    if _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_CACHE.get(device_id) == taps_bytes:
        return

    taps_ptr = _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_MODULE.get_global(
        "shift_cleanup_taps_multi_float_const"
    )
    taps_gpu = cp.ndarray(
        (_MAX_CLEANUP_CONST_TAPS,),
        dtype=cp.float32,
        memptr=taps_ptr,
    )
    taps_gpu[: cleanup_lpf_host.size] = cp.asarray(
        cleanup_lpf_host, dtype=cp.float32
    )
    _SHIFT_CLEANUP_THRESHOLD_MULTI_FLOAT_CACHE[device_id] = taps_bytes


def _get_known_candidate_filter_kernel():
    global _KNOWN_CANDIDATE_FILTER_KERNEL
    if _KNOWN_CANDIDATE_FILTER_KERNEL is not None:
        return _KNOWN_CANDIDATE_FILTER_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
__device__ __forceinline__
int demod_bit_sign(const float2* iq, const long long sample_index) {
    const float2 a = iq[sample_index];
    const float2 b = iq[sample_index + 1];
    const float cross = a.x * b.y - a.y * b.x;
    return cross > 0.0f ? 1 : 0;
}

__device__
unsigned int capture_bits32(
    const float2* iq,
    const long long segment_start,
    const int bit_pos,
    const int sps
) {
    unsigned int value = 0;
    const int center = sps / 2;
    for (int bit = 0; bit < 32; ++bit) {
        const long long sample_index =
            segment_start + static_cast<long long>(bit_pos + bit) * sps + center;
        value |= static_cast<unsigned int>(demod_bit_sign(iq, sample_index)) << bit;
    }
    return value;
}

__device__
unsigned long long capture_bits64(
    const float2* iq,
    const long long segment_start,
    const int bit_pos,
    const int sps
) {
    unsigned long long value = 0;
    const int center = sps / 2;
    for (int bit = 0; bit < 64; ++bit) {
        const long long sample_index =
            segment_start + static_cast<long long>(bit_pos + bit) * sps + center;
        value |= static_cast<unsigned long long>(demod_bit_sign(iq, sample_index)) << bit;
    }
    return value;
}

extern "C" __global__
void known_candidate_filter(
    const float2* iq,
    const long long* starts,
    const long long* ends,
    signed char* keep,
    const long long nsegments,
    const int protocol,
    const int sps_1m,
    const int sps_2m,
    const unsigned int* ble_patterns,
    const int n_ble_patterns,
    const unsigned long long* bredr_patterns,
    const int n_bredr_patterns
) {
    const long long segment_index = blockIdx.x;
    if (segment_index >= nsegments) {
        return;
    }
    if (keep[segment_index]) {
        return;
    }

    const long long start = starts[segment_index];
    const long long end = ends[segment_index];
    const long long len = end - start;

    if (protocol == 1) {
        const int sps_values[2] = {sps_2m, sps_1m};
        for (int mode = 0; mode < 2; ++mode) {
            const int sps = sps_values[mode];
            if (sps <= 0 || len <= static_cast<long long>(33) * sps) {
                continue;
            }
            const int nbits = static_cast<int>((len - 1) / sps);
            const int max_pos = nbits - 32;
            for (int pos = threadIdx.x; pos <= max_pos; pos += blockDim.x) {
                const unsigned int captured = capture_bits32(iq, start, pos, sps);
                for (int pattern_index = 0; pattern_index < n_ble_patterns; ++pattern_index) {
                    if (captured == ble_patterns[pattern_index]) {
                        keep[segment_index] = 1;
                        return;
                    }
                }
            }
        }
        return;
    }

    if (protocol == 2) {
        const int sps = sps_1m;
        if (sps <= 0 || len <= static_cast<long long>(65) * sps) {
            return;
        }
        const int nbits = static_cast<int>((len - 1) / sps);
        const int max_pos = nbits - 64;
        for (int pos = threadIdx.x; pos <= max_pos; pos += blockDim.x) {
            const unsigned long long captured = capture_bits64(iq, start, pos, sps);
            for (int pattern_index = 0; pattern_index < n_bredr_patterns; ++pattern_index) {
                if (__popcll(captured ^ bredr_patterns[pattern_index]) <= 24) {
                    keep[segment_index] = 1;
                    return;
                }
            }
        }
    }
}
"""
    _KNOWN_CANDIDATE_FILTER_KERNEL = cp.RawKernel(
        kernel_code, "known_candidate_filter"
    )
    return _KNOWN_CANDIDATE_FILTER_KERNEL


def _get_pfb_polyphase_kernel():
    global _PFB_POLYPHASE_KERNEL
    if _PFB_POLYPHASE_KERNEL is not None:
        return _PFB_POLYPHASE_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_direct(
    const float2* iq,
    const double* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps
) {
    const long long output_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long phase = blockIdx.y;
    if (output_index >= num_outputs || phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long source_index = output_index + source_offset;
    double acc_re = 0.0;
    double acc_im = 0.0;

    for (long long tap_index = phase, tap_order = 0;
         tap_index < numtaps;
         tap_index += num_channels, ++tap_order) {
        const long long source_sample_index = source_index - tap_order;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index < iq_size) {
            const float2 sample = iq[iq_index];
            const double tap = prototype[tap_index];
            acc_re += static_cast<double>(sample.x) * tap;
            acc_im += static_cast<double>(sample.y) * tap;
        }
    }

    polyphase[phase * num_outputs + output_index] =
        make_float2(static_cast<float>(acc_re), static_cast<float>(acc_im));
}
"""
    _PFB_POLYPHASE_KERNEL = cp.RawKernel(kernel_code, "pfb_polyphase_direct")
    return _PFB_POLYPHASE_KERNEL


def _get_pfb_polyphase_i16_kernel():
    global _PFB_POLYPHASE_I16_KERNEL
    if _PFB_POLYPHASE_I16_KERNEL is not None:
        return _PFB_POLYPHASE_I16_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_direct_i16(
    const short* raw_iq,
    const double* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps
) {
    const long long output_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long phase = blockIdx.y;
    if (output_index >= num_outputs || phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long source_index = output_index + source_offset;
    double acc_re = 0.0;
    double acc_im = 0.0;

    for (long long tap_index = phase, tap_order = 0;
         tap_index < numtaps;
         tap_index += num_channels, ++tap_order) {
        const long long source_sample_index = source_index - tap_order;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index < iq_size) {
            const long long raw_index = iq_index * 2;
            const double sample_re =
                static_cast<double>(raw_iq[raw_index]) * (1.0 / 32768.0);
            const double sample_im =
                static_cast<double>(raw_iq[raw_index + 1]) * (1.0 / 32768.0);
            const double tap = prototype[tap_index];
            acc_re += sample_re * tap;
            acc_im += sample_im * tap;
        }
    }

    polyphase[phase * num_outputs + output_index] =
        make_float2(static_cast<float>(acc_re), static_cast<float>(acc_im));
}
"""
    _PFB_POLYPHASE_I16_KERNEL = cp.RawKernel(
        kernel_code, "pfb_polyphase_direct_i16"
    )
    return _PFB_POLYPHASE_I16_KERNEL


def _get_pfb_polyphase_multi_kernel():
    global _PFB_POLYPHASE_MULTI_KERNEL
    if _PFB_POLYPHASE_MULTI_KERNEL is not None:
        return _PFB_POLYPHASE_MULTI_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_multi(
    const float2* iq,
    const double* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps,
    const long long taps_per_phase
) {
    extern __shared__ float2 tile[];
    const long long outputs_per_thread = 4;
    const long long block_outputs = static_cast<long long>(blockDim.x) * outputs_per_thread;
    const long long block_output_start = static_cast<long long>(blockIdx.x) * block_outputs;
    const long long phase = blockIdx.y;
    if (phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long valid_taps =
        (numtaps > phase) ? ((numtaps - phase + num_channels - 1) / num_channels) : 0;
    const long long tile_samples = block_outputs + valid_taps - 1;
    const long long tile_source_start = block_output_start + source_offset - valid_taps + 1;

    for (long long local = threadIdx.x; local < tile_samples; local += blockDim.x) {
        const long long source_sample_index = tile_source_start + local;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index >= 0 && iq_index < iq_size) {
            tile[local] = iq[iq_index];
        } else {
            tile[local] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    const long long thread_output_base =
        block_output_start + static_cast<long long>(threadIdx.x) * outputs_per_thread;
    double acc_re[4] = {0.0, 0.0, 0.0, 0.0};
    double acc_im[4] = {0.0, 0.0, 0.0, 0.0};

    for (long long tap_order = 0; tap_order < valid_taps; ++tap_order) {
        const long long tap_index = phase + tap_order * num_channels;
        const double tap = prototype[tap_index];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long output_index = thread_output_base + j;
            if (output_index < num_outputs) {
                const long long tile_index =
                    static_cast<long long>(threadIdx.x) * outputs_per_thread
                    + j + valid_taps - 1 - tap_order;
                const float2 sample = tile[tile_index];
                acc_re[j] += static_cast<double>(sample.x) * tap;
                acc_im[j] += static_cast<double>(sample.y) * tap;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const long long output_index = thread_output_base + j;
        if (output_index < num_outputs) {
            polyphase[phase * num_outputs + output_index] =
                make_float2(static_cast<float>(acc_re[j]), static_cast<float>(acc_im[j]));
        }
    }
}
"""
    _PFB_POLYPHASE_MULTI_KERNEL = cp.RawKernel(kernel_code, "pfb_polyphase_multi")
    return _PFB_POLYPHASE_MULTI_KERNEL


def _get_pfb_polyphase_multi_i16_kernel():
    global _PFB_POLYPHASE_MULTI_I16_KERNEL
    if _PFB_POLYPHASE_MULTI_I16_KERNEL is not None:
        return _PFB_POLYPHASE_MULTI_I16_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_multi_i16(
    const short* raw_iq,
    const double* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps,
    const long long taps_per_phase
) {
    extern __shared__ float2 tile[];
    const long long outputs_per_thread = 4;
    const long long block_outputs = static_cast<long long>(blockDim.x) * outputs_per_thread;
    const long long block_output_start = static_cast<long long>(blockIdx.x) * block_outputs;
    const long long phase = blockIdx.y;
    if (phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long valid_taps =
        (numtaps > phase) ? ((numtaps - phase + num_channels - 1) / num_channels) : 0;
    const long long tile_samples = block_outputs + valid_taps - 1;
    const long long tile_source_start = block_output_start + source_offset - valid_taps + 1;

    for (long long local = threadIdx.x; local < tile_samples; local += blockDim.x) {
        const long long source_sample_index = tile_source_start + local;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index >= 0 && iq_index < iq_size) {
            const long long raw_index = iq_index * 2;
            tile[local] = make_float2(
                static_cast<float>(raw_iq[raw_index]) * (1.0f / 32768.0f),
                static_cast<float>(raw_iq[raw_index + 1]) * (1.0f / 32768.0f)
            );
        } else {
            tile[local] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    const long long thread_output_base =
        block_output_start + static_cast<long long>(threadIdx.x) * outputs_per_thread;
    double acc_re[4] = {0.0, 0.0, 0.0, 0.0};
    double acc_im[4] = {0.0, 0.0, 0.0, 0.0};

    for (long long tap_order = 0; tap_order < valid_taps; ++tap_order) {
        const long long tap_index = phase + tap_order * num_channels;
        const double tap = prototype[tap_index];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long output_index = thread_output_base + j;
            if (output_index < num_outputs) {
                const long long tile_index =
                    static_cast<long long>(threadIdx.x) * outputs_per_thread
                    + j + valid_taps - 1 - tap_order;
                const float2 sample = tile[tile_index];
                acc_re[j] += static_cast<double>(sample.x) * tap;
                acc_im[j] += static_cast<double>(sample.y) * tap;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const long long output_index = thread_output_base + j;
        if (output_index < num_outputs) {
            polyphase[phase * num_outputs + output_index] =
                make_float2(static_cast<float>(acc_re[j]), static_cast<float>(acc_im[j]));
        }
    }
}
"""
    _PFB_POLYPHASE_MULTI_I16_KERNEL = cp.RawKernel(
        kernel_code, "pfb_polyphase_multi_i16"
    )
    return _PFB_POLYPHASE_MULTI_I16_KERNEL


def _get_pfb_polyphase_multi_float_kernel():
    global _PFB_POLYPHASE_MULTI_FLOAT_KERNEL
    if _PFB_POLYPHASE_MULTI_FLOAT_KERNEL is not None:
        return _PFB_POLYPHASE_MULTI_FLOAT_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_multi_float(
    const float2* iq,
    const float* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps,
    const long long taps_per_phase
) {
    extern __shared__ float2 tile[];
    const long long outputs_per_thread = 4;
    const long long block_outputs = static_cast<long long>(blockDim.x) * outputs_per_thread;
    const long long block_output_start = static_cast<long long>(blockIdx.x) * block_outputs;
    const long long phase = blockIdx.y;
    if (phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long valid_taps =
        (numtaps > phase) ? ((numtaps - phase + num_channels - 1) / num_channels) : 0;
    const long long tile_samples = block_outputs + valid_taps - 1;
    const long long tile_source_start = block_output_start + source_offset - valid_taps + 1;

    for (long long local = threadIdx.x; local < tile_samples; local += blockDim.x) {
        const long long source_sample_index = tile_source_start + local;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index >= 0 && iq_index < iq_size) {
            tile[local] = iq[iq_index];
        } else {
            tile[local] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    const long long thread_output_base =
        block_output_start + static_cast<long long>(threadIdx.x) * outputs_per_thread;
    float acc_re[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_im[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (long long tap_order = 0; tap_order < valid_taps; ++tap_order) {
        const long long tap_index = phase + tap_order * num_channels;
        const float tap = prototype[tap_index];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long output_index = thread_output_base + j;
            if (output_index < num_outputs) {
                const long long tile_index =
                    static_cast<long long>(threadIdx.x) * outputs_per_thread
                    + j + valid_taps - 1 - tap_order;
                const float2 sample = tile[tile_index];
                acc_re[j] += sample.x * tap;
                acc_im[j] += sample.y * tap;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const long long output_index = thread_output_base + j;
        if (output_index < num_outputs) {
            polyphase[phase * num_outputs + output_index] =
                make_float2(acc_re[j], acc_im[j]);
        }
    }
}
"""
    _PFB_POLYPHASE_MULTI_FLOAT_KERNEL = cp.RawKernel(
        kernel_code, "pfb_polyphase_multi_float"
    )
    return _PFB_POLYPHASE_MULTI_FLOAT_KERNEL


def _get_pfb_polyphase_multi_float_i16_kernel():
    global _PFB_POLYPHASE_MULTI_FLOAT_I16_KERNEL
    if _PFB_POLYPHASE_MULTI_FLOAT_I16_KERNEL is not None:
        return _PFB_POLYPHASE_MULTI_FLOAT_I16_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_multi_float_i16(
    const short* raw_iq,
    const float* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps,
    const long long taps_per_phase
) {
    extern __shared__ float2 tile[];
    const long long outputs_per_thread = 4;
    const long long block_outputs = static_cast<long long>(blockDim.x) * outputs_per_thread;
    const long long block_output_start = static_cast<long long>(blockIdx.x) * block_outputs;
    const long long phase = blockIdx.y;
    if (phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long valid_taps =
        (numtaps > phase) ? ((numtaps - phase + num_channels - 1) / num_channels) : 0;
    const long long tile_samples = block_outputs + valid_taps - 1;
    const long long tile_source_start = block_output_start + source_offset - valid_taps + 1;

    for (long long local = threadIdx.x; local < tile_samples; local += blockDim.x) {
        const long long source_sample_index = tile_source_start + local;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index >= 0 && iq_index < iq_size) {
            const long long raw_index = iq_index * 2;
            tile[local] = make_float2(
                static_cast<float>(raw_iq[raw_index]) * (1.0f / 32768.0f),
                static_cast<float>(raw_iq[raw_index + 1]) * (1.0f / 32768.0f)
            );
        } else {
            tile[local] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    const long long thread_output_base =
        block_output_start + static_cast<long long>(threadIdx.x) * outputs_per_thread;
    float acc_re[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_im[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (long long tap_order = 0; tap_order < valid_taps; ++tap_order) {
        const long long tap_index = phase + tap_order * num_channels;
        const float tap = prototype[tap_index];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long output_index = thread_output_base + j;
            if (output_index < num_outputs) {
                const long long tile_index =
                    static_cast<long long>(threadIdx.x) * outputs_per_thread
                    + j + valid_taps - 1 - tap_order;
                const float2 sample = tile[tile_index];
                acc_re[j] += sample.x * tap;
                acc_im[j] += sample.y * tap;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const long long output_index = thread_output_base + j;
        if (output_index < num_outputs) {
            polyphase[phase * num_outputs + output_index] =
                make_float2(acc_re[j], acc_im[j]);
        }
    }
}
"""
    _PFB_POLYPHASE_MULTI_FLOAT_I16_KERNEL = cp.RawKernel(
        kernel_code, "pfb_polyphase_multi_float_i16"
    )
    return _PFB_POLYPHASE_MULTI_FLOAT_I16_KERNEL


def _get_pfb_polyphase_multi_float_transposed_kernel():
    global _PFB_POLYPHASE_MULTI_FLOAT_TRANSPOSED_KERNEL
    if _PFB_POLYPHASE_MULTI_FLOAT_TRANSPOSED_KERNEL is not None:
        return _PFB_POLYPHASE_MULTI_FLOAT_TRANSPOSED_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_multi_float_transposed(
    const float2* iq,
    const float* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps,
    const long long taps_per_phase
) {
    extern __shared__ float2 tile[];
    const long long outputs_per_thread = 4;
    const long long block_outputs = static_cast<long long>(blockDim.x) * outputs_per_thread;
    const long long block_output_start = static_cast<long long>(blockIdx.x) * block_outputs;
    const long long phase = blockIdx.y;
    if (phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long valid_taps =
        (numtaps > phase) ? ((numtaps - phase + num_channels - 1) / num_channels) : 0;
    const long long tile_samples = block_outputs + valid_taps - 1;
    const long long tile_source_start = block_output_start + source_offset - valid_taps + 1;

    for (long long local = threadIdx.x; local < tile_samples; local += blockDim.x) {
        const long long source_sample_index = tile_source_start + local;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index >= 0 && iq_index < iq_size) {
            tile[local] = iq[iq_index];
        } else {
            tile[local] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    const long long thread_output_base =
        block_output_start + static_cast<long long>(threadIdx.x) * outputs_per_thread;
    float acc_re[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_im[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (long long tap_order = 0; tap_order < valid_taps; ++tap_order) {
        const long long tap_index = phase + tap_order * num_channels;
        const float tap = prototype[tap_index];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long output_index = thread_output_base + j;
            if (output_index < num_outputs) {
                const long long tile_index =
                    static_cast<long long>(threadIdx.x) * outputs_per_thread
                    + j + valid_taps - 1 - tap_order;
                const float2 sample = tile[tile_index];
                acc_re[j] += sample.x * tap;
                acc_im[j] += sample.y * tap;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const long long output_index = thread_output_base + j;
        if (output_index < num_outputs) {
            polyphase[output_index * num_channels + phase] =
                make_float2(acc_re[j], acc_im[j]);
        }
    }
}
"""
    _PFB_POLYPHASE_MULTI_FLOAT_TRANSPOSED_KERNEL = cp.RawKernel(
        kernel_code, "pfb_polyphase_multi_float_transposed"
    )
    return _PFB_POLYPHASE_MULTI_FLOAT_TRANSPOSED_KERNEL


def _get_pfb_polyphase_multi_float_i16_transposed_kernel():
    global _PFB_POLYPHASE_MULTI_FLOAT_I16_TRANSPOSED_KERNEL
    if _PFB_POLYPHASE_MULTI_FLOAT_I16_TRANSPOSED_KERNEL is not None:
        return _PFB_POLYPHASE_MULTI_FLOAT_I16_TRANSPOSED_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_polyphase_multi_float_i16_transposed(
    const short* raw_iq,
    const float* prototype,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps,
    const long long taps_per_phase
) {
    extern __shared__ float2 tile[];
    const long long outputs_per_thread = 4;
    const long long block_outputs = static_cast<long long>(blockDim.x) * outputs_per_thread;
    const long long block_output_start = static_cast<long long>(blockIdx.x) * block_outputs;
    const long long phase = blockIdx.y;
    if (phase >= num_channels) {
        return;
    }

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {
        source_offset = alignment_minus_phase / num_channels;
    } else {
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {
        source_start += num_channels;
    }

    const long long valid_taps =
        (numtaps > phase) ? ((numtaps - phase + num_channels - 1) / num_channels) : 0;
    const long long tile_samples = block_outputs + valid_taps - 1;
    const long long tile_source_start = block_output_start + source_offset - valid_taps + 1;

    for (long long local = threadIdx.x; local < tile_samples; local += blockDim.x) {
        const long long source_sample_index = tile_source_start + local;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index >= 0 && iq_index < iq_size) {
            const long long raw_index = iq_index * 2;
            tile[local] = make_float2(
                static_cast<float>(raw_iq[raw_index]) * (1.0f / 32768.0f),
                static_cast<float>(raw_iq[raw_index + 1]) * (1.0f / 32768.0f)
            );
        } else {
            tile[local] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    const long long thread_output_base =
        block_output_start + static_cast<long long>(threadIdx.x) * outputs_per_thread;
    float acc_re[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_im[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (long long tap_order = 0; tap_order < valid_taps; ++tap_order) {
        const long long tap_index = phase + tap_order * num_channels;
        const float tap = prototype[tap_index];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long output_index = thread_output_base + j;
            if (output_index < num_outputs) {
                const long long tile_index =
                    static_cast<long long>(threadIdx.x) * outputs_per_thread
                    + j + valid_taps - 1 - tap_order;
                const float2 sample = tile[tile_index];
                acc_re[j] += sample.x * tap;
                acc_im[j] += sample.y * tap;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const long long output_index = thread_output_base + j;
        if (output_index < num_outputs) {
            polyphase[output_index * num_channels + phase] =
                make_float2(acc_re[j], acc_im[j]);
        }
    }
}
"""
    _PFB_POLYPHASE_MULTI_FLOAT_I16_TRANSPOSED_KERNEL = cp.RawKernel(
        kernel_code, "pfb_polyphase_multi_float_i16_transposed"
    )
    return _PFB_POLYPHASE_MULTI_FLOAT_I16_TRANSPOSED_KERNEL


def _get_pfb_polyphase_const_kernel():
    global _PFB_POLYPHASE_CONST_MODULE, _PFB_POLYPHASE_CONST_KERNEL
    if _PFB_POLYPHASE_CONST_KERNEL is not None:
        return _PFB_POLYPHASE_CONST_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = rf"""
__constant__ double pfb_prototype_const[{_MAX_PFB_CONST_TAPS}];

extern "C" __global__
void pfb_polyphase_direct_const(
    const float2* iq,
    float2* polyphase,
    const long long iq_size,
    const long long num_channels,
    const long long alignment,
    const long long num_outputs,
    const long long numtaps
) {{
    const long long output_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long phase = blockIdx.y;
    if (output_index >= num_outputs || phase >= num_channels) {{
        return;
    }}

    const long long alignment_minus_phase = alignment - phase;
    long long source_offset;
    if (alignment_minus_phase >= 0) {{
        source_offset = alignment_minus_phase / num_channels;
    }} else {{
        source_offset = -((-alignment_minus_phase + num_channels - 1) / num_channels);
    }}

    long long source_start = alignment_minus_phase % num_channels;
    if (source_start < 0) {{
        source_start += num_channels;
    }}

    const long long source_index = output_index + source_offset;
    double acc_re = 0.0;
    double acc_im = 0.0;

    for (long long tap_index = phase, tap_order = 0;
         tap_index < numtaps;
         tap_index += num_channels, ++tap_order) {{
        const long long source_sample_index = source_index - tap_order;
        const long long iq_index = source_start + num_channels * source_sample_index;
        if (source_sample_index >= 0 && iq_index < iq_size) {{
            const float2 sample = iq[iq_index];
            const double tap = pfb_prototype_const[tap_index];
            acc_re += static_cast<double>(sample.x) * tap;
            acc_im += static_cast<double>(sample.y) * tap;
        }}
    }}

    polyphase[phase * num_outputs + output_index] =
        make_float2(static_cast<float>(acc_re), static_cast<float>(acc_im));
}}
"""
    _PFB_POLYPHASE_CONST_MODULE = cp.RawModule(
        code=kernel_code,
        options=("-std=c++11",),
        name_expressions=("pfb_polyphase_direct_const",),
    )
    _PFB_POLYPHASE_CONST_KERNEL = _PFB_POLYPHASE_CONST_MODULE.get_function(
        "pfb_polyphase_direct_const"
    )
    return _PFB_POLYPHASE_CONST_KERNEL


def _load_pfb_const_taps(prototype_host):
    cp = cuda_backend.require_cuda()
    _get_pfb_polyphase_const_kernel()

    device_id = int(cp.cuda.runtime.getDevice())
    taps_bytes = prototype_host.tobytes()
    if _PFB_POLYPHASE_CONST_CACHE.get(device_id) == taps_bytes:
        return

    prototype_ptr = _PFB_POLYPHASE_CONST_MODULE.get_global("pfb_prototype_const")
    prototype_gpu = cp.ndarray(
        (_MAX_PFB_CONST_TAPS,),
        dtype=cp.float64,
        memptr=prototype_ptr,
    )
    prototype_gpu[: prototype_host.size] = cp.asarray(prototype_host, dtype=cp.float64)
    _PFB_POLYPHASE_CONST_CACHE[device_id] = taps_bytes


def _get_pfb_phase_tables(num_channels, decim, device_id):
    cp = cuda_backend.require_cuda()
    device_key = 0 if device_id is None else int(device_id)
    cache_key = (device_key, int(num_channels), int(decim))
    cached = _PFB_PHASE_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with cuda_backend.use_device(device_id):
        bin_indices = cp.arange(num_channels, dtype=cp.float64)
        signed_bins = _signed_bin_indices(num_channels, cp)
        alignment_phase_table = cp.stack(
            (
                cp.ones((num_channels,), dtype=cp.complex128),
                cp.exp(-1j * 2 * cp.pi * bin_indices * decim / num_channels),
            )
        )
        chunk_mods = cp.arange(num_channels, dtype=cp.float64)
        chunk_phase_table = cp.exp(
            -1j
            * 2
            * cp.pi
            * signed_bins[None, :]
            * chunk_mods[:, None]
            / num_channels
        ).astype(cp.complex64, copy=False)

    cached = (alignment_phase_table, chunk_phase_table)
    _PFB_PHASE_TABLE_CACHE[cache_key] = cached
    return cached


def _get_pfb_combined_phase_table(num_channels, decim, device_id):
    cp = cuda_backend.require_cuda()
    device_key = 0 if device_id is None else int(device_id)
    cache_key = (device_key, int(num_channels), int(decim))
    cached = _PFB_COMBINED_PHASE_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    alignment_phase_table, chunk_phase_table = _get_pfb_phase_tables(
        num_channels, decim, device_id
    )
    with cuda_backend.use_device(device_id):
        combined = (
            alignment_phase_table[:, None, :] * chunk_phase_table[None, :, :]
        ).astype(cp.complex64, copy=False)
        combined = cp.ascontiguousarray(combined)

    _PFB_COMBINED_PHASE_TABLE_CACHE[cache_key] = combined
    return combined


def _get_pfb_phase_apply_kernel():
    global _PFB_PHASE_APPLY_KERNEL
    if _PFB_PHASE_APPLY_KERNEL is not None:
        return _PFB_PHASE_APPLY_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_apply_phase_scatter(
    const float2* aligned,
    const float2* phase_factors,
    float2* channels,
    const long long num_channels,
    const long long num_outputs,
    const long long num_aligned_outputs,
    const long long output_parity
) {
    const long long aligned_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long phase = blockIdx.y;
    if (aligned_index >= num_aligned_outputs || phase >= num_channels) {
        return;
    }

    const long long output_index = output_parity + aligned_index * 2;
    const float2 sample = aligned[phase * num_aligned_outputs + aligned_index];
    const float2 factor = phase_factors[phase];
    const float out_re = sample.x * factor.x - sample.y * factor.y;
    const float out_im = sample.x * factor.y + sample.y * factor.x;

    channels[phase * num_outputs + output_index] =
        make_float2(out_re, out_im);
}
"""
    _PFB_PHASE_APPLY_KERNEL = cp.RawKernel(kernel_code, "pfb_apply_phase_scatter")
    return _PFB_PHASE_APPLY_KERNEL


def _get_pfb_phase_apply_transposed_kernel():
    global _PFB_PHASE_APPLY_TRANSPOSED_KERNEL
    if _PFB_PHASE_APPLY_TRANSPOSED_KERNEL is not None:
        return _PFB_PHASE_APPLY_TRANSPOSED_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void pfb_apply_phase_scatter_transposed(
    const float2* aligned,
    const float2* phase_factors,
    float2* channels,
    const long long num_channels,
    const long long num_outputs,
    const long long num_aligned_outputs,
    const long long output_parity
) {
    const long long aligned_index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long phase = blockIdx.y;
    if (aligned_index >= num_aligned_outputs || phase >= num_channels) {
        return;
    }

    const long long output_index = output_parity + aligned_index * 2;
    const float2 sample = aligned[aligned_index * num_channels + phase];
    const float2 factor = phase_factors[phase];
    const float out_re = sample.x * factor.x - sample.y * factor.y;
    const float out_im = sample.x * factor.y + sample.y * factor.x;

    channels[phase * num_outputs + output_index] =
        make_float2(out_re, out_im);
}
"""
    _PFB_PHASE_APPLY_TRANSPOSED_KERNEL = cp.RawKernel(
        kernel_code, "pfb_apply_phase_scatter_transposed"
    )
    return _PFB_PHASE_APPLY_TRANSPOSED_KERNEL


def _get_cleanup_fir_kernel():
    global _CLEANUP_FIR_KERNEL
    if _CLEANUP_FIR_KERNEL is not None:
        return _CLEANUP_FIR_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void complex64_fir_causal(
    const float2* x,
    const double* taps,
    float2* y,
    const long long rows,
    const long long cols,
    const long long ntaps
) {
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = rows * cols;
    if (idx >= total) {
        return;
    }

    const long long row = idx / cols;
    const long long col = idx - row * cols;
    const long long base = row * cols;
    double acc_re = 0.0;
    double acc_im = 0.0;

    const long long max_k = col + 1 < ntaps ? col + 1 : ntaps;
    for (long long k = 0; k < max_k; ++k) {
        const float2 sample = x[base + col - k];
        const double tap = taps[k];
        acc_re += static_cast<double>(sample.x) * tap;
        acc_im += static_cast<double>(sample.y) * tap;
    }

    y[idx] = make_float2(static_cast<float>(acc_re), static_cast<float>(acc_im));
}
"""
    _CLEANUP_FIR_KERNEL = cp.RawKernel(kernel_code, "complex64_fir_causal")
    return _CLEANUP_FIR_KERNEL


def _get_cleanup_fir_shared_kernel():
    global _CLEANUP_FIR_SHARED_KERNEL
    if _CLEANUP_FIR_SHARED_KERNEL is not None:
        return _CLEANUP_FIR_SHARED_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = r"""
extern "C" __global__
void complex64_fir_causal_shared(
    const float2* x,
    const double* taps,
    float2* y,
    const long long cols,
    const long long ntaps
) {
    extern __shared__ float2 tile[];

    const long long row = blockIdx.y;
    const long long col_base = static_cast<long long>(blockIdx.x) * blockDim.x;
    const long long tid = threadIdx.x;
    const long long row_base = row * cols;
    const long long halo = ntaps - 1;

    const long long tile_len = static_cast<long long>(blockDim.x) + halo;
    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < cols) {
            tile[tile_idx] = x[row_base + source_col];
        } else {
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    const long long col = col_base + tid;
    if (col >= cols) {
        return;
    }

    double acc_re = 0.0;
    double acc_im = 0.0;
    const long long max_k = col + 1 < ntaps ? col + 1 : ntaps;
    const long long sample_offset = halo + tid;
    for (long long k = 0; k < max_k; ++k) {
        const float2 sample = tile[sample_offset - k];
        const double tap = taps[k];
        acc_re += static_cast<double>(sample.x) * tap;
        acc_im += static_cast<double>(sample.y) * tap;
    }

    y[row_base + col] = make_float2(static_cast<float>(acc_re), static_cast<float>(acc_im));
}
"""
    _CLEANUP_FIR_SHARED_KERNEL = cp.RawKernel(
        kernel_code, "complex64_fir_causal_shared"
    )
    return _CLEANUP_FIR_SHARED_KERNEL


def _get_cleanup_fir_const_kernel():
    global _CLEANUP_FIR_CONST_MODULE, _CLEANUP_FIR_CONST_KERNEL
    if _CLEANUP_FIR_CONST_KERNEL is not None:
        return _CLEANUP_FIR_CONST_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = rf"""
__constant__ double cleanup_taps_const[{_MAX_CLEANUP_CONST_TAPS}];

extern "C" __global__
void complex64_fir_causal_const(
    const float2* x,
    float2* y,
    const long long rows,
    const long long cols,
    const long long ntaps
) {{
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = rows * cols;
    if (idx >= total) {{
        return;
    }}

    const long long row = idx / cols;
    const long long col = idx - row * cols;
    const long long base = row * cols;
    double acc_re = 0.0;
    double acc_im = 0.0;

    const long long max_k = col + 1 < ntaps ? col + 1 : ntaps;
    for (long long k = 0; k < max_k; ++k) {{
        const float2 sample = x[base + col - k];
        const double tap = cleanup_taps_const[k];
        acc_re += static_cast<double>(sample.x) * tap;
        acc_im += static_cast<double>(sample.y) * tap;
    }}

    y[idx] = make_float2(static_cast<float>(acc_re), static_cast<float>(acc_im));
}}
"""
    _CLEANUP_FIR_CONST_MODULE = cp.RawModule(
        code=kernel_code,
        options=("-std=c++11",),
        name_expressions=("complex64_fir_causal_const",),
    )
    _CLEANUP_FIR_CONST_KERNEL = _CLEANUP_FIR_CONST_MODULE.get_function(
        "complex64_fir_causal_const"
    )
    return _CLEANUP_FIR_CONST_KERNEL


def _get_cleanup_fir_multi_kernel():
    global _CLEANUP_FIR_MULTI_MODULE, _CLEANUP_FIR_MULTI_KERNEL
    if _CLEANUP_FIR_MULTI_KERNEL is not None:
        return _CLEANUP_FIR_MULTI_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = rf"""
__constant__ double cleanup_taps_multi_const[{_MAX_CLEANUP_CONST_TAPS}];

extern "C" __global__
void complex64_fir_causal_multi(
    const float2* x,
    float2* y,
    const long long cols,
    const long long ntaps
) {{
    extern __shared__ float2 tile[];

    const long long row = blockIdx.y;
    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long row_base = row * cols;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < cols) {{
            tile[tile_idx] = x[row_base + source_col];
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    double acc0_re = 0.0;
    double acc0_im = 0.0;
    double acc1_re = 0.0;
    double acc1_im = 0.0;
    double acc2_re = 0.0;
    double acc2_im = 0.0;
    double acc3_re = 0.0;
    double acc3_im = 0.0;

    // Four adjacent outputs share the same tap stream and shared-memory window.
    for (long long k = 0; k < ntaps; ++k) {{
        const double tap = cleanup_taps_multi_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += static_cast<double>(sample0.x) * tap;
        acc0_im += static_cast<double>(sample0.y) * tap;
        acc1_re += static_cast<double>(sample1.x) * tap;
        acc1_im += static_cast<double>(sample1.y) * tap;
        acc2_re += static_cast<double>(sample2.x) * tap;
        acc2_im += static_cast<double>(sample2.y) * tap;
        acc3_re += static_cast<double>(sample3.x) * tap;
        acc3_im += static_cast<double>(sample3.y) * tap;
    }}

    if (first_col < cols) {{
        y[row_base + first_col] =
            make_float2(static_cast<float>(acc0_re), static_cast<float>(acc0_im));
    }}
    if (first_col + 1 < cols) {{
        y[row_base + first_col + 1] =
            make_float2(static_cast<float>(acc1_re), static_cast<float>(acc1_im));
    }}
    if (first_col + 2 < cols) {{
        y[row_base + first_col + 2] =
            make_float2(static_cast<float>(acc2_re), static_cast<float>(acc2_im));
    }}
    if (first_col + 3 < cols) {{
        y[row_base + first_col + 3] =
            make_float2(static_cast<float>(acc3_re), static_cast<float>(acc3_im));
    }}
}}
"""
    _CLEANUP_FIR_MULTI_MODULE = cp.RawModule(
        code=kernel_code,
        options=("-std=c++11",),
        name_expressions=("complex64_fir_causal_multi",),
    )
    _CLEANUP_FIR_MULTI_KERNEL = _CLEANUP_FIR_MULTI_MODULE.get_function(
        "complex64_fir_causal_multi"
    )
    return _CLEANUP_FIR_MULTI_KERNEL


def _get_cleanup_fir_multi_float_kernel():
    global _CLEANUP_FIR_MULTI_FLOAT_MODULE, _CLEANUP_FIR_MULTI_FLOAT_KERNEL
    if _CLEANUP_FIR_MULTI_FLOAT_KERNEL is not None:
        return _CLEANUP_FIR_MULTI_FLOAT_KERNEL

    cp = cuda_backend.require_cuda()
    kernel_code = rf"""
__constant__ float cleanup_taps_multi_float_const[{_MAX_CLEANUP_CONST_TAPS}];

extern "C" __global__
void complex64_fir_causal_multi_float(
    const float2* x,
    float2* y,
    const long long cols,
    const long long ntaps
) {{
    extern __shared__ float2 tile[];

    const long long row = blockIdx.y;
    const long long tid = threadIdx.x;
    const long long halo = ntaps - 1;
    const long long outputs_per_thread = {_CLEANUP_MULTI_OUTPUTS_PER_THREAD};
    const long long col_base =
        static_cast<long long>(blockIdx.x) * blockDim.x * outputs_per_thread;
    const long long row_base = row * cols;
    const long long tile_len = static_cast<long long>(blockDim.x) * outputs_per_thread + halo;

    for (long long tile_idx = tid; tile_idx < tile_len; tile_idx += blockDim.x) {{
        const long long source_col = col_base + tile_idx - halo;
        if (source_col >= 0 && source_col < cols) {{
            tile[tile_idx] = x[row_base + source_col];
        }} else {{
            tile[tile_idx] = make_float2(0.0f, 0.0f);
        }}
    }}
    __syncthreads();

    const long long first_col = col_base + tid * outputs_per_thread;
    const long long first_sample_offset = halo + tid * outputs_per_thread;

    float acc0_re = 0.0f;
    float acc0_im = 0.0f;
    float acc1_re = 0.0f;
    float acc1_im = 0.0f;
    float acc2_re = 0.0f;
    float acc2_im = 0.0f;
    float acc3_re = 0.0f;
    float acc3_im = 0.0f;

    for (long long k = 0; k < ntaps; ++k) {{
        const float tap = cleanup_taps_multi_float_const[k];
        const float2 sample0 = tile[first_sample_offset - k];
        const float2 sample1 = tile[first_sample_offset + 1 - k];
        const float2 sample2 = tile[first_sample_offset + 2 - k];
        const float2 sample3 = tile[first_sample_offset + 3 - k];
        acc0_re += sample0.x * tap;
        acc0_im += sample0.y * tap;
        acc1_re += sample1.x * tap;
        acc1_im += sample1.y * tap;
        acc2_re += sample2.x * tap;
        acc2_im += sample2.y * tap;
        acc3_re += sample3.x * tap;
        acc3_im += sample3.y * tap;
    }}

    if (first_col < cols) {{
        y[row_base + first_col] = make_float2(acc0_re, acc0_im);
    }}
    if (first_col + 1 < cols) {{
        y[row_base + first_col + 1] = make_float2(acc1_re, acc1_im);
    }}
    if (first_col + 2 < cols) {{
        y[row_base + first_col + 2] = make_float2(acc2_re, acc2_im);
    }}
    if (first_col + 3 < cols) {{
        y[row_base + first_col + 3] = make_float2(acc3_re, acc3_im);
    }}
}}
"""
    _CLEANUP_FIR_MULTI_FLOAT_MODULE = cp.RawModule(
        code=kernel_code,
        options=("-std=c++11",),
        name_expressions=("complex64_fir_causal_multi_float",),
    )
    _CLEANUP_FIR_MULTI_FLOAT_KERNEL = (
        _CLEANUP_FIR_MULTI_FLOAT_MODULE.get_function(
            "complex64_fir_causal_multi_float"
        )
    )
    return _CLEANUP_FIR_MULTI_FLOAT_KERNEL


def _load_cleanup_fir_const_taps(cleanup_lpf_host):
    cp = cuda_backend.require_cuda()
    _get_cleanup_fir_const_kernel()

    device_id = int(cp.cuda.runtime.getDevice())
    taps_bytes = cleanup_lpf_host.tobytes()
    if _CLEANUP_FIR_CONST_CACHE.get(device_id) == taps_bytes:
        return

    taps_ptr = _CLEANUP_FIR_CONST_MODULE.get_global("cleanup_taps_const")
    taps_gpu = cp.ndarray(
        (_MAX_CLEANUP_CONST_TAPS,),
        dtype=cp.float64,
        memptr=taps_ptr,
    )
    taps_gpu[: cleanup_lpf_host.size] = cp.asarray(cleanup_lpf_host, dtype=cp.float64)
    _CLEANUP_FIR_CONST_CACHE[device_id] = taps_bytes


def _load_cleanup_fir_multi_taps(cleanup_lpf_host):
    cp = cuda_backend.require_cuda()
    _get_cleanup_fir_multi_kernel()

    device_id = int(cp.cuda.runtime.getDevice())
    taps_bytes = cleanup_lpf_host.tobytes()
    if _CLEANUP_FIR_MULTI_CACHE.get(device_id) == taps_bytes:
        return

    taps_ptr = _CLEANUP_FIR_MULTI_MODULE.get_global("cleanup_taps_multi_const")
    taps_gpu = cp.ndarray(
        (_MAX_CLEANUP_CONST_TAPS,),
        dtype=cp.float64,
        memptr=taps_ptr,
    )
    taps_gpu[: cleanup_lpf_host.size] = cp.asarray(cleanup_lpf_host, dtype=cp.float64)
    _CLEANUP_FIR_MULTI_CACHE[device_id] = taps_bytes


def _load_cleanup_fir_multi_float_taps(cleanup_lpf_host):
    cp = cuda_backend.require_cuda()
    _get_cleanup_fir_multi_float_kernel()

    device_id = int(cp.cuda.runtime.getDevice())
    taps_bytes = cleanup_lpf_host.tobytes()
    if _CLEANUP_FIR_MULTI_FLOAT_CACHE.get(device_id) == taps_bytes:
        return

    taps_ptr = _CLEANUP_FIR_MULTI_FLOAT_MODULE.get_global(
        "cleanup_taps_multi_float_const"
    )
    taps_gpu = cp.ndarray(
        (_MAX_CLEANUP_CONST_TAPS,),
        dtype=cp.float32,
        memptr=taps_ptr,
    )
    taps_gpu[: cleanup_lpf_host.size] = cp.asarray(
        cleanup_lpf_host, dtype=cp.float32
    )
    _CLEANUP_FIR_MULTI_FLOAT_CACHE[device_id] = taps_bytes


def _signed_bin_indices(num_channels, xp=np):
    indices = xp.arange(num_channels, dtype=xp.float64)
    return xp.where(indices <= (num_channels - 1) // 2, indices, indices - num_channels)
