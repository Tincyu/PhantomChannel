from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter


def bytes_per_wideband_sample(iq_format):
    if iq_format == "complex64":
        return np.dtype(np.complex64).itemsize
    if iq_format == "int16":
        return 2 * np.dtype(np.int16).itemsize
    raise ValueError(f"Unsupported iq_format: {iq_format}")


def iter_iq_chunks(
    input_bin, iq_format, chunk_samples, overlap_samples, decode_int16=True
):
    path = Path(input_bin)
    if iq_format == "complex64":
        raw = np.memmap(path, dtype=np.complex64, mode="r")
        total_samples = raw.size
        decode = lambda arr: np.asarray(arr, dtype=np.complex64)
    elif iq_format == "int16":
        raw_i16 = np.memmap(path, dtype=np.int16, mode="r")
        total_samples = raw_i16.size // 2

        def decode_i16(arr):
            arr = np.asarray(arr)
            arr = arr[: arr.size - (arr.size % 2)]
            iq = np.empty(arr.size // 2, dtype=np.complex64)
            iq.real = arr[0::2]
            iq.imag = arr[1::2]
            iq.real *= np.float32(1.0 / 32768.0)
            iq.imag *= np.float32(1.0 / 32768.0)
            return iq

        raw = raw_i16
        decode = decode_i16 if decode_int16 else np.asarray
    else:
        raise ValueError(f"Unsupported iq_format: {iq_format}")

    core_start = 0
    while core_start < total_samples:
        core_end = min(total_samples, core_start + chunk_samples)
        read_start = max(0, core_start - overlap_samples)
        read_end = min(total_samples, core_end + overlap_samples)
        if iq_format == "complex64":
            chunk = decode(raw[read_start:read_end])
        else:
            chunk = decode(raw[read_start * 2 : read_end * 2])
        yield read_start, core_start, core_end, chunk
        core_start = core_end


def target_freqs_mhz(start_mhz, stop_mhz, step_mhz, center_freq_hz, bandwidth_hz, sub_bw_hz):
    low = center_freq_hz - bandwidth_hz / 2 + sub_bw_hz / 2
    high = center_freq_hz + bandwidth_hz / 2 - sub_bw_hz / 2
    freqs = []
    value = start_mhz
    while value <= stop_mhz + 1e-9:
        freq_hz = value * 1e6
        if low <= freq_hz <= high:
            freqs.append(value)
        value += step_mhz
    return freqs


def ble_target_freqs_mhz(center_freq_hz, bandwidth_hz):
    # BLE 2 MHz channel centers. For 40 MHz centered at 2420 MHz, this yields 2402..2438 MHz.
    return target_freqs_mhz(2402.0, 2480.0, 2.0, center_freq_hz, bandwidth_hz, sub_bw_hz=2e6)


def bredr_target_freqs_mhz(center_freq_hz, bandwidth_hz):
    # BR/EDR 1 MHz channel centers, channel number = freq_mhz - 2402.
    return target_freqs_mhz(2402.0, 2480.0, 1.0, center_freq_hz, bandwidth_hz, sub_bw_hz=1e6)


def design_channel_lpf(sample_rate, cutoff_hz, numtaps):
    return firwin(numtaps, cutoff_hz, fs=sample_rate)


def channelize_to_4m(iq, chunk_start_index, sample_rate, center_freq_hz, target_freq_mhz, decim, lpf):
    target_freq_hz = target_freq_mhz * 1e6
    center_offset_hz = target_freq_hz - center_freq_hz
    # Use global sample index in the mixer so chunk boundaries do not change phase.
    t = (chunk_start_index + np.arange(iq.size, dtype=np.float64)) / sample_rate
    shifted = iq * np.exp(-1j * 2 * np.pi * center_offset_hz * t)
    filtered = lfilter(lpf, 1.0, shifted)
    return filtered[::decim].astype(np.complex64, copy=False)


def wideband_index_from_subband(chunk_start_index, subband_sample_index, decim, filter_delay_samples):
    return float(chunk_start_index) + float(subband_sample_index) * float(decim) - float(filter_delay_samples)


def timestamp_us_from_wideband_index(wideband_sample_index, sample_rate):
    return float(wideband_sample_index) * 1e6 / float(sample_rate)
