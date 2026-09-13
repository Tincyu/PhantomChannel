from __future__ import annotations

import numpy as np

from tools.bpl_sync import (
    BPLConfig,
    ble_preamble_bits,
    generate_gfsk_template,
    localize_preamble,
)


def embedded_template(config: BPLConfig, *, polarity: int = 0, offset: int = 37) -> np.ndarray:
    template = generate_gfsk_template(ble_preamble_bits(config.phy, polarity=polarity), config)
    signal = np.zeros(offset + template.size + 80, dtype=np.complex64)
    signal[offset : offset + template.size] = template
    return signal


def test_template_bit_order_and_polarity() -> None:
    config = BPLConfig()
    assert ble_preamble_bits("LE_1M", polarity=0) == [0] * 8
    assert ble_preamble_bits("LE_1M", polarity=1) == [1] * 8
    assert len(ble_preamble_bits("LE_2M", polarity=0)) == 16
    assert not np.allclose(
        generate_gfsk_template([0] * 8, config),
        generate_gfsk_template([1] * 8, config),
    )


def test_1m_integer_delay_recovery() -> None:
    config = BPLConfig(min_normalized_peak=0.8, min_peak_ratio=1.01)
    result = localize_preamble(embedded_template(config, offset=37), config)
    assert result["accepted"]
    assert abs(float(result["packet_start_sample"]) - 37.0) <= 1.0


def test_search_bounds_are_in_input_coordinates() -> None:
    config = BPLConfig(min_normalized_peak=0.8, min_peak_ratio=1.01)
    signal = embedded_template(config, offset=19)
    outside = embedded_template(config, offset=110)
    combined = np.zeros(max(signal.size, outside.size), dtype=np.complex64)
    combined[: signal.size] += signal
    combined[: outside.size] += outside
    result = localize_preamble(combined, config, search_start_sample=0, search_end_sample=80)
    assert result["accepted"]
    assert 18.0 <= float(result["packet_start_sample"]) <= 20.0
    assert result["search_start_sample"] == 0
    assert result["search_end_sample"] == 80


def test_2m_integer_delay_recovery() -> None:
    config = BPLConfig(phy="LE_2M", max_peak_width_samples=96, min_normalized_peak=0.8, min_peak_ratio=1.01)
    result = localize_preamble(embedded_template(config, offset=19), config)
    assert result["accepted"]
    assert abs(float(result["packet_start_sample"]) - 19.0) <= 1.0


def test_fractional_delay_recovery() -> None:
    config = BPLConfig(min_normalized_peak=0.75, min_peak_ratio=1.01)
    original = embedded_template(config, offset=37)
    delay = 0.35
    indices = np.arange(original.size, dtype=np.float64)
    shifted = np.interp(indices - delay, indices, original.real, left=0.0, right=0.0)
    shifted = shifted + 1j * np.interp(indices - delay, indices, original.imag, left=0.0, right=0.0)
    result = localize_preamble(shifted.astype(np.complex64), config)
    assert result["accepted"]
    assert abs(float(result["packet_start_sample"]) - (37.0 + delay)) <= 2.0


def test_initial_phase_invariance() -> None:
    config = BPLConfig(min_normalized_peak=0.8, min_peak_ratio=1.01)
    signal = embedded_template(config, offset=31)
    phase = np.exp(1j * 1.2345).astype(np.complex64)
    left = localize_preamble(signal, config)
    right = localize_preamble(signal * phase, config)
    assert left["accepted"] and right["accepted"]
    assert abs(left["normalized_peak"] - right["normalized_peak"]) < 1e-5
    assert abs(float(left["packet_start_sample"]) - float(right["packet_start_sample"])) <= 1.0


def test_cfo_bank_recovery() -> None:
    config = BPLConfig(min_normalized_peak=0.7, min_peak_ratio=1.01, cfo_search_hz=(-50_000.0, 0.0, 50_000.0))
    base = embedded_template(config, offset=21)
    indices = np.arange(base.size, dtype=np.float32)
    active = base[21 : 21 + 32] * np.exp(2j * np.pi * 50_000.0 * indices[:32] / config.sample_rate_sps)
    signal = base.copy()
    signal[21 : 21 + 32] = active
    result = localize_preamble(signal, config)
    assert result["accepted"]
    assert result["coarse_cfo_hz"] == 50_000.0


def test_multipath_peak_metrics_and_competing_peak() -> None:
    config = BPLConfig(min_normalized_peak=0.1, min_peak_ratio=1.01, max_peak_width_samples=999)
    template = generate_gfsk_template(ble_preamble_bits(config.phy, polarity=0), config)
    single = np.zeros(220, dtype=np.complex64)
    single[35 : 35 + template.size] = template
    signal = np.zeros(220, dtype=np.complex64)
    signal[35 : 35 + template.size] += template
    signal[35 + 18 : 35 + 18 + template.size] += 0.8 * template
    single_result = localize_preamble(single, config)
    result = localize_preamble(signal, config)
    assert result["peak_width_samples"] > single_result["peak_width_samples"]
    assert result["second_peak"] >= 0.0
    assert result["competing_peak_count"] >= 1


def test_competing_peak_metric_is_nonzero_for_equal_paths() -> None:
    config = BPLConfig(min_normalized_peak=0.1, min_peak_ratio=1.01, max_peak_width_samples=999)
    template = generate_gfsk_template(ble_preamble_bits(config.phy, polarity=0), config)
    signal = np.zeros(240, dtype=np.complex64)
    signal[30 : 30 + template.size] += template
    signal[62 : 62 + template.size] += template
    result = localize_preamble(signal, config)
    assert result["competing_peak_count"] >= 1
    assert result["peak_to_second_ratio"] < 1.1


def test_noise_false_accept_is_quantified() -> None:
    config = BPLConfig(min_normalized_peak=0.75, min_peak_ratio=1.5)
    rng = np.random.default_rng(20260809)
    result = localize_preamble((rng.normal(size=400) + 1j * rng.normal(size=400)).astype(np.complex64), config)
    assert not result["accepted"]
    assert result["failure_code"] in {"B1_LOW_CORRELATION", "B2_AMBIGUOUS_PEAKS", "B3_BROAD_PEAK"}
