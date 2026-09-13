from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tools.two_stage_known_aa_parse as parser_module
from tools.two_stage_known_aa_parse import (
    add_stage1_fields,
    deduplicate_rows,
    expand_stage2_known_access_addresses,
    measure_physical_length,
    remap_stage2_row,
    tolerant_access_address_clusters,
)


def connection_row(aa, sample, channel=17, post_crc_hex=""):
    return {
        "packet_type": "BLE_CONN",
        "direction_hint": "connection",
        "ble_pdu_type": "",
        "access_address": aa,
        "wideband_sample_index": str(sample),
        "timestamp_s": str(sample / 100_000_000),
        "channel": str(channel),
        "confidence_score": "100.000",
        "payload_len": "20",
        "post_crc_hex": post_crc_hex,
    }


def test_bootstrap_excludes_advertising_aliases_and_has_no_upper_bound():
    rows = []
    rows.extend(connection_row("0x11223344", 1000 + i * 100) for i in range(10))
    rows.extend(connection_row("0x11223345", 2000 + i * 100) for i in range(2))
    rows.extend(connection_row("0x55667788", 3000 + i * 100) for i in range(10))
    rows.extend(connection_row("0x99AABBCC", 4000 + i * 100) for i in range(10))
    rows.extend(connection_row("0xD6BE898E", 5000 + i * 100) for i in range(30))

    clusters, stats = tolerant_access_address_clusters(
        rows,
        tolerance_bits=2,
        min_count=10,
        excluded_addresses=("0x8E89BED6", "0xD6BE898E"),
    )

    selected = {cluster["canonical_access_address"] for cluster in clusters}
    assert selected == {"0x11223344", "0x55667788", "0x99AABBCC"}
    assert len(clusters) == 3
    assert all(cluster["observation_count"] >= 10 for cluster in clusters)
    assert stats["rejected_advertising_rows"] == 30


def test_stage2_remap_uses_global_sample_and_known_cluster():
    cluster = {
        "cluster_id": "aa:0x11223344",
        "canonical_access_address": "0x11223344",
        "canonical_access_address_hex": "11223344",
    }
    row = connection_row("0x11223345", 100_100, post_crc_hex="AABB")
    result = remap_stage2_row(
        row,
        source_start_sample=199_900_000,
        boundary_sample=200_000_000,
        total_samples=1_000_000_000,
        decim=25,
        sample_rate_sps=100_000_000,
        subband_sample_rate_sps=4_000_000,
        phy_rate_sps=1_000_000,
        known_clusters=[cluster],
        tolerance_bits=2,
    )

    assert result is not None
    assert result["parse_mode"] == "known_aa_stage2"
    assert result["wideband_sample_index"] == 200_000_100
    assert result["aa_hamming_distance"] == 1
    assert result["known_aa_cluster_id"] == "aa:0x11223344"
    assert result["physical_burst_start_sample"] == 200_000_100
    assert result["link_layer_length_bytes"] == 30
    assert result["post_crc_or_tail_length_bytes"] == 2


def test_stage2_known_addresses_include_observed_aliases_and_one_bit_neighbours():
    addresses, stats = expand_stage2_known_access_addresses(
        [
            {
                "canonical_access_address": "0x11223344",
                "aliases": [
                    {"access_address": "0x11223345", "count": 12},
                ],
            }
        ],
        bit_tolerance=1,
        excluded_addresses=("0x8E89BED6", "0xD6BE898E"),
    )

    assert "0x11223344" in addresses
    assert "0x11223345" in addresses
    assert "0x11223346" in addresses
    assert stats["stage1_observed_alias_count"] == 2
    assert stats["stage2_known_address_count"] == 33


def test_stage2_known_address_expansion_does_not_reintroduce_advertising_aa():
    addresses, _stats = expand_stage2_known_access_addresses(
        [
            {
                "canonical_access_address": "0x8E89BED7",
                "aliases": [],
            }
        ],
        bit_tolerance=1,
        excluded_addresses=("0x8E89BED6", "0xD6BE898E"),
    )
    assert "0x8E89BED6" not in addresses


def test_stage2_rows_before_boundary_are_dropped():
    cluster = {
        "cluster_id": "aa:0x11223344",
        "canonical_access_address": "0x11223344",
        "canonical_access_address_hex": "11223344",
    }
    row = connection_row("0x11223344", 10)
    assert (
        remap_stage2_row(
            row,
            source_start_sample=199_900_000,
            boundary_sample=200_000_000,
            total_samples=1_000_000_000,
            decim=25,
            sample_rate_sps=100_000_000,
            subband_sample_rate_sps=4_000_000,
            phy_rate_sps=1_000_000,
            known_clusters=[cluster],
            tolerance_bits=2,
        )
        is None
    )


def test_deduplicate_rows_prefers_higher_confidence():
    first = add_stage1_fields(connection_row("0x11223344", 100))
    second = dict(first)
    first["confidence_score"] = "98"
    second["confidence_score"] = "100"
    result = deduplicate_rows([first, second])
    assert len(result) == 1
    assert result[0]["confidence_score"] == "100"


def test_saturated_iq_duration_is_not_converted_to_tail_bytes():
    row = {
        "physical_burst_start_sample": "100000",
        "channel": "17",
        "subband_freq_mhz": "2440",
        "standard_link_layer_duration_us": "240",
    }
    args = SimpleNamespace(
        iq_path=Path("/tmp/synthetic.sc16"),
        sample_rate_sps=100_000_000.0,
        center_frequency_hz=2_440e6,
        length_expected_max_us=5_000.0,
        length_pre_margin_us=20.0,
        length_post_margin_us=200.0,
        length_lowpass_hz=900_000.0,
        length_smooth_us=1.5,
        length_threshold_sigma=8.0,
        length_min_threshold_ratio=2.0,
        phy_rate_sps=1_000_000.0,
    )
    saturated = {
        "measured_duration_us": 5_000.0,
        "notes": "duration_hits_search_end",
        "threshold": 1.0,
        "duration_confidence": 12.0,
    }
    with patch.object(parser_module, "estimate_burst_duration_us", return_value=saturated):
        result = measure_physical_length(row, args)

    assert result["physical_length_estimation_method"] == "iq_energy_unbounded_search_end"
    assert result["physical_burst_duration_us_observed"] == "5000.000"
    assert result["physical_residual_tail_length_bytes"] == ""
