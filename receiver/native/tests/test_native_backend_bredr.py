import numpy as np

from bt_pipeline.native_backend import (
    build_btclassic_packet_candidates_native,
    parse_btclassic_packet_segments_native_finalized,
    validate_native_parser_backend,
)
from bt_pipeline.parsers import (
    LockedUAPCache,
    build_btclassic_packet_candidates,
    finalize_btclassic_packet_candidates,
)
from bt_pipeline.pfb_channelizer import HostSegmentBatch


def main():
    sample_rate = 4e6
    freq_dev = 250e3
    center_freq = 2402e6
    cutoff = 750e3
    iq = np.zeros(256, dtype=np.complex64)
    segments = [(0, len(iq), iq, 0)]

    python_candidates = build_btclassic_packet_candidates(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        cutoff,
    )
    native_candidates = build_btclassic_packet_candidates_native(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        cutoff,
    )
    native_batch_candidates = build_btclassic_packet_candidates_native(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        cutoff,
        thread_count=2,
        batch_candidates=True,
    )
    compact_segments = HostSegmentBatch.from_segments(segments)
    native_compact_batch_candidates = build_btclassic_packet_candidates_native(
        compact_segments,
        sample_rate,
        center_freq,
        freq_dev,
        cutoff,
        thread_count=2,
        batch_candidates=True,
    )
    assert native_candidates == python_candidates
    assert native_batch_candidates == python_candidates
    assert native_compact_batch_candidates == native_batch_candidates

    expected_packets = finalize_btclassic_packet_candidates(
        native_batch_candidates,
        sample_rate,
        center_freq,
        uap_cache=LockedUAPCache(),
    )
    finalized_packets = parse_btclassic_packet_segments_native_finalized(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        cutoff,
        thread_count=2,
        uap_cache=LockedUAPCache(),
    )
    compact_finalized_packets = parse_btclassic_packet_segments_native_finalized(
        compact_segments,
        sample_rate,
        center_freq,
        freq_dev,
        cutoff,
        thread_count=2,
        uap_cache=LockedUAPCache(),
    )
    assert finalized_packets == expected_packets
    assert compact_finalized_packets == finalized_packets
    assert validate_native_parser_backend("python", "cpp") is not None
    print("PASS: native BR/EDR backend wrapper")


if __name__ == "__main__":
    main()
