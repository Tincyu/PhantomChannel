from tools.audit_advertising_timing_gap import (
    Packet,
    pair_gap_us,
    run_synthetic_checks,
    stats,
    strict_reconstruct,
)


def packet(frame, timestamp_us, channel, length=6):
    return Packet(
        frame=frame,
        timestamp_us=timestamp_us,
        channel=channel,
        packet_counter=None,
        delta_time_us=None,
        delta_time_ss_us=None,
        crc_valid=True,
        length=length,
        phy_mbps=1.0,
        phy_source="test_1M",
    )


def event(start_us, first_gap=293.0, second_gap=288.0):
    p37 = packet(1, start_us, 37)
    p38 = packet(2, start_us + 128.0 + first_gap, 38)
    p39 = packet(3, p38.timestamp_us + 128.0 + second_gap, 39)
    return [p37, p38, p39]


def test_tail_boundary_does_not_enter_gap_feature():
    base_events, boundaries, _ = strict_reconstruct(event(0.0) + event(20_000.0))
    shifted_events, _, _ = strict_reconstruct(event(0.0) + event(21_000.0))
    base_gaps = [pair_gap_us(left, right)[0] for item in base_events for left, right in zip(item.packets, item.packets[1:])]
    shifted_gaps = [pair_gap_us(left, right)[0] for item in shifted_events for left, right in zip(item.packets, item.packets[1:])]
    assert base_gaps == shifted_gaps
    assert len(boundaries) == 1


def test_append_every_changes_both_intra_event_pairs():
    events, _, _ = strict_reconstruct(event(0.0, first_gap=500.0, second_gap=600.0))
    gaps = [pair_gap_us(left, right)[0] for left, right in zip(events[0].packets, events[0].packets[1:])]
    assert gaps == [500.0, 600.0]


def test_intervening_target_packet_is_not_skipped():
    interrupted = event(0.0)[:2] + [packet(4, 5_000.0, 37)] + event(10_000.0)
    events, _, _ = strict_reconstruct(interrupted)
    assert len(events) == 1
    assert events[0].packets[0].timestamp_us == 10_000.0


def test_delta_start_to_start_minus_airtime_is_gap():
    first, second = event(0.0)[:2]
    gap, start_to_start = pair_gap_us(first, second)
    assert start_to_start == 421.0
    assert gap == 293.0


def test_mad_zero_keeps_raw_statistic():
    result = stats([4.0, 4.0, 4.0])
    assert result["median"] == 4.0
    assert result["mad"] == 0.0
    assert run_synthetic_checks()["passed"]
