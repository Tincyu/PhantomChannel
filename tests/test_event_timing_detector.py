import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/analyze_event_timing_detector.py"
SPEC = importlib.util.spec_from_file_location("event_timing_detector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def adv_packet(frame, timestamp_us, channel, counter):
    return MODULE.AdvPacket(frame, timestamp_us, "d1:22:33:44:55:66", channel, counter, 100.0)


def test_advertising_windows_are_non_overlapping_30_event_chunks():
    events = []
    for index in range(60):
        start = index * 20_000.0
        base = index * 3
        events.append(MODULE.AdvEvent(
            index,
            start,
            [
                adv_packet(index * 3, start, 37, base),
                adv_packet(index * 3 + 1, start + 100.0, 38, base + 1),
                adv_packet(index * 3 + 2, start + 200.0, 39, base + 2),
            ],
        ))

    windows = MODULE.adv_windows(events, "normal", "capture", 30)

    assert len(windows) == 2
    assert [window.window_index for window in windows] == [0, 1]
    assert [window.event_count for window in windows] == [30, 30]
    assert windows[0].end_us < windows[1].start_us


def conn_packet(frame, timestamp_us, direction, counter, event_counter):
    return MODULE.ConnPacket(
        frame=frame,
        timestamp_us=timestamp_us,
        access_address="0x12345678",
        direction=direction,
        channel=20,
        event_counter=event_counter,
        packet_counter=counter,
        gap_us=100.0,
        llid=1,
        length=0,
    )


def test_connection_windows_allow_missing_event_counters_and_keep_role_pair():
    events = []
    for index in range(50):
        start = index * 50_000.0
        event_counter = 100 + index
        events.append(MODULE.ConnEvent(
            event_counter,
            [
                conn_packet(index * 2, start, "C2P", index * 2, event_counter),
                conn_packet(index * 2 + 1, start + 100.0, "P2C", index * 2 + 1, event_counter),
            ],
        ))

    assert events[0].inter_frame_gaps("central")[0][1] == "central-request-response"
    assert not events[0].inter_frame_gaps("peripheral")
    reverse_event = MODULE.ConnEvent(
        200,
        [
            conn_packet(0, 0.0, "P2C", 0, 200),
            conn_packet(1, 100.0, "C2P", 1, 200),
        ],
    )
    assert reverse_event.inter_frame_gaps("peripheral")[0][1] == "peripheral-notification-response"
    windows = MODULE.connection_windows(events, "normal", "capture", 50, "central", 0.8)

    assert len(windows) == 1
    assert windows[0].event_count == 50
    assert windows[0].pair_coverage == 1.0
    assert windows[0].gap_pair_counts == {"central-request-response": 50}

    gapped = []
    for index in range(50):
        start = index * 60_000.0
        event_counter = 200 + index * 2
        gapped.append(MODULE.ConnEvent(
            event_counter,
            [
                conn_packet(index * 2, start, "C2P", index * 2, event_counter),
                conn_packet(index * 2 + 1, start + 100.0, "P2C", index * 2 + 1, event_counter),
            ],
        ))
    gapped_windows = MODULE.connection_windows(gapped, "normal", "gapped", 50, "central", 0.8)
    assert len(gapped_windows) == 1
    assert gapped_windows[0].event_counter_step_median == 2
    assert gapped_windows[0].observed_event_gap_count == 49
    assert gapped_windows[0].event_interval_median_us == 30_000.0

    censored_windows = MODULE.connection_windows(gapped, "normal", "censored", 50, "peripheral", 0.8)
    assert len(censored_windows) == 1
    assert censored_windows[0].gap_median_us is None
    assert censored_windows[0].pair_coverage == 0.0


def test_auc_uses_half_credit_for_ties():
    auc, points, tpr_at_five = MODULE.auc_and_roc([1.0, 2.0], [2.0, 3.0])

    assert auc == 0.875
    assert points
    assert tpr_at_five == 0.5


def test_standard_timing_keeps_crc_error_rows_unless_strict_mode():
    rows = [
        {
            "frame.time_epoch": "1.0",
            "btle.access_address": "0x12345678",
            "nordic_ble.direction": "True",
            "nordic_ble.event_counter": "10",
            "nordic_ble.packet_counter": "20",
            "nordic_ble.delta_time": "",
            "btle.data_header.llid": "1",
            "btle.data_header.length": "0",
            "nordic_ble.crcok": "False",
        },
        {
            "frame.time_epoch": "1.0002",
            "btle.access_address": "0x12345678",
            "nordic_ble.direction": "False",
            "nordic_ble.event_counter": "10",
            "nordic_ble.packet_counter": "21",
            "nordic_ble.delta_time": "200",
            "btle.data_header.llid": "1",
            "btle.data_header.length": "0",
            "nordic_ble.crcok": "True",
        },
    ]
    events, _, count, _ = MODULE.connection_events_from_rows(rows, "")
    assert count == 2
    assert len(events) == 1
    assert events[0].packets[0].crc_valid is False
    strict_events, _, strict_count, _ = MODULE.connection_events_from_rows(
        rows, "", require_crc_valid=True,
    )
    assert strict_count == 1
    assert len(strict_events) == 1
    assert len(strict_events[0].packets) == 1
