import csv
from pathlib import Path


BLE_FIELDS = [
    "packet_type",
    "sample_index",
    "timestamp_us",
    "timestamp_s",
    "access_address",
    "ble_pdu_type",
    "whitened_pdu_hex",
    "advertiser_address",
    "advertiser_address_type",
    "peer_address",
    "peer_address_type",
    "ble_device_address",
    "channel",
    "center_freq_desc",
    "crc_ok",
    "payload_len",
    "rssi",
    "cfo_hz",
    "confidence_score",
    "raw_offset_info",
    "direction_hint",
    "dewhitened_pdu_hex",
    "captured_crc_hex",
    "post_crc_hex",
    "crc_and_post_crc_hex",
    "crc_capture_status",
]

BT_FIELDS = [
    "packet_type",
    "sample_index",
    "timestamp_us",
    "timestamp_s",
    "lap",
    "uap",
    "nap",
    "bdaddr",
    "channel",
    "center_freq_desc",
    "packet_header_info",
    "hec_ok",
    "crc_ok",
    "rssi",
    "cfo_hz",
]

EVENT_FIELDS = [
    "packet_type",
    "sample_index",
    "timestamp_us",
    "timestamp_s",
    "protocol_id",
    "access_address",
    "ble_device_address",
    "lap",
    "bdaddr",
    "rssi",
    "cfo_hz",
    "channel",
    "wideband_sample_index",
    "subband_freq_mhz",
    "hw_timestamp_s",
    "hw_timestamp_us",
    "timestamp_status",
    "payload_len",
    "crc_ok",
    "direction_hint",
    "dewhitened_pdu_hex",
    "captured_crc_hex",
    "post_crc_hex",
    "crc_and_post_crc_hex",
    "crc_capture_status",
]

PAIR_SCORE_FIELDS = [
    "ble_access_address",
    "ble_access_address_aliases",
    "ble_device_address",
    "ble_device_address_aliases",
    "bt_lap",
    "bt_bdaddr",
    "bt_id_aliases",
    "valid_pair_count",
    "rmse_us",
    "rmse_demean_us",
    "mode_delta_us",
    "mean_delta_us",
    "median_delta_us",
    "std_delta_us",
    "mad_us",
    "iqr_us",
    "decision",
    "ambiguity_flag",
    "insufficient_packets_flag",
    "rank_in_ble_group",
    "rmse_rel_score",
    "gap_ratio_vs_runner_up",
    "circular_concentration",
    "auto_decision",
    "filtered_pair_count",
]

LINK_RESULT_FIELDS = [
    "ble_access_address",
    "ble_access_address_aliases",
    "ble_device_address",
    "ble_device_address_aliases",
    "best_bt_lap",
    "best_bt_bdaddr",
    "best_bt_id_aliases",
    "best_score_us",
    "valid_pair_count",
    "decision",
    "reason",
]

BLE_BLE_PAIR_SCORE_FIELDS = [
    "advertiser_address",
    "advertiser_address_aliases",
    "connection_access_address",
    "connection_access_address_aliases",
    "advertising_packet_count",
    "connection_packet_count",
    "valid_pair_count",
    "rmse_us",
    "rmse_demean_us",
    "mode_delta_us",
    "mean_delta_us",
    "median_delta_us",
    "std_delta_us",
    "mad_us",
    "iqr_us",
    "decision",
    "ambiguity_flag",
    "insufficient_packets_flag",
    "rank_in_ble_group",
    "rmse_rel_score",
    "gap_ratio_vs_runner_up",
    "circular_concentration",
    "auto_decision",
    "filtered_pair_count",
]

BLE_BLE_LINK_RESULT_FIELDS = [
    "advertiser_address",
    "advertiser_address_aliases",
    "best_connection_access_address",
    "best_connection_access_address_aliases",
    "best_score_us",
    "valid_pair_count",
    "decision",
    "reason",
]

DEVICE_GROUP_FIELDS = [
    "group_id",
    "identifiers",
    "bt_lap_list",
    "ble_access_address_list",
    "ble_device_address_list",
    "link_edges",
    "best_rmse_us",
    "edge_count",
]

CLASSIFICATION_REPORT_FIELDS = [
    "ble_id",
    "ble_id_type",
    "matched_id",
    "matched_id_type",
    "rmse_demean_us",
    "valid_pair_count",
    "rank_in_ble_group",
    "rmse_rel_score",
    "gap_ratio_vs_runner_up",
    "circular_concentration",
    "auto_decision",
]


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_packet_events(ble_packets, bt_packets, output_path):
    rows = []
    for pkt in ble_packets:
        aa = pkt.get("access_address", "")
        ble_device_address = pkt.get("ble_device_address", "")
        protocol_id = f"BLE_DEV_{ble_device_address}" if ble_device_address else f"BLE_AA_{aa}"
        rows.append(
            {
                "packet_type": pkt.get("packet_type", "BLE_CONN"),
                "sample_index": pkt.get("sample_index"),
                "timestamp_us": pkt.get("timestamp_us"),
                "timestamp_s": pkt.get("timestamp_s"),
                "protocol_id": protocol_id,
                "access_address": aa,
                "ble_device_address": ble_device_address,
                "lap": "",
                "bdaddr": "",
                "rssi": pkt.get("rssi"),
                "cfo_hz": pkt.get("cfo_hz"),
                "channel": pkt.get("channel"),
                "wideband_sample_index": pkt.get("wideband_sample_index"),
                "subband_freq_mhz": pkt.get("subband_freq_mhz"),
                "hw_timestamp_s": pkt.get("hw_timestamp_s"),
                "hw_timestamp_us": pkt.get("hw_timestamp_us"),
                "timestamp_status": pkt.get("timestamp_status"),
                "payload_len": pkt.get("payload_len"),
                "crc_ok": pkt.get("crc_ok"),
                "direction_hint": pkt.get("direction_hint"),
                "dewhitened_pdu_hex": pkt.get("dewhitened_pdu_hex"),
                "captured_crc_hex": pkt.get("captured_crc_hex"),
                "post_crc_hex": pkt.get("post_crc_hex"),
                "crc_and_post_crc_hex": pkt.get("crc_and_post_crc_hex"),
                "crc_capture_status": pkt.get("crc_capture_status"),
            }
        )
    for pkt in bt_packets:
        lap = pkt.get("lap", "")
        bdaddr = pkt.get("bdaddr", "")
        proto_id = f"BT_BDADDR_{bdaddr}" if bdaddr else f"BT_LAP_{lap}"
        rows.append(
            {
                "packet_type": pkt.get("packet_type", "BT_CLASSIC"),
                "sample_index": pkt.get("sample_index"),
                "timestamp_us": pkt.get("timestamp_us"),
                "timestamp_s": pkt.get("timestamp_s"),
                "protocol_id": proto_id,
                "access_address": "",
                "ble_device_address": "",
                "lap": lap,
                "bdaddr": bdaddr,
                "rssi": pkt.get("rssi"),
                "cfo_hz": pkt.get("cfo_hz"),
                "channel": pkt.get("channel"),
                "wideband_sample_index": pkt.get("wideband_sample_index"),
                "subband_freq_mhz": pkt.get("subband_freq_mhz"),
                "hw_timestamp_s": pkt.get("hw_timestamp_s"),
                "hw_timestamp_us": pkt.get("hw_timestamp_us"),
                "timestamp_status": pkt.get("timestamp_status"),
                "payload_len": "",
                "crc_ok": pkt.get("crc_ok"),
                "direction_hint": "",
                "dewhitened_pdu_hex": "",
                "captured_crc_hex": "",
                "post_crc_hex": "",
                "crc_and_post_crc_hex": "",
                "crc_capture_status": "",
            }
        )
    rows.sort(key=lambda row: int(float(row["sample_index"] or 0)))
    write_csv(output_path, rows, EVENT_FIELDS)
    return rows
