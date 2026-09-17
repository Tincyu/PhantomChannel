import csv
import json
from pathlib import Path

from .csv_utils import EVENT_FIELDS


def packet_event_rows(ble_rows, bt_rows):
    rows = []
    for pkt in ble_rows:
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
                "crc_capture_status": pkt.get("crc_capture_status"),
            }
        )
    for pkt in bt_rows:
        lap = pkt.get("lap", "")
        bdaddr = pkt.get("bdaddr", "")
        protocol_id = f"BT_BDADDR_{bdaddr}" if bdaddr else f"BT_LAP_{lap}"
        rows.append(
            {
                "packet_type": pkt.get("packet_type", "BT_CLASSIC"),
                "sample_index": pkt.get("sample_index"),
                "timestamp_us": pkt.get("timestamp_us"),
                "timestamp_s": pkt.get("timestamp_s"),
                "protocol_id": protocol_id,
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
                "crc_capture_status": "",
            }
        )
    rows.sort(key=lambda row: int(float(row["sample_index"] or 0)))
    return rows


class CsvAppendSink:
    def __init__(self, output_dir, ble_fields, bt_fields):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._files = []
        self._ble_writer = self._open_writer("ble_packets.csv", ble_fields)
        self._bt_writer = self._open_writer("btclassic_packets.csv", bt_fields)
        self._event_writer = self._open_writer("packet_events.csv", EVENT_FIELDS)

    def _open_writer(self, filename, fieldnames):
        handle = (self.output_dir / filename).open("w", newline="", encoding="utf-8")
        self._files.append(handle)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        return writer

    def write_rows(self, ble_rows, bt_rows):
        self._ble_writer.writerows(ble_rows)
        self._bt_writer.writerows(bt_rows)
        self._event_writer.writerows(packet_event_rows(ble_rows, bt_rows))
        for handle in self._files:
            handle.flush()

    def close(self):
        for handle in self._files:
            handle.close()
        self._files = []


class JsonlEventSink:
    def __init__(self, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write_rows(self, ble_rows, bt_rows):
        for row in packet_event_rows(ble_rows, bt_rows):
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()


class MultiSink:
    def __init__(self, sinks):
        self.sinks = list(sinks)

    def write_rows(self, ble_rows, bt_rows):
        for sink in self.sinks:
            sink.write_rows(ble_rows, bt_rows)

    def close(self):
        for sink in self.sinks:
            sink.close()
