import argparse
import json
from pathlib import Path

from bt_pipeline.csv_utils import BLE_FIELDS, BT_FIELDS, save_packet_events, write_csv
from bt_pipeline.io_utils import (
    capture_framed_udp_to_bin,
    capture_udp_to_bin,
    load_metadata,
    load_rx_time_tags,
    sample_index_to_hw_time,
)
from bt_pipeline.parsers import parse_ble_packets, parse_btclassic_packets
from bt_pipeline.wideband_channelizer import (
    ble_target_freqs_mhz,
    bredr_target_freqs_mhz,
    channelize_to_4m,
    design_channel_lpf,
    iter_iq_chunks,
    wideband_index_from_subband,
)


WIDEBAND_SAMPLE_RATE = 40_000_000.0
SUBBAND_SAMPLE_RATE = 4_000_000.0
DEFAULT_BANDWIDTH = 40_000_000.0

BLE_40M_FIELDS = BLE_FIELDS + [
    "wideband_sample_index",
    "subband_sample_index",
    "subband_freq_mhz",
    "hw_timestamp_s",
    "hw_timestamp_us",
    "timestamp_status",
    "rx_time_tag_block_seq",
    "rx_time_tag_offset",
    "gap_samples_est",
    "overflow_count_total",
    "parse_result",
]
BT_40M_FIELDS = BT_FIELDS + [
    "wideband_sample_index",
    "subband_sample_index",
    "subband_freq_mhz",
    "hw_timestamp_s",
    "hw_timestamp_us",
    "timestamp_status",
    "rx_time_tag_block_seq",
    "rx_time_tag_offset",
    "gap_samples_est",
    "overflow_count_total",
    "parse_result",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="40 MHz wideband Bluetooth capture and offline parse entry. "
        "It channelizes to 4 MHz subbands and does not modify the single-channel parser."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Capture 40 MHz wideband UDP IQ for 10 seconds.")
    capture.add_argument("--udp-ip", default="0.0.0.0")
    capture.add_argument("--udp-port", type=int, default=9000)
    capture.add_argument("--sample-rate", type=float, default=WIDEBAND_SAMPLE_RATE)
    capture.add_argument("--center-freq", type=float, default=2420e6)
    capture.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    capture.add_argument("--duration", type=float, default=10.0)
    capture.add_argument("--iq-format", choices=["complex64", "int16"], default="complex64")
    capture.add_argument("--output-bin", default="data/captures/capture_40m_10s.bin")
    capture.add_argument("--metadata", default="data/metadata/metadata_40m.json")
    capture.add_argument("--udp-framed", action="store_true", help="Receive UHD metadata framed UDP packets.")
    capture.add_argument(
        "--rx-tags-csv",
        default="data/rx_tags/rx_time_tags.csv",
        help="CSV path for UHD rx_time block tags.",
    )
    capture.add_argument("--udp-payload-format", choices=["sc16", "complex64"], default="sc16")
    capture.add_argument(
        "--strict-fragments",
        action="store_true",
        help="Discard incomplete fragmented blocks. Missing samples are never zero-filled.",
    )
    capture.add_argument("--max-packet-size", type=int, default=65535)
    capture.add_argument("--socket-rcvbuf", type=int, default=256 * 1024 * 1024)

    parse = sub.add_parser("parse", help="Parse a 40 MHz wideband bin by channelizing to 4 MHz.")
    parse.add_argument("--input-bin", required=True)
    parse.add_argument("--metadata", default="")
    parse.add_argument("--rx-tags-csv", default="")
    parse.add_argument("--timestamp-mode", choices=["sample_index", "uhd_rx_time", "auto"], default="auto")
    parse.add_argument("--sample-rate", type=float, default=WIDEBAND_SAMPLE_RATE)
    parse.add_argument("--center-freq", type=float, default=2420e6)
    parse.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parse.add_argument(
        "--subband-sample-rate",
        type=float,
        default=SUBBAND_SAMPLE_RATE,
        help="Output sample rate for each channelized subband. Default keeps every subchannel at 4 MHz.",
    )
    parse.add_argument("--iq-format", choices=["complex64", "int16"], default="complex64")
    parse.add_argument("--output-dir", default="artifacts/40m/parse")
    parse.add_argument("--freq-dev", type=float, default=250e3)
    parse.add_argument("--ble-threshold", type=float, default=0.01)
    parse.add_argument(
        "--ble-candidate-detector",
        choices=("fixed", "adaptive_hysteresis"),
        default="fixed",
        help="BLE candidate gate. The default preserves the legacy fixed-amplitude behavior.",
    )
    parse.add_argument("--ble-envelope-window-samples", type=int, default=8)
    parse.add_argument("--ble-start-noise-multiplier", type=float, default=3.5)
    parse.add_argument("--ble-hold-noise-multiplier", type=float, default=1.8)
    parse.add_argument("--ble-candidate-gap-tolerance-samples", type=int, default=0)
    parse.add_argument("--ble-candidate-prepad-samples", type=int, default=0)
    parse.add_argument("--ble-candidate-postpad-samples", type=int, default=0)
    parse.add_argument("--br-threshold", type=float, default=0.005)
    parse.add_argument("--ble-segment-min-len", type=int, default=150)
    parse.add_argument("--br-segment-min-len", type=int, default=200)
    parse.add_argument("--ble-score-threshold", type=float, default=3.0)
    parse.add_argument("--br-cutoff", type=float, default=0.5e6)
    parse.add_argument("--ble-lpf-cutoff", type=float, default=1.0e6)
    parse.add_argument("--bredr-lpf-cutoff", type=float, default=0.75e6)
    parse.add_argument("--numtaps", type=int, default=401)
    parse.add_argument("--chunk-samples", type=int, default=4_000_000)
    parse.add_argument("--overlap-samples", type=int, default=200_000)
    parse.add_argument(
        "--ble-freqs-mhz",
        default="",
        help="Optional comma-separated BLE subband centers, for example 2402,2426. Empty means all in-band BLE centers.",
    )
    parse.add_argument(
        "--bredr-freqs-mhz",
        default="",
        help="Optional comma-separated BR/EDR subband centers, for example 2402,2420. Empty means all in-band BR/EDR centers.",
    )
    parse.add_argument("--skip-ble", action="store_true")
    parse.add_argument("--skip-bredr", action="store_true")
    return parser


def update_40m_metadata(path, bandwidth, mode):
    path = Path(path)
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["bandwidth"] = float(bandwidth)
    data["wideband_mode"] = mode
    data["timestamp_note"] = (
        f"Wideband sample rate: {float(data.get('sample_rate', 0.0)):.6f} Hz. "
        "Subbands are channelized to 4 MHz when the wideband sample rate is an integer multiple of 4 MHz."
    )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_capture(args):
    if args.udp_framed:
        capture_framed_udp_to_bin(
            udp_ip=args.udp_ip,
            udp_port=args.udp_port,
            sample_rate=args.sample_rate,
            duration=args.duration,
            iq_format=args.iq_format,
            output_bin=args.output_bin,
            metadata_json=args.metadata,
            rx_tags_csv=args.rx_tags_csv,
            center_freq=args.center_freq,
            expected_payload_format=args.udp_payload_format,
            strict_fragments=args.strict_fragments,
            socket_rcvbuf=args.socket_rcvbuf,
            max_packet_size=args.max_packet_size,
            bandwidth=args.bandwidth,
        )
    else:
        capture_udp_to_bin(
            args.udp_ip,
            args.udp_port,
            args.sample_rate,
            args.duration,
            args.iq_format,
            args.output_bin,
            args.metadata,
            center_freq=args.center_freq,
        )
        update_40m_metadata(args.metadata, args.bandwidth, "40m_capture")


def parse_freq_list(value):
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def keep_packet_in_core(packet, chunk_start, core_start, core_end, decim, filter_delay):
    sub_idx = float(packet["sample_index"])
    wide_idx = wideband_index_from_subband(chunk_start, sub_idx, decim, filter_delay)
    return core_start <= wide_idx < core_end


def resolve_parse_capture_params(args, metadata):
    if metadata:
        metadata_sample_rate = metadata.get("actual_sample_rate_sps", metadata.get("sample_rate_sps", metadata.get("sample_rate")))
        metadata_center_freq = metadata.get("actual_center_frequency_hz", metadata.get("center_frequency_hz", metadata.get("center_freq")))
        metadata_bandwidth = metadata.get("processing_bandwidth_hz", metadata.get("bandwidth"))
        if args.sample_rate == WIDEBAND_SAMPLE_RATE and metadata_sample_rate:
            args.sample_rate = float(metadata_sample_rate)
        if args.center_freq == 2420e6 and metadata_center_freq:
            args.center_freq = float(metadata_center_freq)
        if args.bandwidth == DEFAULT_BANDWIDTH and metadata_bandwidth:
            args.bandwidth = float(metadata_bandwidth)

    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.subband_sample_rate <= 0:
        raise ValueError("--subband-sample-rate must be positive")

    decim_float = args.sample_rate / args.subband_sample_rate
    decim = int(round(decim_float))
    if decim < 1 or abs(decim_float - decim) > 1e-6:
        raise ValueError(
            f"--sample-rate {args.sample_rate:g} must be an integer multiple of "
            f"--subband-sample-rate {args.subband_sample_rate:g}. "
            "Use rates such as 8e6, 12e6, 16e6, 20e6, or 40e6 for 4 MHz subbands."
        )

    effective_bandwidth = min(float(args.bandwidth), float(args.sample_rate))
    if effective_bandwidth < args.bandwidth:
        print(
            f"WARNING: --bandwidth {args.bandwidth:g} exceeds --sample-rate {args.sample_rate:g}; "
            f"using effective bandwidth {effective_bandwidth:g} for channel selection."
        )
    if effective_bandwidth < args.subband_sample_rate:
        raise ValueError("Effective bandwidth is narrower than one 4 MHz subband.")

    return decim, effective_bandwidth


def resolve_timestamp_config(args, metadata):
    if args.rx_tags_csv:
        rx_tags_csv = args.rx_tags_csv
    else:
        rx_tags_csv = metadata.get("rx_tags_csv", "")
        if rx_tags_csv and not Path(rx_tags_csv).is_absolute() and args.metadata:
            direct_path = Path(rx_tags_csv)
            metadata_relative_path = Path(args.metadata).resolve().parent / rx_tags_csv
            rx_tags_csv = str(direct_path if direct_path.exists() else metadata_relative_path)
    if args.timestamp_mode == "sample_index":
        return "sample_index", "", []
    if args.timestamp_mode == "uhd_rx_time" and not rx_tags_csv:
        raise ValueError("--timestamp-mode uhd_rx_time requires --rx-tags-csv or metadata rx_tags_csv")
    if args.timestamp_mode == "auto" and not rx_tags_csv:
        return "sample_index", "", []

    tags = load_rx_time_tags(rx_tags_csv)
    if args.timestamp_mode == "uhd_rx_time" and not tags:
        raise ValueError(f"No usable rx_time tags found: {rx_tags_csv}")
    if not tags:
        return "sample_index", "", []
    return "uhd_rx_time", rx_tags_csv, tags


def validate_rx_tag_sample_rate(args, timestamp_mode, rx_time_tags):
    if timestamp_mode != "uhd_rx_time" or not rx_time_tags:
        return

    tag_rates = [
        tag.sample_rate
        for tag in rx_time_tags
        if tag.sample_rate > 0 and tag.nsamps > 0 and tag.complete
    ]
    if not tag_rates:
        return

    first_rate = float(tag_rates[0])
    min_rate = min(tag_rates)
    max_rate = max(tag_rates)
    if max_rate - min_rate > 1.0:
        raise ValueError(
            "rx_time_tags.csv contains inconsistent sample rates: "
            f"min={min_rate:g}, max={max_rate:g}. Check the UHD sender."
        )

    if abs(first_rate - args.sample_rate) > 1.0:
        raise ValueError(
            "Sample-rate mismatch between parse arguments and UHD rx_time tags: "
            f"--sample-rate={args.sample_rate:g}, rx_tags sample_rate={first_rate:g}. "
            "Use the actual UHD sample rate from rx_time_tags.csv, or fix the capture sender. "
            "The channelizer must use the real IQ sample rate."
        )


def convert_packet_timestamps(
    packet,
    chunk_start,
    subband_freq_mhz,
    sample_rate,
    subband_sample_rate,
    decim,
    filter_delay,
    timestamp_mode="sample_index",
    rx_time_tags=None,
):
    sub_idx = float(packet["sample_index"])
    wide_idx = wideband_index_from_subband(chunk_start, sub_idx, decim, filter_delay)
    timestamp_status = "sample_index"
    hw_timestamp_s = ""
    hw_timestamp_us = ""
    rx_info = {
        "rx_time_tag_block_seq": "",
        "rx_time_tag_offset": "",
        "gap_samples_est": "",
        "overflow_count_total": "",
    }

    if timestamp_mode == "uhd_rx_time":
        hw_time_sec, timestamp_status, info = sample_index_to_hw_time(wide_idx, rx_time_tags or [], sample_rate)
        if hw_time_sec is None:
            timestamp_us = wide_idx * 1e6 / sample_rate
        else:
            timestamp_us = hw_time_sec * 1e6
            hw_timestamp_s = f"{hw_time_sec:.9f}"
            hw_timestamp_us = f"{timestamp_us:.3f}"
            rx_info.update(info)
    else:
        timestamp_us = wide_idx * 1e6 / sample_rate

    equiv_subband_sample_index = int(round(timestamp_us * subband_sample_rate / 1e6))
    packet["subband_sample_index"] = int(round(sub_idx))
    packet["wideband_sample_index"] = int(round(wide_idx))
    packet["sample_index"] = equiv_subband_sample_index
    packet["timestamp_us"] = f"{timestamp_us:.3f}"
    packet["timestamp_s"] = f"{timestamp_us / 1e6:.9f}"
    packet["subband_freq_mhz"] = f"{subband_freq_mhz:.3f}"
    packet["hw_timestamp_s"] = hw_timestamp_s
    packet["hw_timestamp_us"] = hw_timestamp_us
    packet["timestamp_status"] = timestamp_status
    packet["rx_time_tag_block_seq"] = rx_info["rx_time_tag_block_seq"]
    packet["rx_time_tag_offset"] = rx_info["rx_time_tag_offset"]
    packet["gap_samples_est"] = rx_info["gap_samples_est"]
    packet["overflow_count_total"] = rx_info["overflow_count_total"]
    return packet


def parse_ble_wideband_chunk(
    args,
    chunk,
    chunk_start,
    core_start,
    core_end,
    lpf,
    filter_delay,
    decim,
    effective_bandwidth,
    timestamp_mode,
    rx_time_tags,
):
    rows = []
    target_freqs = parse_freq_list(args.ble_freqs_mhz) or ble_target_freqs_mhz(args.center_freq, effective_bandwidth)
    for freq_mhz in target_freqs:
        sub_iq = channelize_to_4m(chunk, chunk_start, args.sample_rate, args.center_freq, freq_mhz, decim, lpf)
        packets = parse_ble_packets(
            sub_iq,
            args.subband_sample_rate,
            freq_mhz * 1e6,
            args.freq_dev,
            args.ble_threshold,
            args.ble_segment_min_len,
            args.ble_score_threshold,
        )
        for packet in packets:
            if not keep_packet_in_core(packet, chunk_start, core_start, core_end, decim, filter_delay):
                continue
            packet = convert_packet_timestamps(
                packet,
                chunk_start,
                freq_mhz,
                args.sample_rate,
                args.subband_sample_rate,
                decim,
                filter_delay,
                timestamp_mode,
                rx_time_tags,
            )
            ble_id = packet.get("ble_device_address") or packet.get("access_address")
            packet["parse_result"] = (
                f"BLE ch={packet.get('channel')} freq={freq_mhz:.3f}MHz "
                f"AA={packet.get('access_address')} dev={ble_id} len={packet.get('payload_len')} "
                f"score={packet.get('confidence_score')}"
            )
            rows.append(packet)
            print(
                f"BLE | ch={packet.get('channel')} | freq={freq_mhz:.3f} MHz | "
                f"ts_us={packet['timestamp_us']} | AA={packet.get('access_address')} | dev={ble_id} | "
                f"len={packet.get('payload_len')} | score={packet.get('confidence_score')}"
            )
    return rows


def parse_bredr_wideband_chunk(
    args,
    chunk,
    chunk_start,
    core_start,
    core_end,
    lpf,
    filter_delay,
    decim,
    effective_bandwidth,
    timestamp_mode,
    rx_time_tags,
):
    rows = []
    target_freqs = parse_freq_list(args.bredr_freqs_mhz) or bredr_target_freqs_mhz(args.center_freq, effective_bandwidth)
    for freq_mhz in target_freqs:
        sub_iq = channelize_to_4m(chunk, chunk_start, args.sample_rate, args.center_freq, freq_mhz, decim, lpf)
        packets = parse_btclassic_packets(
            sub_iq,
            args.subband_sample_rate,
            freq_mhz * 1e6,
            args.freq_dev,
            args.br_threshold,
            args.br_segment_min_len,
            args.br_cutoff,
        )
        bredr_channel = int(round(freq_mhz - 2402.0))
        for packet in packets:
            if not keep_packet_in_core(packet, chunk_start, core_start, core_end, decim, filter_delay):
                continue
            packet = convert_packet_timestamps(
                packet,
                chunk_start,
                freq_mhz,
                args.sample_rate,
                args.subband_sample_rate,
                decim,
                filter_delay,
                timestamp_mode,
                rx_time_tags,
            )
            packet["channel"] = bredr_channel
            packet["parse_result"] = (
                f"BR/EDR ch={bredr_channel} freq={freq_mhz:.3f}MHz "
                f"LAP={packet.get('lap')} header={packet.get('packet_header_info')}"
            )
            rows.append(packet)
            print(
                f"BR/EDR | ch={bredr_channel} | freq={freq_mhz:.3f} MHz | "
                f"ts_us={packet['timestamp_us']} | LAP={packet.get('lap')} | "
                f"header={packet.get('packet_header_info')}"
            )
    return rows


def run_parse(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    decim, effective_bandwidth = resolve_parse_capture_params(args, metadata)
    timestamp_mode, rx_tags_csv, rx_time_tags = resolve_timestamp_config(args, metadata)
    validate_rx_tag_sample_rate(args, timestamp_mode, rx_time_tags)
    ble_freqs = parse_freq_list(args.ble_freqs_mhz) or ble_target_freqs_mhz(args.center_freq, effective_bandwidth)
    bredr_freqs = parse_freq_list(args.bredr_freqs_mhz) or bredr_target_freqs_mhz(args.center_freq, effective_bandwidth)

    print(
        f"Wideband mode: sample_rate={args.sample_rate:g} Hz, "
        f"center_freq={args.center_freq / 1e6:.3f} MHz, bandwidth={effective_bandwidth:g} Hz."
    )
    print(
        f"Channelizer: decimation by {decim} gives "
        f"{args.subband_sample_rate / 1e6:.3f} MHz subbands."
    )
    print("BLE channel numbers and BR/EDR channel numbers are derived from each subband center frequency.")
    print(f"BLE subbands ({len(ble_freqs)}): {', '.join(f'{freq:.3f}' for freq in ble_freqs)} MHz")
    print(f"BR/EDR subbands ({len(bredr_freqs)}): {', '.join(f'{freq:.3f}' for freq in bredr_freqs)} MHz")
    print(f"Timestamp mode: {timestamp_mode}")
    if timestamp_mode == "uhd_rx_time":
        print(f"Using UHD rx_time tags: {rx_tags_csv}")
        print(
            "rx_time fixes the receive time axis after overflow, but it cannot recover missing IQ samples. "
            "Packets near overflow/gap/incomplete blocks are marked in timestamp_status."
        )
    else:
        print("Timestamp uses FIR-delay-compensated wideband_sample_index / sample_rate.")

    ble_lpf = design_channel_lpf(args.sample_rate, args.ble_lpf_cutoff, args.numtaps)
    bredr_lpf = design_channel_lpf(args.sample_rate, args.bredr_lpf_cutoff, args.numtaps)
    filter_delay = (args.numtaps - 1) / 2.0

    ble_rows = []
    bt_rows = []
    for chunk_start, core_start, core_end, chunk in iter_iq_chunks(
        args.input_bin, args.iq_format, args.chunk_samples, args.overlap_samples
    ):
        print(
            f"Processing wideband chunk: read_start={chunk_start}, "
            f"core=[{core_start}, {core_end}), samples={chunk.size}"
        )
        if not args.skip_ble:
            ble_rows.extend(
                parse_ble_wideband_chunk(
                    args,
                    chunk,
                    chunk_start,
                    core_start,
                    core_end,
                    ble_lpf,
                    filter_delay,
                    decim,
                    effective_bandwidth,
                    timestamp_mode,
                    rx_time_tags,
                )
            )
        if not args.skip_bredr:
            bt_rows.extend(
                parse_bredr_wideband_chunk(
                    args,
                    chunk,
                    chunk_start,
                    core_start,
                    core_end,
                    bredr_lpf,
                    filter_delay,
                    decim,
                    effective_bandwidth,
                    timestamp_mode,
                    rx_time_tags,
                )
            )

    ble_rows.sort(key=lambda row: int(row["sample_index"]))
    bt_rows.sort(key=lambda row: int(row["sample_index"]))
    write_csv(output_dir / "ble_packets.csv", ble_rows, BLE_40M_FIELDS)
    write_csv(output_dir / "btclassic_packets.csv", bt_rows, BT_40M_FIELDS)
    save_packet_events(ble_rows, bt_rows, output_dir / "packet_events.csv")
    print(f"Saved BLE rows: {len(ble_rows)} -> {output_dir / 'ble_packets.csv'}")
    print(f"Saved BR/EDR rows: {len(bt_rows)} -> {output_dir / 'btclassic_packets.csv'}")
    print(f"Saved packet events -> {output_dir / 'packet_events.csv'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "capture":
        run_capture(args)
    elif args.command == "parse":
        run_parse(args)


if __name__ == "__main__":
    main()
