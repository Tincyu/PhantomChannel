from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import bt_40m_pipeline as legacy
import numpy as np
from bt_pipeline.csv_utils import save_packet_events, write_csv
from bt_pipeline.cuda_backend import describe_cuda_status, require_cuda, synchronize, use_device
from bt_pipeline.io_utils import build_rx_time_tag_lookup
from bt_pipeline.native_backend import (
    build_btclassic_packet_candidates_native,
    native_status,
    parse_btclassic_packet_segments_native_finalized,
    parse_ble_packet_segments_native,
    validate_native_parser_backend,
)
from bt_pipeline.parsers import (
    LockedUAPCache,
    build_btclassic_packet_candidates,
    finalize_btclassic_packet_candidates,
    known_lap_sync_words,
    parse_ble_packet_segments,
    parse_ble_packets,
    parse_btclassic_packet_segments,
    parse_btclassic_packets,
)
from bt_pipeline.pfb_channelizer import (
    HostSegmentBatch,
    apply_cleanup_lpf,
    channelize_oversampled_pfb,
    describe_target_mapping,
    design_cleanup_lpf,
    design_oversampled_pfb,
    select_coarse_subband,
    shift_cleanup_threshold_segments,
    shift_cleanup_threshold_segments_batch,
    shift_within_subband,
    shift_within_subband_batch,
    threshold_segments_with_iq,
)
from bt_pipeline.wideband_channelizer import (
    ble_target_freqs_mhz,
    bredr_target_freqs_mhz,
    iter_iq_chunks,
)

TARGET_PROFILE_PREFIX = "target_profile|"
TARGET_PROFILE_METRICS = (
    "targets",
    "segment_count",
    "segment_copy_calls",
    "segment_copy_samples",
    "segment_copy_back_s",
    "segment_buffers",
    "segment_descriptor_count",
    "segment_input_build_s",
    "threshold_mask_s",
    "segment_materialize_s",
    "segment_materialize_samples",
    "segment_materialize_calls",
    "fused_target_s",
)
TARGET_SELECTION_METRICS = (
    "segment_copy_samples",
    "segment_count",
    "segment_copy_calls",
    "fused_target_s",
)
TARGET_SELECTION_FIELDS = (
    "protocol",
    "freq_mhz",
    "selected",
    "rank",
    "metric",
)


@dataclass
class PfbRuntimeConfig:
    args: object
    metadata: dict
    decim: int
    effective_bandwidth: float
    prototype: np.ndarray
    num_channels: int
    ble_cleanup_lpf: np.ndarray
    bredr_cleanup_lpf: np.ndarray
    ble_freqs: list
    bredr_freqs: list
    filter_delay: float
    timestamp_mode: str
    rx_tags_csv: str
    rx_time_tags: list
    cuda_device_id: int | None
    cuda_device: dict | None
    cuda_iq_overlap_cache: dict | None = None
    realtime: bool = False


def build_parser():
    parser = legacy.build_parser()
    parser.description = (
        "Wideband Bluetooth offline parse entry using a 2x oversampled FFT/PFB coarse "
        "channelizer and lightweight within-subband frequency shifts."
    )
    parse_parser = next(
        action for action in parser._actions if action.dest == "command"
    ).choices["parse"]
    parse_parser.add_argument(
        "--pfb-cutoff",
        type=float,
        default=1.9e6,
        help="Prototype PFB low-pass cutoff. Must be below half the 4 MHz subband rate.",
    )
    parse_parser.add_argument(
        "--pfb-numtaps",
        type=int,
        default=401,
        help="Prototype PFB FIR tap count.",
    )
    parse_parser.add_argument(
        "--cleanup-numtaps",
        type=int,
        default=81,
        help="Short protocol-specific cleanup FIR tap count at the subband sample rate.",
    )
    parse_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Coarse-subband worker processes. Default 1 keeps serial execution.",
    )
    parse_parser.add_argument(
        "--parse-workers",
        type=int,
        default=1,
        help=(
            "CPU worker processes for parsing host threshold segments. BLE parses "
            "packets in workers; BR/EDR workers build stateless candidates and the "
            "main process applies sniffer state in segment order. Default 1 keeps "
            "all segment parsing serial."
        ),
    )
    parse_parser.add_argument(
        "--parse-worker-chunk-segments",
        type=int,
        default=32,
        help="Number of threshold segments per CPU parse worker task.",
    )
    parse_parser.add_argument(
        "--cpp-parser-threads",
        type=int,
        default=1,
        help=(
            "Threads used inside bt_native parser kernels. Default 1 keeps the "
            "single-thread correctness baseline."
        ),
    )
    parse_parser.add_argument(
        "--ble-parser-backend",
        choices=("python", "cpp"),
        default="python",
        help=(
            "BLE protocol parser backend. Default python keeps the current verified "
            "parser; cpp is reserved for the staged bt_native implementation."
        ),
    )
    parse_parser.add_argument(
        "--bredr-parser-backend",
        choices=("python", "cpp", "hybrid"),
        default="python",
        help=(
            "BR/EDR protocol parser backend. Default python keeps the current "
            "verified parser; cpp uses the native finalized segment path; hybrid "
            "uses the same native path when possible and falls back for unsupported "
            "known-LAP modes."
        ),
    )
    parse_parser.add_argument(
        "--parallel-bredr-candidates",
        action="store_true",
        help=(
            "Experimental: with --parse-workers > 1, build BR/EDR stateless sniffer "
            "candidates in worker processes and apply PatientSniffer state in the "
            "main process. Correct but slower on the current capture."
        ),
    )
    parse_parser.add_argument(
        "--native-br-batch-candidates",
        action="store_true",
        help=(
            "Experimental: build BR/EDR native hybrid candidates as a C++ segment "
            "batch with --cpp-parser-threads. Currently not byte-identical because "
            "cfo_hz is estimated in C++."
        ),
    )
    parse_parser.add_argument(
        "--native-segment-input",
        choices=("legacy", "compact"),
        default="legacy",
        help=(
            "Native parser segment input mode. compact keeps CUDA copy-back as "
            "merged host buffers plus descriptors and is only used for native "
            "parser backends; legacy preserves the per-segment tuple/list path."
        ),
    )
    parse_parser.add_argument(
        "--cuda-target-dsp-materialization",
        choices=("full", "threshold_then_segments"),
        default="full",
        help=(
            "CUDA fused target DSP materialization mode. full preserves the current "
            "cleaned-IQ matrix path; threshold_then_segments writes only threshold "
            "masks first and materializes IQ only for merged parser segments."
        ),
    )
    parse_parser.add_argument(
        "--known-ble-aa",
        default="",
        help=(
            "Comma-separated known BLE access addresses. When set, BLE parser "
            "skips non-matching access addresses early."
        ),
    )
    parse_parser.add_argument(
        "--known-bredr-lap",
        default="",
        help=(
            "Comma-separated known BR/EDR LAP values. When set, BR/EDR parser "
            "uses direct known-LAP sync-word matching before sniffer decoding."
        ),
    )
    parse_parser.add_argument(
        "--known-bredr-lap-fast-path",
        action="store_true",
        help=(
            "Experimental: use direct known-LAP sync-word matching before BR/EDR "
            "sniffer decoding. Correct on current tests but slower on the current "
            "capture, so disabled by default."
        ),
    )
    parse_parser.add_argument(
        "--known-packet-csv",
        action="append",
        default=[],
        help=(
            "Optional previous packet CSV to seed known BLE access_address and "
            "BR/EDR lap tables. Can be provided more than once."
        ),
    )
    parse_parser.add_argument(
        "--learned-parser-fast-path",
        dest="learned_parser_fast_path",
        action="store_true",
        default=False,
        help=(
            "Dynamically learn BLE access addresses and BR/EDR LAPs during CPU "
            "parsing, then use the learned table to skip repeated PHY/LAP checks. "
            "Experimental; disabled by default because current timing is slightly "
            "slower on the 20m_conn_adv capture."
        ),
    )
    parse_parser.add_argument(
        "--no-learned-parser-fast-path",
        dest="learned_parser_fast_path",
        action="store_false",
        help="Disable dynamic learned AA/LAP CPU parser fast paths.",
    )
    parse_parser.add_argument(
        "--cuda-known-candidate-filter",
        dest="cuda_known_candidate_filter",
        action="store_true",
        default=True,
        help=(
            "With --use-cuda and known AA/LAP tables, filter threshold segments on "
            "GPU before copying IQ back to CPU parsers. Enabled by default."
        ),
    )
    parse_parser.add_argument(
        "--no-cuda-known-candidate-filter",
        dest="cuda_known_candidate_filter",
        action="store_false",
        help="Disable GPU known AA/LAP candidate filtering after threshold detect.",
    )
    parse_parser.add_argument(
        "--use-cuda",
        action="store_true",
        help=(
            "Run PFB, within-subband shift, and cleanup FIR on CUDA; protocol parsers "
            "still run on CPU."
        ),
    )
    parse_parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device id for --use-cuda. Default uses device 0.",
    )
    parse_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-packet parse logs while keeping chunk progress and summaries.",
    )
    parse_parser.add_argument(
        "--timing",
        action="store_true",
        help="Print stage timing summaries for each chunk and the full parse run.",
    )
    parse_parser.add_argument(
        "--cuda-sync-timing",
        action="store_true",
        help=(
            "Synchronize CUDA after timed GPU stages and accumulate gpu_sync_wait_s. "
            "This gives precise CUDA stage timings but adds synchronization overhead."
        ),
    )
    parse_parser.add_argument(
        "--cuda-batch-targets",
        action="store_true",
        help=(
            "Experimental: batch same-protocol targets within each coarse bin on CUDA; "
            "with --cuda-fuse-target-dsp this batches residuals in one fused kernel."
        ),
    )
    parse_parser.add_argument(
        "--cuda-pfb-backend",
        choices=(
            "cupyx",
            "kernel",
            "kernel_multi",
            "kernel_multi_float",
            "kernel_multi_float_phase",
            "kernel_multi_float_phase_t",
            "kernel_phase",
            "kernel_precomp",
            "kernel_const",
        ),
        default="kernel",
        help=(
            "CUDA PFB backend. Default kernel uses the custom polyphase FIR kernel "
            "kernel_multi uses an experimental shared-window multi-output kernel; "
            "kernel_multi_float uses float taps/accumulators in that layout; "
            "kernel_multi_float_phase also fuses IFFT-post phase scatter; "
            "kernel_multi_float_phase_t uses a transposed IFFT input layout; "
            "kernel_phase fuses IFFT-post phase correction into a CUDA kernel; "
            "kernel_precomp uses precomputed alignment/chunk phase tables; "
            "to reduce Python phase loops; kernel_const uses an experimental "
            "constant-memory prototype kernel; cupyx keeps the lfilter-based fallback."
        ),
    )
    parse_parser.add_argument(
        "--cuda-cleanup-backend",
        choices=(
            "cupyx",
            "kernel",
            "kernel_shared",
            "kernel_multi",
            "kernel_multi_float",
            "kernel_const",
        ),
        default="kernel_multi_float",
        help=(
            "CUDA cleanup FIR backend. Default kernel_multi_float uses the fastest "
            "multi-output shared-window FIR kernel with float taps/accumulators; "
            "kernel uses the original custom FIR kernel; "
            "kernel_shared uses an experimental shared-memory FIR kernel; "
            "kernel_multi uses a double-accumulator multi-output shared-window FIR kernel; "
            "kernel_const uses an experimental constant-memory FIR kernel; "
            "cupyx uses cupyx.scipy.signal.lfilter as a fallback."
        ),
    )
    parse_parser.add_argument(
        "--cuda-fuse-target-dsp",
        action="store_true",
        help=(
            "Experimental: fuse per-target shift, cleanup FIR, and threshold mask "
            "generation into one CUDA kernel. Requires CUDA threshold detect."
        ),
    )
    parse_parser.add_argument(
        "--cuda-threshold-detect",
        dest="cuda_threshold_detect",
        action="store_true",
        default=True,
        help=(
            "Run amplitude threshold segmentation on CUDA and copy only candidate "
            "segments back to CPU parsers. Enabled by default with --use-cuda."
        ),
    )
    parse_parser.add_argument(
        "--no-cuda-threshold-detect",
        dest="cuda_threshold_detect",
        action="store_false",
        help="Disable CUDA threshold segmentation and copy full cleaned target subbands.",
    )
    parse_parser.add_argument(
        "--cuda-segment-copy-merge-gap-samples",
        type=int,
        default=4096,
        help=(
            "Maximum gap between CUDA threshold segments copied back in one host "
            "transfer. Default 4096 preserves the previous hard-coded behavior."
        ),
    )
    parse_parser.add_argument(
        "--cuda-segment-copy-max-merged-samples",
        type=int,
        default=262144,
        help=(
            "Maximum span copied back in one merged CUDA threshold transfer. "
            "Default 262144 preserves the previous internal limit."
        ),
    )
    parse_parser.add_argument(
        "--target-dsp-profile",
        action="store_true",
        default=False,
        help=(
            "Collect per protocol/frequency/residual target DSP copy-back profile "
            "metrics. Intended for diagnostics; packet output is unchanged."
        ),
    )
    parse_parser.add_argument(
        "--target-selection",
        choices=("full", "profile_topk"),
        default="full",
        help=(
            "Target frequency selection mode. full preserves the complete target "
            "list; profile_topk ranks targets from --target-profile-csv and keeps "
            "the requested top BLE/BR/EDR frequencies."
        ),
    )
    parse_parser.add_argument(
        "--target-profile-csv",
        default="",
        help="target_dsp_profile.csv used by --target-selection profile_topk.",
    )
    parse_parser.add_argument(
        "--target-profile-metric",
        choices=TARGET_SELECTION_METRICS,
        default="segment_copy_samples",
        help="Profile metric used to rank targets for profile_topk selection.",
    )
    parse_parser.add_argument(
        "--target-profile-top-ble",
        type=int,
        default=0,
        help="Keep top N BLE targets in profile_topk mode. Default 0 keeps all BLE targets.",
    )
    parse_parser.add_argument(
        "--target-profile-top-bredr",
        type=int,
        default=0,
        help="Keep top N BR/EDR targets in profile_topk mode. Default 0 keeps all BR/EDR targets.",
    )
    for action in parse_parser._actions:
        if action.dest == "chunk_samples":
            action.default = 16_000_000
            action.help = (
                "Wideband core samples per PFB chunk. Default 16M is tuned for the "
                "CUDA PFB path; lower this on memory-constrained machines."
            )
            break
    parse_parser.set_defaults(output_dir="artifacts/40m_pfb/parse")
    return parser


def add_timing(timings, key, seconds):
    timings[key] = timings.get(key, 0.0) + seconds


def is_target_profile_key(key):
    return isinstance(key, str) and key.startswith(TARGET_PROFILE_PREFIX)


def target_profile_timing_context(config, timings):
    if config.get("timing") or config.get("target_dsp_profile"):
        return timings
    return None


def target_profile_snapshot(timings):
    return {
        metric: timings.get(metric, 0.0)
        for metric in (
            "segment_count",
            "segment_copy_calls",
            "segment_copy_samples",
            "segment_copy_back_s",
            "segment_buffers",
            "segment_descriptor_count",
            "segment_input_build_s",
            "threshold_mask_s",
            "segment_materialize_s",
            "segment_materialize_samples",
            "segment_materialize_calls",
        )
    }


def add_target_profile(
    timings,
    config,
    protocol,
    freq_mhz,
    residual_hz,
    before,
    elapsed_s,
    scale=1.0,
):
    if not config.get("target_dsp_profile"):
        return
    key_prefix = (
        f"{TARGET_PROFILE_PREFIX}{protocol}|"
        f"{float(freq_mhz):.6f}|{float(residual_hz):.6f}|"
    )
    add_timing(timings, f"{key_prefix}targets", 1)
    add_timing(timings, f"{key_prefix}fused_target_s", elapsed_s * scale)
    for metric in (
        "segment_count",
        "segment_copy_calls",
        "segment_copy_samples",
        "segment_copy_back_s",
        "segment_buffers",
        "segment_descriptor_count",
        "segment_input_build_s",
        "threshold_mask_s",
        "segment_materialize_s",
        "segment_materialize_samples",
        "segment_materialize_calls",
    ):
        add_timing(
            timings,
            f"{key_prefix}{metric}",
            (timings.get(metric, 0.0) - before.get(metric, 0.0)) * scale,
        )


def add_timing_metrics_from_profile(timings, profile):
    for metric in (
        "segment_count",
        "segment_copy_calls",
        "segment_copy_samples",
        "segment_copy_back_s",
        "segment_buffers",
        "segment_descriptor_count",
        "segment_input_build_s",
        "threshold_mask_s",
        "segment_materialize_s",
        "segment_materialize_samples",
        "segment_materialize_calls",
    ):
        if metric in profile:
            add_timing(timings, metric, profile[metric])


def target_profile_rows(timings):
    rows = {}
    for key, value in timings.items():
        if not is_target_profile_key(key):
            continue
        parts = key.split("|")
        if len(parts) != 5:
            continue
        _prefix, protocol, freq_mhz, residual_hz, metric = parts
        if metric not in TARGET_PROFILE_METRICS:
            continue
        row_key = (protocol, freq_mhz, residual_hz)
        row = rows.setdefault(
            row_key,
            {
                "protocol": protocol,
                "freq_mhz": freq_mhz,
                "residual_hz": residual_hz,
                "targets": 0.0,
                "segment_count": 0.0,
                "segment_copy_calls": 0.0,
                "segment_copy_samples": 0.0,
                "segment_copy_back_s": 0.0,
                "segment_buffers": 0.0,
                "segment_descriptor_count": 0.0,
                "segment_input_build_s": 0.0,
                "threshold_mask_s": 0.0,
                "segment_materialize_s": 0.0,
                "segment_materialize_samples": 0.0,
                "segment_materialize_calls": 0.0,
                "fused_target_s": 0.0,
            },
        )
        row[metric] += value
    return sorted(
        rows.values(),
        key=lambda row: (
            -float(row["segment_copy_samples"]),
            -float(row["segment_copy_calls"]),
            row["protocol"],
            float(row["freq_mhz"]),
        ),
    )


def write_target_dsp_profile_csv(path, timings):
    rows = target_profile_rows(timings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol",
        "freq_mhz",
        "residual_hz",
        "targets",
        "segment_count",
        "segment_copy_calls",
        "segment_copy_samples",
        "segment_copy_back_s",
        "segment_buffers",
        "segment_descriptor_count",
        "segment_input_build_s",
        "threshold_mask_s",
        "segment_materialize_s",
        "segment_materialize_samples",
        "segment_materialize_calls",
        "fused_target_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def _freq_key(freq_mhz):
    return round(float(freq_mhz), 6)


def _format_freq_mhz(freq_mhz):
    return f"{float(freq_mhz):.6f}"


def load_target_profile_metrics(path, metric):
    if metric not in TARGET_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported target profile metric {metric!r}; choose one of "
            f"{', '.join(TARGET_SELECTION_METRICS)}."
        )
    if not path:
        raise ValueError("--target-profile-csv is required for profile_topk selection.")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target profile CSV not found: {path}")
    metrics = {"ble": {}, "bredr": {}}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Target profile CSV is empty: {path}")
        required = {"protocol", "freq_mhz", metric}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Target profile CSV {path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )
        for row in reader:
            protocol = row.get("protocol", "").strip().lower()
            if protocol not in metrics:
                continue
            try:
                freq = _freq_key(row["freq_mhz"])
                value = float(row.get(metric, "") or 0.0)
            except ValueError as exc:
                raise ValueError(f"Invalid target profile row in {path}: {row}") from exc
            metrics[protocol][freq] = metrics[protocol].get(freq, 0.0) + value
    return metrics


def select_profile_topk_targets(
    ble_freqs,
    bredr_freqs,
    profile_metrics,
    top_ble=0,
    top_bredr=0,
):
    def select_protocol(protocol, freqs, top_n):
        ranked = []
        protocol_metrics = profile_metrics.get(protocol, {})
        for freq in freqs:
            key = _freq_key(freq)
            ranked.append(
                {
                    "protocol": protocol,
                    "freq_mhz": float(freq),
                    "metric": float(protocol_metrics.get(key, 0.0)),
                    "selected": False,
                    "rank": 0,
                }
            )
        ranked.sort(key=lambda row: (-row["metric"], row["freq_mhz"]))
        keep_count = len(ranked) if top_n <= 0 or top_n >= len(ranked) else top_n
        selected_keys = set()
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            if rank <= keep_count:
                row["selected"] = True
                selected_keys.add(_freq_key(row["freq_mhz"]))
        selected = [freq for freq in freqs if _freq_key(freq) in selected_keys]
        return selected, ranked

    selected_ble, ble_rows = select_protocol("ble", ble_freqs, top_ble)
    selected_bredr, bredr_rows = select_protocol("bredr", bredr_freqs, top_bredr)
    return selected_ble, selected_bredr, ble_rows + bredr_rows


def write_target_selection_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_SELECTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "protocol": row["protocol"],
                    "freq_mhz": _format_freq_mhz(row["freq_mhz"]),
                    "selected": int(bool(row["selected"])),
                    "rank": int(row["rank"]),
                    "metric": f"{float(row['metric']):.9g}",
                }
            )


def apply_target_selection(args, ble_freqs, bredr_freqs):
    mode = getattr(args, "target_selection", "full")
    rows = []
    if mode == "full":
        for protocol, freqs in (("ble", ble_freqs), ("bredr", bredr_freqs)):
            for rank, freq in enumerate(freqs, start=1):
                rows.append(
                    {
                        "protocol": protocol,
                        "freq_mhz": float(freq),
                        "selected": True,
                        "rank": rank,
                        "metric": 0.0,
                    }
                )
        return ble_freqs, bredr_freqs, rows
    if mode != "profile_topk":
        raise ValueError(f"Unsupported --target-selection mode: {mode}")
    profile_metrics = load_target_profile_metrics(
        getattr(args, "target_profile_csv", ""),
        getattr(args, "target_profile_metric", "segment_copy_samples"),
    )
    return select_profile_topk_targets(
        ble_freqs,
        bredr_freqs,
        profile_metrics,
        top_ble=getattr(args, "target_profile_top_ble", 0),
        top_bredr=getattr(args, "target_profile_top_bredr", 0),
    )


def format_target_selection_summary(args, original_ble, original_bredr, selected_ble, selected_bredr):
    mode = getattr(args, "target_selection", "full")
    metric = getattr(args, "target_profile_metric", "segment_copy_samples")
    return (
        f"Target selection: mode={mode}, metric={metric}, "
        f"BLE={len(selected_ble)}/{len(original_ble)}, "
        f"BR/EDR={len(selected_bredr)}/{len(original_bredr)}."
    )


def format_target_dsp_profile_summary(timings, limit=8):
    rows = target_profile_rows(timings)[:limit]
    if not rows:
        return "Target DSP profile: no target profile rows collected."
    parts = []
    for row in rows:
        parts.append(
            f"{row['protocol']} {float(row['freq_mhz']):.3f}MHz "
            f"res={float(row['residual_hz']):+.1f}Hz "
            f"segments={int(row['segment_count'])} "
            f"calls={int(row['segment_copy_calls'])} "
            f"samples={int(row['segment_copy_samples'])} "
            f"copy={format_seconds(float(row['segment_copy_back_s']))} "
            f"buffers={int(row['segment_buffers'])} "
            f"desc={int(row['segment_descriptor_count'])} "
            f"input_build={format_seconds(float(row['segment_input_build_s']))} "
            f"mask={format_seconds(float(row['threshold_mask_s']))} "
            f"materialize={format_seconds(float(row['segment_materialize_s']))} "
            f"fused={format_seconds(float(row['fused_target_s']))}"
        )
    return "Target DSP profile top: " + " | ".join(parts)


def maybe_synchronize_cuda(config, timings=None):
    if config.get("cuda_sync_timing", False) and config["use_cuda"]:
        stage_start = perf_counter()
        with use_device(config["cuda_device"]):
            synchronize()
        if timings is not None:
            add_timing(timings, "gpu_sync_wait_s", perf_counter() - stage_start)


def use_compact_native_segment_input(protocol, config):
    if config.get("native_segment_input") != "compact":
        return False
    if protocol == "ble":
        return config.get("ble_parser_backend") == "cpp"
    return config.get("bredr_parser_backend") in ("cpp", "hybrid")


def segment_return_mode_for_protocol(protocol, config):
    return "compact" if use_compact_native_segment_input(protocol, config) else "segments"


def legacy_segments_for_python_parser(segments):
    if isinstance(segments, HostSegmentBatch):
        return segments.to_segments()
    if hasattr(segments, "to_segments") and all(
        hasattr(segments, name)
        for name in (
            "buffers",
            "buffer_indices",
            "starts",
            "ends",
            "offsets",
            "lengths",
            "segment_indices",
        )
    ):
        return segments.to_segments()
    return segments


def format_seconds(seconds):
    return f"{seconds:.3f}s"


def normalize_hex_id(value, digits):
    text = str(value).strip().replace("0x", "").replace("0X", "").upper()
    if not text:
        return ""
    try:
        return f"{int(text, 16):0{digits}X}"[-digits:]
    except ValueError:
        return text.zfill(digits)[-digits:]


def parse_hex_id_list(text, digits):
    if not text:
        return set()
    return {
        normalized
        for item in str(text).replace(";", ",").split(",")
        if (normalized := normalize_hex_id(item, digits))
    }


def resolve_known_packet_ids(args):
    ble_aas = parse_hex_id_list(args.known_ble_aa, 8)
    bredr_laps = parse_hex_id_list(args.known_bredr_lap, 6)
    for csv_path in args.known_packet_csv:
        with open(csv_path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("access_address"):
                    ble_aas.add(normalize_hex_id(row["access_address"], 8))
                if row.get("lap"):
                    bredr_laps.add(normalize_hex_id(row["lap"], 6))
    ble_aas.discard("")
    bredr_laps.discard("")
    return tuple(sorted(ble_aas)), tuple(sorted(bredr_laps))


def group_coarse_subband_tasks(
    channel_bank,
    args,
    chunk_start,
    decim,
    ble_cleanup_lpf,
    bredr_cleanup_lpf,
    ble_freqs,
    bredr_freqs,
):
    tasks = {}

    def add_target(protocol, freq_mhz):
        coarse_iq, coarse_offset_hz, residual_hz = select_coarse_subband(
            channel_bank, freq_mhz, args.center_freq, args.sample_rate
        )
        coarse_bin = int(round(coarse_offset_hz / (args.sample_rate / channel_bank.shape[0])))
        bank_index = coarse_bin % channel_bank.shape[0]
        task = tasks.setdefault(
            bank_index,
            {
                "bank_index": bank_index,
                "coarse_iq": coarse_iq,
                "ble_targets": [],
                "bredr_targets": [],
            },
        )
        task[f"{protocol}_targets"].append((freq_mhz, residual_hz))

    if not args.skip_ble:
        for freq_mhz in ble_freqs:
            add_target("ble", freq_mhz)
    if not args.skip_bredr:
        for freq_mhz in bredr_freqs:
            add_target("bredr", freq_mhz)

    config = {
        "chunk_start": chunk_start,
        "decim": decim,
        "sample_rate": args.sample_rate,
        "subband_sample_rate": args.subband_sample_rate,
        "use_cuda": args.use_cuda,
        "cuda_device": args.cuda_device,
        "cuda_batch_targets": args.cuda_batch_targets,
        "cuda_cleanup_backend": args.cuda_cleanup_backend,
        "cuda_fuse_target_dsp": args.cuda_fuse_target_dsp,
        "cuda_threshold_detect": args.cuda_threshold_detect,
        "cuda_segment_copy_merge_gap_samples": args.cuda_segment_copy_merge_gap_samples,
        "cuda_segment_copy_max_merged_samples": args.cuda_segment_copy_max_merged_samples,
        "cuda_target_dsp_materialization": args.cuda_target_dsp_materialization,
        "parse_executor": getattr(args, "_parse_executor", None),
        "parse_worker_chunk_segments": args.parse_worker_chunk_segments,
        "cpp_parser_threads": args.cpp_parser_threads,
        "ble_parser_backend": args.ble_parser_backend,
        "bredr_parser_backend": args.bredr_parser_backend,
        "native_segment_input": args.native_segment_input,
        "parallel_bredr_candidates": args.parallel_bredr_candidates,
        "native_br_batch_candidates": args.native_br_batch_candidates,
        "known_ble_aas": args.known_ble_aas,
        "known_bredr_laps": args.known_bredr_laps,
        "known_bredr_lap_sync_words": args.known_bredr_lap_sync_words,
        "known_bredr_lap_fast_path": args.known_bredr_lap_fast_path,
        "learned_parser_fast_path": args.learned_parser_fast_path,
        "learned_ble_aas": args.learned_ble_aas,
        "learned_ble_modes": args.learned_ble_modes,
        "learned_bredr_laps": args.learned_bredr_laps,
        "learned_bredr_lap_sync_words": args.learned_bredr_lap_sync_words,
        "cuda_known_candidate_filter": args.cuda_known_candidate_filter,
        "target_dsp_profile": getattr(args, "target_dsp_profile", False),
        "timing": args.timing,
        "cuda_sync_timing": getattr(args, "cuda_sync_timing", False),
        "freq_dev": args.freq_dev,
        "ble_threshold": args.ble_threshold,
        "ble_candidate_detector": getattr(args, "ble_candidate_detector", "fixed"),
        "ble_envelope_window_samples": getattr(args, "ble_envelope_window_samples", 8),
        "ble_start_noise_multiplier": getattr(args, "ble_start_noise_multiplier", 3.5),
        "ble_hold_noise_multiplier": getattr(args, "ble_hold_noise_multiplier", 1.8),
        "ble_candidate_gap_tolerance_samples": getattr(args, "ble_candidate_gap_tolerance_samples", 0),
        "ble_candidate_prepad_samples": getattr(args, "ble_candidate_prepad_samples", 0),
        "ble_candidate_postpad_samples": getattr(args, "ble_candidate_postpad_samples", 0),
        "ble_segment_min_len": args.ble_segment_min_len,
        "ble_score_threshold": args.ble_score_threshold,
        "br_threshold": args.br_threshold,
        "br_segment_min_len": args.br_segment_min_len,
        "br_cutoff": args.br_cutoff,
        "uap_cache": getattr(args, "_uap_cache", LockedUAPCache()),
    }
    return [
        tasks[key]
        | {
            "config": config,
            "ble_cleanup_lpf": ble_cleanup_lpf,
            "bredr_cleanup_lpf": bredr_cleanup_lpf,
        }
        for key in sorted(tasks)
    ]


def parse_coarse_subband_task(task):
    config = task["config"]
    coarse_iq = task["coarse_iq"]
    ble_cleanup_lpf = task["ble_cleanup_lpf"]
    bredr_cleanup_lpf = task["bredr_cleanup_lpf"]
    results = []
    timings = {}

    if config["use_cuda"] and config["cuda_batch_targets"]:
        _parse_cuda_target_batch(
            task["ble_targets"],
            "ble",
            coarse_iq,
            ble_cleanup_lpf,
            config,
            results,
            timings,
        )
        _parse_cuda_target_batch(
            task["bredr_targets"],
            "bredr",
            coarse_iq,
            bredr_cleanup_lpf,
            config,
            results,
            timings,
        )
    elif config["use_cuda"]:
        _parse_cuda_targets(
            task["ble_targets"],
            "ble",
            coarse_iq,
            ble_cleanup_lpf,
            config,
            results,
            timings,
        )
        _parse_cuda_targets(
            task["bredr_targets"],
            "bredr",
            coarse_iq,
            bredr_cleanup_lpf,
            config,
            results,
            timings,
        )
    else:
        _parse_cpu_targets(
            task["ble_targets"],
            "ble",
            coarse_iq,
            ble_cleanup_lpf,
            config,
            results,
            timings,
        )
        _parse_cpu_targets(
            task["bredr_targets"],
            "bredr",
            coarse_iq,
            bredr_cleanup_lpf,
            config,
            results,
            timings,
        )
    return task["bank_index"], results, timings


def build_coarse_subband_parse_inputs(task):
    config = task["config"]
    coarse_iq = task["coarse_iq"]
    ble_cleanup_lpf = task["ble_cleanup_lpf"]
    bredr_cleanup_lpf = task["bredr_cleanup_lpf"]
    parse_inputs = []
    timings = {}

    if config["use_cuda"] and config["cuda_batch_targets"]:
        _build_cuda_target_batch_parse_inputs(
            task["ble_targets"],
            "ble",
            coarse_iq,
            ble_cleanup_lpf,
            config,
            parse_inputs,
            timings,
        )
        _build_cuda_target_batch_parse_inputs(
            task["bredr_targets"],
            "bredr",
            coarse_iq,
            bredr_cleanup_lpf,
            config,
            parse_inputs,
            timings,
        )
    elif config["use_cuda"]:
        _build_cuda_target_parse_inputs(
            task["ble_targets"],
            "ble",
            coarse_iq,
            ble_cleanup_lpf,
            config,
            parse_inputs,
            timings,
        )
        _build_cuda_target_parse_inputs(
            task["bredr_targets"],
            "bredr",
            coarse_iq,
            bredr_cleanup_lpf,
            config,
            parse_inputs,
            timings,
        )
    else:
        _build_cpu_target_parse_inputs(
            task["ble_targets"],
            "ble",
            coarse_iq,
            ble_cleanup_lpf,
            config,
            parse_inputs,
            timings,
        )
        _build_cpu_target_parse_inputs(
            task["bredr_targets"],
            "bredr",
            coarse_iq,
            bredr_cleanup_lpf,
            config,
            parse_inputs,
            timings,
        )
    return task["bank_index"], parse_inputs, timings


def parse_coarse_subband_inputs(target_dsp_result):
    bank_index, parse_inputs, _input_timings = target_dsp_result
    timings = {}
    results = []
    for item in parse_inputs:
        if item["kind"] == "segments":
            _parse_host_segments(
                item["protocol"],
                item["freq_mhz"],
                item["segments"],
                item["config"],
                results,
                timings,
            )
        elif item["kind"] == "sub_iq":
            _parse_host_target(
                item["protocol"],
                item["freq_mhz"],
                item["sub_iq"],
                item["config"],
                results,
                timings,
            )
        elif item["kind"] == "packets":
            results.append((item["protocol"], item["freq_mhz"], item["packets"]))
        else:
            raise ValueError(f"Unsupported parse input kind: {item['kind']}")
    return bank_index, results, timings


def _build_cuda_target_parse_inputs(targets, protocol, coarse_iq, cleanup_lpf, config, parse_inputs, timings):
    if config["cuda_fuse_target_dsp"] and config["cuda_threshold_detect"]:
        grouped_targets = []
        by_residual = {}
        for freq_mhz, residual_hz in targets:
            key = round(float(residual_hz), 6)
            if key not in by_residual:
                by_residual[key] = [residual_hz, []]
                grouped_targets.append(by_residual[key])
            by_residual[key][1].append(freq_mhz)

        if len(grouped_targets) < len(targets):
            add_timing(timings, "fused_target_reuse", len(targets) - len(grouped_targets))

        for residual_hz, freq_mhz_values in grouped_targets:
            if protocol == "ble":
                threshold = config["ble_threshold"]
                segment_min_len = config["ble_segment_min_len"]
            else:
                threshold = config["br_threshold"]
                segment_min_len = config["br_segment_min_len"]

            stage_start = perf_counter()
            before_profile = target_profile_snapshot(timings)
            segments = shift_cleanup_threshold_segments(
                coarse_iq,
                cleanup_lpf,
                threshold,
                segment_min_len,
                config["chunk_start"],
                config["decim"],
                config["sample_rate"],
                residual_hz,
                use_cuda=True,
                device_id=config["cuda_device"],
                timing_context=target_profile_timing_context(config, timings),
                segment_copy_merge_gap_samples=config["cuda_segment_copy_merge_gap_samples"],
                segment_copy_max_merged_samples=config["cuda_segment_copy_max_merged_samples"],
                known_candidate_protocol=(
                    protocol
                    if config["cuda_known_candidate_filter"]
                    else None
                ),
                known_ble_aas=config["known_ble_aas"],
                known_bredr_lap_sync_words=config["known_bredr_lap_sync_words"],
                segment_return_mode=segment_return_mode_for_protocol(protocol, config),
                target_dsp_materialization=config["cuda_target_dsp_materialization"],
            )
            maybe_synchronize_cuda(config, timings)
            elapsed_s = perf_counter() - stage_start
            add_timing(timings, "fused_target_s", elapsed_s)
            profile_scale = 1.0 / max(1, len(freq_mhz_values))
            for freq_mhz in freq_mhz_values:
                add_target_profile(
                    timings,
                    config,
                    protocol,
                    freq_mhz,
                    residual_hz,
                    before_profile,
                    elapsed_s,
                    scale=profile_scale,
                )
                parse_inputs.append(
                    {
                        "kind": "segments",
                        "protocol": protocol,
                        "freq_mhz": freq_mhz,
                        "segments": segments,
                        "config": config,
                    }
                )
                add_timing(timings, "targets", 1)
        return

    for freq_mhz, residual_hz in targets:
        stage_start = perf_counter()
        sub_iq = shift_within_subband(
            coarse_iq,
            config["chunk_start"],
            config["decim"],
            config["sample_rate"],
            residual_hz,
            use_cuda=True,
            device_id=config["cuda_device"],
            return_host=False,
        )
        maybe_synchronize_cuda(config, timings)
        add_timing(timings, "shift_s", perf_counter() - stage_start)

        stage_start = perf_counter()
        sub_iq = apply_cleanup_lpf(
            sub_iq,
            cleanup_lpf,
            use_cuda=True,
            device_id=config["cuda_device"],
            return_host=not config["cuda_threshold_detect"],
            cuda_fir_backend=config["cuda_cleanup_backend"],
        )
        maybe_synchronize_cuda(config, timings)
        add_timing(timings, "cleanup_s", perf_counter() - stage_start)

        if config["cuda_threshold_detect"]:
            _build_cuda_thresholded_parse_input(
                protocol,
                freq_mhz,
                sub_iq,
                config,
                parse_inputs,
                timings,
            )
        else:
            parse_inputs.append(
                {
                    "kind": "sub_iq",
                    "protocol": protocol,
                    "freq_mhz": freq_mhz,
                    "sub_iq": sub_iq,
                    "config": config,
                }
            )
        add_timing(timings, "targets", 1)


def _build_cuda_target_batch_parse_inputs(targets, protocol, coarse_iq, cleanup_lpf, config, parse_inputs, timings):
    if not targets:
        return

    if config["cuda_fuse_target_dsp"] and config["cuda_threshold_detect"]:
        if protocol == "ble":
            threshold = config["ble_threshold"]
            segment_min_len = config["ble_segment_min_len"]
        else:
            threshold = config["br_threshold"]
            segment_min_len = config["br_segment_min_len"]

        grouped_targets = []
        by_residual = {}
        for freq_mhz, residual_hz in targets:
            key = round(float(residual_hz), 6)
            if key not in by_residual:
                by_residual[key] = [residual_hz, []]
                grouped_targets.append(by_residual[key])
            by_residual[key][1].append(freq_mhz)

        if len(grouped_targets) < len(targets):
            add_timing(timings, "fused_target_reuse", len(targets) - len(grouped_targets))
        add_timing(timings, "fused_target_batches", 1)
        add_timing(timings, "fused_target_batch_items", len(grouped_targets))
        if len(grouped_targets) > 1:
            add_timing(timings, "fused_target_shared_input_batches", 1)

        stage_start = perf_counter()
        batch_target_profiles = (
            [{} for _residual_hz, _freqs in grouped_targets]
            if config.get("target_dsp_profile")
            else None
        )
        segment_batches = shift_cleanup_threshold_segments_batch(
            coarse_iq,
            cleanup_lpf,
            threshold,
            segment_min_len,
            config["chunk_start"],
            config["decim"],
            config["sample_rate"],
            [residual_hz for residual_hz, _freqs in grouped_targets],
            use_cuda=True,
            device_id=config["cuda_device"],
            timing_context=target_profile_timing_context(config, timings),
            candidate_detector=(
                config["ble_candidate_detector"] if protocol == "ble" else "fixed"
            ),
            envelope_window_samples=config["ble_envelope_window_samples"],
            start_noise_multiplier=config["ble_start_noise_multiplier"],
            hold_noise_multiplier=config["ble_hold_noise_multiplier"],
            candidate_gap_tolerance_samples=config["ble_candidate_gap_tolerance_samples"],
            candidate_prepad_samples=config["ble_candidate_prepad_samples"],
            candidate_postpad_samples=config["ble_candidate_postpad_samples"],
            segment_copy_merge_gap_samples=config["cuda_segment_copy_merge_gap_samples"],
            segment_copy_max_merged_samples=config["cuda_segment_copy_max_merged_samples"],
            known_candidate_protocol=(
                protocol
                if config["cuda_known_candidate_filter"]
                else None
            ),
            known_ble_aas=config["known_ble_aas"],
            known_bredr_lap_sync_words=config["known_bredr_lap_sync_words"],
            per_target_timing_contexts=batch_target_profiles,
            segment_return_mode=segment_return_mode_for_protocol(protocol, config),
            target_dsp_materialization=config["cuda_target_dsp_materialization"],
        )
        maybe_synchronize_cuda(config, timings)
        elapsed_s = perf_counter() - stage_start
        add_timing(timings, "fused_target_s", elapsed_s)
        per_residual_count = max(1, len(grouped_targets))

        for target_index, ((_residual_hz, freq_mhz_values), segments) in enumerate(
            zip(grouped_targets, segment_batches)
        ):
            target_profile = (
                batch_target_profiles[target_index]
                if batch_target_profiles is not None
                else {}
            )
            if batch_target_profiles is not None:
                before_profile = target_profile_snapshot(timings)
                add_timing_metrics_from_profile(timings, target_profile)
            else:
                before_profile = {}
            profile_scale = 1.0 / max(1, len(freq_mhz_values))
            for freq_mhz in freq_mhz_values:
                add_target_profile(
                    timings,
                    config,
                    protocol,
                    freq_mhz,
                    _residual_hz,
                    before_profile,
                    elapsed_s / per_residual_count,
                    scale=profile_scale,
                )
                parse_inputs.append(
                    {
                        "kind": "segments",
                        "protocol": protocol,
                        "freq_mhz": freq_mhz,
                        "segments": segments,
                        "config": config,
                    }
                )
                add_timing(timings, "targets", 1)
        return

    freq_mhz_values = [freq_mhz for freq_mhz, _residual_hz in targets]
    residual_hz_values = [residual_hz for _freq_mhz, residual_hz in targets]

    stage_start = perf_counter()
    sub_iq_batch = shift_within_subband_batch(
        coarse_iq,
        config["chunk_start"],
        config["decim"],
        config["sample_rate"],
        residual_hz_values,
        use_cuda=True,
        device_id=config["cuda_device"],
        return_host=False,
    )
    maybe_synchronize_cuda(config, timings)
    add_timing(timings, "shift_s", perf_counter() - stage_start)

    stage_start = perf_counter()
    sub_iq_batch = apply_cleanup_lpf(
        sub_iq_batch,
        cleanup_lpf,
        use_cuda=True,
        device_id=config["cuda_device"],
        return_host=not config["cuda_threshold_detect"],
        cuda_fir_backend=config["cuda_cleanup_backend"],
    )
    maybe_synchronize_cuda(config, timings)
    add_timing(timings, "cleanup_s", perf_counter() - stage_start)

    for target_index, (freq_mhz, residual_hz) in enumerate(
        zip(freq_mhz_values, residual_hz_values)
    ):
        sub_iq = sub_iq_batch[target_index]
        if config["cuda_threshold_detect"]:
            _build_cuda_thresholded_parse_input(
                protocol,
                freq_mhz,
                residual_hz,
                sub_iq,
                config,
                parse_inputs,
                timings,
            )
        else:
            parse_inputs.append(
                {
                    "kind": "sub_iq",
                    "protocol": protocol,
                    "freq_mhz": freq_mhz,
                    "sub_iq": sub_iq,
                    "config": config,
                }
            )
        add_timing(timings, "targets", 1)


def _build_cpu_target_parse_inputs(targets, protocol, coarse_iq, cleanup_lpf, config, parse_inputs, timings):
    for freq_mhz, residual_hz in targets:
        stage_start = perf_counter()
        sub_iq = shift_within_subband(
            coarse_iq,
            config["chunk_start"],
            config["decim"],
            config["sample_rate"],
            residual_hz,
        )
        add_timing(timings, "shift_s", perf_counter() - stage_start)

        stage_start = perf_counter()
        sub_iq = apply_cleanup_lpf(sub_iq, cleanup_lpf)
        add_timing(timings, "cleanup_s", perf_counter() - stage_start)

        parse_inputs.append(
            {
                "kind": "sub_iq",
                "protocol": protocol,
                "freq_mhz": freq_mhz,
                "sub_iq": sub_iq,
                "config": config,
            }
        )
        add_timing(timings, "targets", 1)


def _build_cuda_thresholded_parse_input(protocol, freq_mhz, residual_hz, sub_iq, config, parse_inputs, timings):
    if protocol == "ble":
        threshold = config["ble_threshold"]
        segment_min_len = config["ble_segment_min_len"]
    else:
        threshold = config["br_threshold"]
        segment_min_len = config["br_segment_min_len"]

    stage_start = perf_counter()
    before_profile = target_profile_snapshot(timings)
    segments = threshold_segments_with_iq(
        sub_iq,
        threshold,
        segment_min_len,
        use_cuda=True,
        device_id=config["cuda_device"],
        known_candidate_protocol=(
            protocol
            if config["cuda_known_candidate_filter"]
            else None
        ),
        known_ble_aas=config["known_ble_aas"],
        known_bredr_lap_sync_words=config["known_bredr_lap_sync_words"],
        sample_rate=config["subband_sample_rate"],
        timing_context=target_profile_timing_context(config, timings),
        segment_copy_merge_gap_samples=config["cuda_segment_copy_merge_gap_samples"],
        segment_copy_max_merged_samples=config["cuda_segment_copy_max_merged_samples"],
        segment_return_mode=segment_return_mode_for_protocol(protocol, config),
    )
    maybe_synchronize_cuda(config, timings)
    elapsed_s = perf_counter() - stage_start
    add_timing(timings, "threshold_s", elapsed_s)
    add_target_profile(
        timings,
        config,
        protocol,
        freq_mhz,
        residual_hz,
        before_profile,
        elapsed_s,
    )
    parse_inputs.append(
        {
            "kind": "segments",
            "protocol": protocol,
            "freq_mhz": freq_mhz,
            "segments": segments,
            "config": config,
        }
    )


def _parse_cuda_targets(targets, protocol, coarse_iq, cleanup_lpf, config, results, timings):
    if config["cuda_fuse_target_dsp"] and config["cuda_threshold_detect"]:
        grouped_targets = []
        by_residual = {}
        for freq_mhz, residual_hz in targets:
            key = round(float(residual_hz), 6)
            if key not in by_residual:
                by_residual[key] = [residual_hz, []]
                grouped_targets.append(by_residual[key])
            by_residual[key][1].append(freq_mhz)

        if len(grouped_targets) < len(targets):
            add_timing(timings, "fused_target_reuse", len(targets) - len(grouped_targets))

        for residual_hz, freq_mhz_values in grouped_targets:
            if protocol == "ble":
                threshold = config["ble_threshold"]
                segment_min_len = config["ble_segment_min_len"]
            else:
                threshold = config["br_threshold"]
                segment_min_len = config["br_segment_min_len"]

            stage_start = perf_counter()
            before_profile = target_profile_snapshot(timings)
            segments = shift_cleanup_threshold_segments(
                coarse_iq,
                cleanup_lpf,
                threshold,
                segment_min_len,
                config["chunk_start"],
                config["decim"],
                config["sample_rate"],
                residual_hz,
                use_cuda=True,
                device_id=config["cuda_device"],
                timing_context=target_profile_timing_context(config, timings),
                segment_copy_merge_gap_samples=config["cuda_segment_copy_merge_gap_samples"],
                segment_copy_max_merged_samples=config["cuda_segment_copy_max_merged_samples"],
                known_candidate_protocol=(
                    protocol
                    if config["cuda_known_candidate_filter"]
                    else None
                ),
                known_ble_aas=config["known_ble_aas"],
                known_bredr_lap_sync_words=config["known_bredr_lap_sync_words"],
                segment_return_mode=segment_return_mode_for_protocol(protocol, config),
                target_dsp_materialization=config["cuda_target_dsp_materialization"],
            )
            maybe_synchronize_cuda(config, timings)
            elapsed_s = perf_counter() - stage_start
            add_timing(timings, "fused_target_s", elapsed_s)
            profile_scale = 1.0 / max(1, len(freq_mhz_values))
            for freq_mhz in freq_mhz_values:
                add_target_profile(
                    timings,
                    config,
                    protocol,
                    freq_mhz,
                    residual_hz,
                    before_profile,
                    elapsed_s,
                    scale=profile_scale,
                )
                _parse_host_segments(protocol, freq_mhz, segments, config, results, timings)
                add_timing(timings, "targets", 1)
        return

    for freq_mhz, residual_hz in targets:
        stage_start = perf_counter()
        sub_iq = shift_within_subband(
            coarse_iq,
            config["chunk_start"],
            config["decim"],
            config["sample_rate"],
            residual_hz,
            use_cuda=True,
            device_id=config["cuda_device"],
            return_host=False,
        )
        maybe_synchronize_cuda(config, timings)
        add_timing(timings, "shift_s", perf_counter() - stage_start)

        stage_start = perf_counter()
        sub_iq = apply_cleanup_lpf(
            sub_iq,
            cleanup_lpf,
            use_cuda=True,
            device_id=config["cuda_device"],
            return_host=not config["cuda_threshold_detect"],
            cuda_fir_backend=config["cuda_cleanup_backend"],
        )
        maybe_synchronize_cuda(config, timings)
        add_timing(timings, "cleanup_s", perf_counter() - stage_start)

        if config["cuda_threshold_detect"]:
            _parse_cuda_thresholded_target(
                protocol,
                freq_mhz,
                residual_hz,
                sub_iq,
                config,
                results,
                timings,
            )
        else:
            _parse_host_target(protocol, freq_mhz, sub_iq, config, results, timings)
        add_timing(timings, "targets", 1)


def _parse_cuda_target_batch(targets, protocol, coarse_iq, cleanup_lpf, config, results, timings):
    if not targets:
        return

    if config["cuda_fuse_target_dsp"] and config["cuda_threshold_detect"]:
        if protocol == "ble":
            threshold = config["ble_threshold"]
            segment_min_len = config["ble_segment_min_len"]
        else:
            threshold = config["br_threshold"]
            segment_min_len = config["br_segment_min_len"]

        grouped_targets = []
        by_residual = {}
        for freq_mhz, residual_hz in targets:
            key = round(float(residual_hz), 6)
            if key not in by_residual:
                by_residual[key] = [residual_hz, []]
                grouped_targets.append(by_residual[key])
            by_residual[key][1].append(freq_mhz)

        if len(grouped_targets) < len(targets):
            add_timing(timings, "fused_target_reuse", len(targets) - len(grouped_targets))
        add_timing(timings, "fused_target_batches", 1)
        add_timing(timings, "fused_target_batch_items", len(grouped_targets))
        if len(grouped_targets) > 1:
            add_timing(timings, "fused_target_shared_input_batches", 1)

        stage_start = perf_counter()
        batch_target_profiles = (
            [{} for _residual_hz, _freqs in grouped_targets]
            if config.get("target_dsp_profile")
            else None
        )
        segment_batches = shift_cleanup_threshold_segments_batch(
            coarse_iq,
            cleanup_lpf,
            threshold,
            segment_min_len,
            config["chunk_start"],
            config["decim"],
            config["sample_rate"],
            [residual_hz for residual_hz, _freqs in grouped_targets],
            use_cuda=True,
            device_id=config["cuda_device"],
            timing_context=target_profile_timing_context(config, timings),
            candidate_detector=(
                config["ble_candidate_detector"] if protocol == "ble" else "fixed"
            ),
            envelope_window_samples=config["ble_envelope_window_samples"],
            start_noise_multiplier=config["ble_start_noise_multiplier"],
            hold_noise_multiplier=config["ble_hold_noise_multiplier"],
            candidate_gap_tolerance_samples=config["ble_candidate_gap_tolerance_samples"],
            candidate_prepad_samples=config["ble_candidate_prepad_samples"],
            candidate_postpad_samples=config["ble_candidate_postpad_samples"],
            segment_copy_merge_gap_samples=config["cuda_segment_copy_merge_gap_samples"],
            segment_copy_max_merged_samples=config["cuda_segment_copy_max_merged_samples"],
            known_candidate_protocol=(
                protocol
                if config["cuda_known_candidate_filter"]
                else None
            ),
            known_ble_aas=config["known_ble_aas"],
            known_bredr_lap_sync_words=config["known_bredr_lap_sync_words"],
            per_target_timing_contexts=batch_target_profiles,
            segment_return_mode=segment_return_mode_for_protocol(protocol, config),
            target_dsp_materialization=config["cuda_target_dsp_materialization"],
        )
        maybe_synchronize_cuda(config, timings)
        elapsed_s = perf_counter() - stage_start
        add_timing(timings, "fused_target_s", elapsed_s)
        per_residual_count = max(1, len(grouped_targets))

        for target_index, ((_residual_hz, freq_mhz_values), segments) in enumerate(
            zip(grouped_targets, segment_batches)
        ):
            target_profile = (
                batch_target_profiles[target_index]
                if batch_target_profiles is not None
                else {}
            )
            if batch_target_profiles is not None:
                before_profile = target_profile_snapshot(timings)
                add_timing_metrics_from_profile(timings, target_profile)
            else:
                before_profile = {}
            profile_scale = 1.0 / max(1, len(freq_mhz_values))
            for freq_mhz in freq_mhz_values:
                add_target_profile(
                    timings,
                    config,
                    protocol,
                    freq_mhz,
                    _residual_hz,
                    before_profile,
                    elapsed_s / per_residual_count,
                    scale=profile_scale,
                )
                _parse_host_segments(protocol, freq_mhz, segments, config, results, timings)
                add_timing(timings, "targets", 1)
        return

    freq_mhz_values = [freq_mhz for freq_mhz, _residual_hz in targets]
    residual_hz_values = [residual_hz for _freq_mhz, residual_hz in targets]

    stage_start = perf_counter()
    sub_iq_batch = shift_within_subband_batch(
        coarse_iq,
        config["chunk_start"],
        config["decim"],
        config["sample_rate"],
        residual_hz_values,
        use_cuda=True,
        device_id=config["cuda_device"],
        return_host=False,
    )
    maybe_synchronize_cuda(config, timings)
    add_timing(timings, "shift_s", perf_counter() - stage_start)

    stage_start = perf_counter()
    sub_iq_batch = apply_cleanup_lpf(
        sub_iq_batch,
        cleanup_lpf,
        use_cuda=True,
        device_id=config["cuda_device"],
        return_host=not config["cuda_threshold_detect"],
        cuda_fir_backend=config["cuda_cleanup_backend"],
    )
    maybe_synchronize_cuda(config, timings)
    add_timing(timings, "cleanup_s", perf_counter() - stage_start)

    for target_index, (freq_mhz, residual_hz) in enumerate(
        zip(freq_mhz_values, residual_hz_values)
    ):
        sub_iq = sub_iq_batch[target_index]
        if config["cuda_threshold_detect"]:
            _parse_cuda_thresholded_target(
                protocol,
                freq_mhz,
                residual_hz,
                sub_iq,
                config,
                results,
                timings,
            )
        else:
            _parse_host_target(protocol, freq_mhz, sub_iq, config, results, timings)
        add_timing(timings, "targets", 1)


def _parse_cpu_targets(targets, protocol, coarse_iq, cleanup_lpf, config, results, timings):
    for freq_mhz, residual_hz in targets:
        stage_start = perf_counter()
        sub_iq = shift_within_subband(
            coarse_iq,
            config["chunk_start"],
            config["decim"],
            config["sample_rate"],
            residual_hz,
        )
        add_timing(timings, "shift_s", perf_counter() - stage_start)

        stage_start = perf_counter()
        sub_iq = apply_cleanup_lpf(sub_iq, cleanup_lpf)
        add_timing(timings, "cleanup_s", perf_counter() - stage_start)

        _parse_host_target(protocol, freq_mhz, sub_iq, config, results, timings)
        add_timing(timings, "targets", 1)


def _parse_host_target(protocol, freq_mhz, sub_iq, config, results, timings):
    stage_start = perf_counter()
    if protocol == "ble":
        if config["ble_parser_backend"] == "cpp":
            raise NotImplementedError(
                "--ble-parser-backend cpp currently requires segmented input from "
                "CUDA threshold detect."
            )
        else:
            packets = parse_ble_packets(
                sub_iq,
                config["subband_sample_rate"],
                freq_mhz * 1e6,
                config["freq_dev"],
                config["ble_threshold"],
                config["ble_segment_min_len"],
                config["ble_score_threshold"],
                known_access_addresses=config["known_ble_aas"],
                learned_access_addresses=(
                    config["learned_ble_aas"] if config["learned_parser_fast_path"] else None
                ),
                learned_ble_modes=(
                    config["learned_ble_modes"] if config["learned_parser_fast_path"] else None
                ),
            )
        add_timing(timings, "ble_parse_s", perf_counter() - stage_start)
    else:
        packets = parse_btclassic_packets(
            sub_iq,
            config["subband_sample_rate"],
            freq_mhz * 1e6,
            config["freq_dev"],
            config["br_threshold"],
            config["br_segment_min_len"],
            config["br_cutoff"],
            known_laps=config["known_bredr_laps"],
            known_lap_fast_path=config["known_bredr_lap_fast_path"],
            learned_laps=(
                config["learned_bredr_laps"] if config["learned_parser_fast_path"] else None
            ),
            learned_lap_sync_words=(
                config["learned_bredr_lap_sync_words"]
                if config["learned_parser_fast_path"]
                else None
            ),
            uap_cache=config["uap_cache"],
        )
        add_timing(timings, "bredr_parse_s", perf_counter() - stage_start)
    results.append((protocol, freq_mhz, packets))


def _parse_cuda_thresholded_target(protocol, freq_mhz, residual_hz, sub_iq, config, results, timings):
    if protocol == "ble":
        threshold = config["ble_threshold"]
        segment_min_len = config["ble_segment_min_len"]
    else:
        threshold = config["br_threshold"]
        segment_min_len = config["br_segment_min_len"]

    stage_start = perf_counter()
    before_profile = target_profile_snapshot(timings)
    segments = threshold_segments_with_iq(
        sub_iq,
        threshold,
        segment_min_len,
        use_cuda=True,
        device_id=config["cuda_device"],
        known_candidate_protocol=(
            protocol
            if config["cuda_known_candidate_filter"]
            else None
        ),
        known_ble_aas=config["known_ble_aas"],
        known_bredr_lap_sync_words=config["known_bredr_lap_sync_words"],
        sample_rate=config["subband_sample_rate"],
        timing_context=target_profile_timing_context(config, timings),
        segment_copy_merge_gap_samples=config["cuda_segment_copy_merge_gap_samples"],
        segment_copy_max_merged_samples=config["cuda_segment_copy_max_merged_samples"],
        segment_return_mode=segment_return_mode_for_protocol(protocol, config),
    )
    maybe_synchronize_cuda(config, timings)
    elapsed_s = perf_counter() - stage_start
    add_timing(timings, "threshold_s", elapsed_s)
    add_target_profile(
        timings,
        config,
        protocol,
        freq_mhz,
        residual_hz,
        before_profile,
        elapsed_s,
    )

    _parse_host_segments(protocol, freq_mhz, segments, config, results, timings)


def _parse_host_segments(protocol, freq_mhz, segments, config, results, timings):
    stage_start = perf_counter()
    submit_start = perf_counter()
    parse_executor = config.get("parse_executor")
    compact_segments = isinstance(segments, HostSegmentBatch) or (
        hasattr(segments, "to_segments")
        and hasattr(segments, "buffer_indices")
        and hasattr(segments, "lengths")
    )
    can_parallel_parse = (
        not compact_segments
        and (
            (protocol == "ble" and config["ble_parser_backend"] == "python")
            or config.get("parallel_bredr_candidates")
        )
    )
    if parse_executor is not None and len(segments) > 1 and can_parallel_parse:
        packets = _parse_host_segments_parallel(
            parse_executor,
            protocol,
            freq_mhz,
            segments,
            config,
            timings,
        )
    elif protocol == "ble":
        add_timing(timings, "parser_submit_s", perf_counter() - submit_start)
        common_kwargs = {
            "known_access_addresses": config["known_ble_aas"],
            "learned_access_addresses": (
                config["learned_ble_aas"] if config["learned_parser_fast_path"] else None
            ),
            "learned_ble_modes": (
                config["learned_ble_modes"] if config["learned_parser_fast_path"] else None
            ),
        }
        if config["ble_parser_backend"] == "cpp":
            packets = parse_ble_packet_segments_native(
                segments,
                config["subband_sample_rate"],
                freq_mhz * 1e6,
                config["freq_dev"],
                config["ble_score_threshold"],
                thread_count=config["cpp_parser_threads"],
                timing_context=timings,
                **common_kwargs,
            )
        else:
            packets = parse_ble_packet_segments(
                legacy_segments_for_python_parser(segments),
                config["subband_sample_rate"],
                freq_mhz * 1e6,
                config["freq_dev"],
                config["ble_score_threshold"],
                **common_kwargs,
            )
    else:
        add_timing(timings, "parser_submit_s", perf_counter() - submit_start)
        if config["bredr_parser_backend"] == "cpp":
            if config["learned_parser_fast_path"]:
                raise NotImplementedError(
                    "--bredr-parser-backend cpp does not support learned parser "
                    "fast path yet."
                )
            packets = parse_btclassic_packet_segments_native_finalized(
                segments,
                config["subband_sample_rate"],
                freq_mhz * 1e6,
                config["freq_dev"],
                config["br_cutoff"],
                known_laps=config["known_bredr_laps"],
                known_lap_fast_path=config["known_bredr_lap_fast_path"],
                thread_count=config["cpp_parser_threads"],
                uap_cache=config["uap_cache"],
                timing_context=timings,
            )
        elif config["bredr_parser_backend"] == "hybrid":
            if config["learned_parser_fast_path"]:
                raise NotImplementedError(
                    "--bredr-parser-backend hybrid does not support learned parser "
                    "fast path yet."
                )
            try:
                packets = parse_btclassic_packet_segments_native_finalized(
                    segments,
                    config["subband_sample_rate"],
                    freq_mhz * 1e6,
                    config["freq_dev"],
                    config["br_cutoff"],
                    known_laps=config["known_bredr_laps"],
                    known_lap_fast_path=config["known_bredr_lap_fast_path"],
                    thread_count=config["cpp_parser_threads"],
                    uap_cache=config["uap_cache"],
                    timing_context=timings,
                )
            except NotImplementedError:
                candidates = build_btclassic_packet_candidates_native(
                    segments,
                    config["subband_sample_rate"],
                    freq_mhz * 1e6,
                    config["freq_dev"],
                    config["br_cutoff"],
                    known_laps=config["known_bredr_laps"],
                    known_lap_fast_path=config["known_bredr_lap_fast_path"],
                    thread_count=config["cpp_parser_threads"],
                    batch_candidates=config["native_br_batch_candidates"],
                    timing_context=timings,
                )
                packets = finalize_btclassic_packet_candidates(
                    candidates,
                    config["subband_sample_rate"],
                    freq_mhz * 1e6,
                    uap_cache=config["uap_cache"],
                )
        else:
            packets = parse_btclassic_packet_segments(
                legacy_segments_for_python_parser(segments),
                config["subband_sample_rate"],
                freq_mhz * 1e6,
                config["freq_dev"],
                config["br_cutoff"],
                known_laps=config["known_bredr_laps"],
                known_lap_fast_path=config["known_bredr_lap_fast_path"],
                learned_laps=(
                    config["learned_bredr_laps"] if config["learned_parser_fast_path"] else None
                ),
                learned_lap_sync_words=(
                    config["learned_bredr_lap_sync_words"]
                    if config["learned_parser_fast_path"]
                    else None
                ),
                uap_cache=config["uap_cache"],
            )
    if protocol == "ble":
        add_timing(timings, "ble_parse_s", perf_counter() - stage_start)
    else:
        add_timing(timings, "bredr_parse_s", perf_counter() - stage_start)
    results.append((protocol, freq_mhz, packets))


def _parse_host_segments_parallel(parse_executor, protocol, freq_mhz, segments, config, timings):
    chunk_size = max(1, int(config["parse_worker_chunk_segments"]))
    submit_start = perf_counter()
    jobs = []
    for start_index in range(0, len(segments), chunk_size):
        jobs.append(
            (
                protocol,
                freq_mhz,
                segments[start_index : start_index + chunk_size],
                start_index,
                config["subband_sample_rate"],
                config["freq_dev"],
                config["ble_score_threshold"],
                config["br_cutoff"],
                config["known_ble_aas"],
                config["known_bredr_laps"],
                config["known_bredr_lap_fast_path"],
            )
        )
    futures = [
        parse_executor.submit(_parse_host_segment_chunk_worker, job)
        for job in jobs
    ]
    add_timing(timings, "parser_submit_s", perf_counter() - submit_start)
    chunk_results = [future.result() for future in futures]
    packets = []
    for chunk_packets in chunk_results:
        packets.extend(chunk_packets)
    if protocol == "ble":
        return packets
    return finalize_btclassic_packet_candidates(
        packets,
        config["subband_sample_rate"],
        freq_mhz * 1e6,
        uap_cache=config["uap_cache"],
    )


def _parse_host_segment_chunk_worker(job):
    (
        protocol,
        freq_mhz,
        segments,
        segment_index_offset,
        sample_rate,
        freq_dev,
        ble_score_threshold,
        br_cutoff,
        known_ble_aas,
        known_bredr_laps,
        known_bredr_lap_fast_path,
    ) = job
    center_freq = freq_mhz * 1e6
    if protocol == "ble":
        return parse_ble_packet_segments(
            segments,
            sample_rate,
            center_freq,
            freq_dev,
            ble_score_threshold,
            segment_index_offset=segment_index_offset,
            known_access_addresses=known_ble_aas,
        )
    return build_btclassic_packet_candidates(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        br_cutoff,
        segment_index_offset=segment_index_offset,
        known_laps=known_bredr_laps,
        known_lap_fast_path=known_bredr_lap_fast_path,
    )


def finalize_task_results(
    task_results,
    args,
    chunk_start,
    core_start,
    core_end,
    filter_delay,
    decim,
    timestamp_mode,
    rx_time_tags,
):
    ble_rows = []
    bt_rows = []
    for _bank_index, results, _timings in sorted(task_results):
        for protocol, freq_mhz, packets in results:
            for packet in packets:
                if not legacy.keep_packet_in_core(
                    packet, chunk_start, core_start, core_end, decim, filter_delay
                ):
                    continue
                packet = legacy.convert_packet_timestamps(
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
                if protocol == "ble":
                    ble_id = packet.get("ble_device_address") or packet.get("access_address")
                    packet["parse_result"] = (
                        f"BLE PFB ch={packet.get('channel')} freq={freq_mhz:.3f}MHz "
                        f"AA={packet.get('access_address')} dev={ble_id} len={packet.get('payload_len')} "
                        f"score={packet.get('confidence_score')}"
                    )
                    ble_rows.append(packet)
                    if not args.quiet:
                        print(
                            f"BLE PFB | ch={packet.get('channel')} | freq={freq_mhz:.3f} MHz | "
                            f"ts_us={packet['timestamp_us']} | AA={packet.get('access_address')} | dev={ble_id} | "
                            f"len={packet.get('payload_len')} | score={packet.get('confidence_score')}"
                        )
                    continue

                bredr_channel = int(round(freq_mhz - 2402.0))
                packet["channel"] = bredr_channel
                packet["parse_result"] = (
                    f"BR/EDR PFB ch={bredr_channel} freq={freq_mhz:.3f}MHz "
                    f"LAP={packet.get('lap')} header={packet.get('packet_header_info')}"
                )
                bt_rows.append(packet)
                if not args.quiet:
                    print(
                        f"BR/EDR PFB | ch={bredr_channel} | freq={freq_mhz:.3f} MHz | "
                        f"ts_us={packet['timestamp_us']} | LAP={packet.get('lap')} | "
                        f"header={packet.get('packet_header_info')}"
                    )
    return ble_rows, bt_rows


def print_target_mapping(label, freqs_mhz, args, num_channels):
    mappings = describe_target_mapping(freqs_mhz, args.center_freq, args.sample_rate, num_channels)
    formatted = ", ".join(
        f"{target:.3f}->{coarse:.3f}({residual / 1e6:+.3f})"
        for target, coarse, residual in mappings
    )
    print(f"{label} target->coarse MHz(residual MHz): {formatted}")


def resolve_cuda_parse_config(args):
    if not args.use_cuda:
        return None, None
    require_cuda()
    cuda_status = describe_cuda_status()
    device_count = int(cuda_status["device_count"])
    device_id = 0 if args.cuda_device is None else args.cuda_device
    if device_id < 0 or device_id >= device_count:
        raise ValueError(
            f"--cuda-device {device_id} is invalid. Available CUDA device ids: "
            f"0..{device_count - 1}."
        )
    return device_id, cuda_status["devices"][device_id]


def validate_parse_args(args):
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.parse_workers < 1:
        raise ValueError("--parse-workers must be at least 1.")
    if args.parse_worker_chunk_segments < 1:
        raise ValueError("--parse-worker-chunk-segments must be at least 1.")
    if args.cpp_parser_threads < 1:
        raise ValueError("--cpp-parser-threads must be at least 1.")
    if args.cuda_segment_copy_merge_gap_samples < 0:
        raise ValueError("--cuda-segment-copy-merge-gap-samples must be non-negative.")
    if args.cuda_segment_copy_max_merged_samples < 1:
        raise ValueError("--cuda-segment-copy-max-merged-samples must be at least 1.")
    if getattr(args, "target_profile_top_ble", 0) < 0:
        raise ValueError("--target-profile-top-ble must be non-negative.")
    if getattr(args, "target_profile_top_bredr", 0) < 0:
        raise ValueError("--target-profile-top-bredr must be non-negative.")
    if getattr(args, "target_selection", "full") == "profile_topk" and not getattr(
        args, "target_profile_csv", ""
    ):
        raise ValueError("--target-profile-csv is required with --target-selection profile_topk.")
    if args.parse_workers > 1 and args.workers != 1:
        raise ValueError("--parse-workers > 1 currently requires --workers 1.")
    ble_backend = "python" if args.skip_ble else args.ble_parser_backend
    bredr_backend = "python" if args.skip_bredr else args.bredr_parser_backend
    validate_native_parser_backend(ble_backend, bredr_backend)


def initialize_parser_state(args, persistent_uap_cache=False):
    args.known_ble_aas, args.known_bredr_laps = resolve_known_packet_ids(args)
    lap_sync_words = known_lap_sync_words(args.known_bredr_laps)
    args.known_bredr_lap_sync_words = (
        tuple(lap_sync_words.keys()) if lap_sync_words else tuple()
    )
    args.learned_ble_aas = set(args.known_ble_aas)
    args.learned_ble_modes = {}
    args.learned_bredr_laps = set(args.known_bredr_laps)
    learned_lap_sync_words = known_lap_sync_words(args.learned_bredr_laps)
    args.learned_bredr_lap_sync_words = learned_lap_sync_words or {}
    if persistent_uap_cache:
        args._uap_cache = LockedUAPCache()


def build_pfb_runtime_config(args, metadata=None, realtime=False):
    validate_parse_args(args)
    initialize_parser_state(args, persistent_uap_cache=realtime)
    cuda_device_id, cuda_device = resolve_cuda_parse_config(args)
    if args.use_cuda:
        args.cuda_device = cuda_device_id
        if args.workers != 1:
            raise ValueError(
                "--use-cuda currently requires --workers 1 because GPU arrays stay on "
                "device through subband cleanup."
            )
        if args.cuda_fuse_target_dsp and not args.cuda_threshold_detect:
            raise ValueError("--cuda-fuse-target-dsp requires CUDA threshold detect.")

    metadata = legacy.load_metadata(args.metadata) if metadata is None else metadata
    decim, effective_bandwidth = legacy.resolve_parse_capture_params(args, metadata)
    timestamp_mode, rx_tags_csv, rx_time_tags = legacy.resolve_timestamp_config(args, metadata)
    legacy.validate_rx_tag_sample_rate(args, timestamp_mode, rx_time_tags)
    rx_time_tags = build_rx_time_tag_lookup(rx_time_tags)
    ble_freqs = legacy.parse_freq_list(args.ble_freqs_mhz) or ble_target_freqs_mhz(
        args.center_freq, effective_bandwidth
    )
    bredr_freqs = legacy.parse_freq_list(args.bredr_freqs_mhz) or bredr_target_freqs_mhz(
        args.center_freq, effective_bandwidth
    )
    original_ble_freqs = list(ble_freqs)
    original_bredr_freqs = list(bredr_freqs)
    ble_freqs, bredr_freqs, target_selection_rows = apply_target_selection(
        args,
        ble_freqs,
        bredr_freqs,
    )
    args.target_selection_rows = target_selection_rows
    args.target_selection_summary = format_target_selection_summary(
        args,
        original_ble_freqs,
        original_bredr_freqs,
        ble_freqs,
        bredr_freqs,
    )
    if getattr(args, "output_dir", ""):
        write_target_selection_csv(Path(args.output_dir) / "target_selection.csv", target_selection_rows)
    prototype, pfb_decim, num_channels = design_oversampled_pfb(
        args.sample_rate,
        args.subband_sample_rate,
        args.pfb_numtaps,
        args.pfb_cutoff,
    )
    if pfb_decim != decim:
        raise ValueError("PFB decimation does not match the pipeline subband decimation.")
    ble_cleanup_lpf = design_cleanup_lpf(
        args.subband_sample_rate, args.ble_lpf_cutoff, args.cleanup_numtaps
    )
    bredr_cleanup_lpf = design_cleanup_lpf(
        args.subband_sample_rate, args.bredr_lpf_cutoff, args.cleanup_numtaps
    )
    pfb_filter_delay = (args.pfb_numtaps - 1) / 2.0
    cleanup_filter_delay = (args.cleanup_numtaps - 1) / 2.0 * decim
    filter_delay = pfb_filter_delay + cleanup_filter_delay
    cuda_iq_overlap_cache = (
        {}
        if args.use_cuda and args.iq_format == "int16" and args.overlap_samples > 0
        else None
    )
    return PfbRuntimeConfig(
        args=args,
        metadata=metadata,
        decim=decim,
        effective_bandwidth=effective_bandwidth,
        prototype=prototype,
        num_channels=num_channels,
        ble_cleanup_lpf=ble_cleanup_lpf,
        bredr_cleanup_lpf=bredr_cleanup_lpf,
        ble_freqs=ble_freqs,
        bredr_freqs=bredr_freqs,
        filter_delay=filter_delay,
        timestamp_mode=timestamp_mode,
        rx_tags_csv=rx_tags_csv,
        rx_time_tags=rx_time_tags,
        cuda_device_id=cuda_device_id,
        cuda_device=cuda_device,
        cuda_iq_overlap_cache=cuda_iq_overlap_cache,
        realtime=realtime,
    )


def print_pfb_runtime_summary(runtime):
    args = runtime.args
    pfb_filter_delay = (args.pfb_numtaps - 1) / 2.0
    cleanup_filter_delay = (args.cleanup_numtaps - 1) / 2.0 * runtime.decim
    print(
        f"PFB wideband mode: sample_rate={args.sample_rate:g} Hz, "
        f"center_freq={args.center_freq / 1e6:.3f} MHz, bandwidth={runtime.effective_bandwidth:g} Hz."
    )
    print(
        f"Channelizer: 2x oversampled FFT/PFB with {runtime.num_channels} coarse bins, "
        f"{args.sample_rate / runtime.num_channels / 1e6:.3f} MHz spacing, "
        f"{args.subband_sample_rate / 1e6:.3f} MHz output rate, {args.pfb_numtaps} taps."
    )
    print(
        f"Cleanup FIR: {args.cleanup_numtaps} taps at {args.subband_sample_rate / 1e6:.3f} MHz, "
        f"BLE cutoff={args.ble_lpf_cutoff / 1e6:.3f} MHz, "
        f"BR/EDR cutoff={args.bredr_lpf_cutoff / 1e6:.3f} MHz."
    )
    print(
        f"FIR delay compensation: PFB={pfb_filter_delay:g} wideband samples, "
        f"cleanup={cleanup_filter_delay:g}, total={runtime.filter_delay:g}."
    )
    print(
        f"Coarse-subband execution: workers={args.workers}, "
        f"mode={'serial' if args.workers == 1 else 'process_pool'}, "
        f"parse_workers={args.parse_workers}, "
        f"parse_chunk_segments={args.parse_worker_chunk_segments}, "
        f"cpp_parser_threads={args.cpp_parser_threads}, "
        f"ble_parser_backend={args.ble_parser_backend}, "
        f"bredr_parser_backend={args.bredr_parser_backend}, "
            f"native_segment_input={args.native_segment_input}, "
        f"parallel_bredr_candidates={'enabled' if args.parallel_bredr_candidates else 'disabled'}, "
        f"native_br_batch_candidates={'enabled' if args.native_br_batch_candidates else 'disabled'}."
    )
    status = native_status()
    if status["available"]:
        print(f"C++ parser extension: available, bt_native={status['version']}.")
    else:
        print("C++ parser extension: unavailable, Python parser backend remains active.")
    print(
        f"Known packet tables: BLE_AA={len(args.known_ble_aas)}, "
        f"BR_EDR_LAP={len(args.known_bredr_laps)}, "
        f"bredr_lap_fast_path={'enabled' if args.known_bredr_lap_fast_path else 'disabled'}, "
        f"learned_parser_fast_path={'enabled' if args.learned_parser_fast_path else 'disabled'}."
    )
    if hasattr(args, "target_selection_summary"):
        print(args.target_selection_summary)
    if args.use_cuda:
        print(
            f"CUDA DSP frontend: enabled, device={runtime.cuda_device['id']} "
            f"{runtime.cuda_device['name']}; PFB/shift/cleanup on GPU, target subbands return "
            f"to CPU before protocol parsers; "
            f"batch_targets={'enabled' if args.cuda_batch_targets else 'disabled'}, "
            f"pfb_backend={args.cuda_pfb_backend}, "
            f"cleanup_backend={args.cuda_cleanup_backend}, "
            f"fuse_target_dsp={'enabled' if args.cuda_fuse_target_dsp else 'disabled'}, "
            f"threshold_detect={'enabled' if args.cuda_threshold_detect else 'disabled'}, "
            f"target_dsp_materialization={args.cuda_target_dsp_materialization}, "
            f"known_candidate_filter={'enabled' if args.cuda_known_candidate_filter else 'disabled'}, "
            f"segment_copy_merge_gap={args.cuda_segment_copy_merge_gap_samples} samples, "
            f"segment_copy_max_merged={args.cuda_segment_copy_max_merged_samples} samples, "
            f"target_dsp_profile={'enabled' if getattr(args, 'target_dsp_profile', False) else 'disabled'}."
        )
    else:
        print("CUDA DSP frontend: disabled, using CPU.")
    print_target_mapping("BLE", runtime.ble_freqs, args, runtime.num_channels)
    print_target_mapping("BR/EDR", runtime.bredr_freqs, args, runtime.num_channels)
    print(f"Timestamp mode: {runtime.timestamp_mode}")
    if runtime.timestamp_mode == "uhd_rx_time":
        print(f"Using UHD rx_time tags: {runtime.rx_tags_csv}")
    else:
        print("Timestamp uses FIR-delay-compensated wideband_sample_index / sample_rate.")


def channelize_pfb_chunk(chunk, chunk_start, runtime):
    args = runtime.args
    chunk_timing = {}
    stage_start = perf_counter()
    channel_bank = channelize_oversampled_pfb(
        chunk,
        chunk_start,
        args.sample_rate,
        runtime.decim,
        runtime.prototype,
        use_cuda=args.use_cuda,
        device_id=runtime.cuda_device_id,
        return_host=False,
        cuda_pfb_backend=args.cuda_pfb_backend,
        timing_context=chunk_timing if args.timing else None,
        cuda_iq_overlap_cache=runtime.cuda_iq_overlap_cache,
        cuda_iq_overlap_cache_samples=2 * args.overlap_samples,
    )
    if getattr(args, "cuda_sync_timing", False) and args.use_cuda:
        stage_start_sync = perf_counter()
        with use_device(runtime.cuda_device_id):
            synchronize()
        add_timing(chunk_timing, "gpu_sync_wait_s", perf_counter() - stage_start_sync)
    add_timing(chunk_timing, "pfb_s", perf_counter() - stage_start)
    return channel_bank, chunk_timing


def build_pfb_chunk_tasks(channel_bank, chunk_start, runtime):
    args = runtime.args
    chunk_timing = {}
    stage_start = perf_counter()
    tasks = group_coarse_subband_tasks(
        channel_bank,
        args,
        chunk_start,
        runtime.decim,
        runtime.ble_cleanup_lpf,
        runtime.bredr_cleanup_lpf,
        runtime.ble_freqs,
        runtime.bredr_freqs,
    )
    add_timing(chunk_timing, "group_tasks_s", perf_counter() - stage_start)
    print(f"Coarse-subband tasks: {len(tasks)}")
    return tasks, chunk_timing


def run_pfb_chunk_tasks(tasks, executor=None):
    chunk_timing = {}
    stage_start = perf_counter()
    if executor is None:
        task_results = [parse_coarse_subband_task(task) for task in tasks]
    else:
        task_results = list(executor.map(parse_coarse_subband_task, tasks))
    add_timing(chunk_timing, "task_wall_s", perf_counter() - stage_start)
    for _bank_index, _results, task_timing in task_results:
        for key, seconds in task_timing.items():
            add_timing(chunk_timing, key, seconds)
    return task_results, chunk_timing


def run_pfb_target_dsp_tasks(tasks, executor=None):
    chunk_timing = {}
    stage_start = perf_counter()
    if executor is None:
        target_dsp_results = [build_coarse_subband_parse_inputs(task) for task in tasks]
    else:
        target_dsp_results = list(executor.map(build_coarse_subband_parse_inputs, tasks))
    add_timing(chunk_timing, "target_dsp_wall_s", perf_counter() - stage_start)
    for _bank_index, _parse_inputs, task_timing in target_dsp_results:
        for key, seconds in task_timing.items():
            add_timing(chunk_timing, key, seconds)
    return target_dsp_results, chunk_timing


def run_pfb_parser_tasks(target_dsp_results):
    chunk_timing = {}
    stage_start = perf_counter()
    task_results = [parse_coarse_subband_inputs(item) for item in target_dsp_results]
    add_timing(chunk_timing, "parser_wall_s", perf_counter() - stage_start)
    for _bank_index, _results, task_timing in task_results:
        for key, seconds in task_timing.items():
            add_timing(chunk_timing, key, seconds)
    return task_results, chunk_timing


def finalize_pfb_chunk_results(task_results, chunk_start, core_start, core_end, runtime):
    args = runtime.args
    chunk_timing = {}
    stage_start = perf_counter()
    rx_time_tags = runtime.rx_time_tags
    if hasattr(rx_time_tags, "slice_for_range"):
        rx_time_tags = rx_time_tags.slice_for_range(core_start, core_end)
    chunk_ble_rows, chunk_bt_rows = finalize_task_results(
        task_results,
        args,
        chunk_start,
        core_start,
        core_end,
        runtime.filter_delay,
        runtime.decim,
        runtime.timestamp_mode,
        rx_time_tags,
    )
    add_timing(chunk_timing, "finalize_s", perf_counter() - stage_start)
    return chunk_ble_rows, chunk_bt_rows, chunk_timing


def process_pfb_chunk(chunk, chunk_start, core_start, core_end, runtime, executor=None):
    chunk_timing = {}
    channel_bank, stage_timing = channelize_pfb_chunk(chunk, chunk_start, runtime)
    for key, seconds in stage_timing.items():
        add_timing(chunk_timing, key, seconds)

    tasks, stage_timing = build_pfb_chunk_tasks(channel_bank, chunk_start, runtime)
    for key, seconds in stage_timing.items():
        add_timing(chunk_timing, key, seconds)

    task_results, stage_timing = run_pfb_chunk_tasks(tasks, executor=executor)
    for key, seconds in stage_timing.items():
        add_timing(chunk_timing, key, seconds)

    chunk_ble_rows, chunk_bt_rows, stage_timing = finalize_pfb_chunk_results(
        task_results,
        chunk_start,
        core_start,
        core_end,
        runtime,
    )
    for key, seconds in stage_timing.items():
        add_timing(chunk_timing, key, seconds)
    return chunk_ble_rows, chunk_bt_rows, chunk_timing


def run_parse(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_pfb_runtime_config(args, realtime=False)
    print_pfb_runtime_summary(runtime)

    ble_rows = []
    bt_rows = []
    timing_totals = {}
    chunk_count = 0
    parse_start = perf_counter()
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    parse_executor = (
        ProcessPoolExecutor(max_workers=args.parse_workers)
        if args.parse_workers > 1
        else None
    )
    args._parse_executor = parse_executor
    try:
        chunk_iter = iter_iq_chunks(
            args.input_bin,
            args.iq_format,
            args.chunk_samples,
            args.overlap_samples,
            decode_int16=not (args.use_cuda and args.iq_format == "int16"),
        )
        while True:
            chunk_timing = {}
            stage_start = perf_counter()
            try:
                chunk_start, core_start, core_end, chunk = next(chunk_iter)
            except StopIteration:
                break
            add_timing(chunk_timing, "read_chunk_s", perf_counter() - stage_start)
            chunk_count += 1
            chunk_samples = chunk.size // 2 if chunk.dtype == np.int16 else chunk.size
            print(
                f"Processing PFB chunk: read_start={chunk_start}, "
                f"core=[{core_start}, {core_end}), samples={chunk_samples}"
            )
            chunk_ble_rows, chunk_bt_rows, process_timing = process_pfb_chunk(
                chunk,
                chunk_start,
                core_start,
                core_end,
                runtime,
                executor=executor,
            )
            for key, seconds in process_timing.items():
                add_timing(chunk_timing, key, seconds)
            ble_rows.extend(chunk_ble_rows)
            bt_rows.extend(chunk_bt_rows)
            for key, seconds in chunk_timing.items():
                add_timing(timing_totals, key, seconds)
            if args.timing:
                print(
                    "Timing chunk: "
                    f"pfb={format_seconds(chunk_timing.get('pfb_s', 0.0))}, "
                    f"shift={format_seconds(chunk_timing.get('shift_s', 0.0))}, "
                    f"cleanup={format_seconds(chunk_timing.get('cleanup_s', 0.0))}, "
                    f"threshold={format_seconds(chunk_timing.get('threshold_s', 0.0))}, "
                    f"read_chunk={format_seconds(chunk_timing.get('read_chunk_s', 0.0))}, "
                    f"host_to_gpu={format_seconds(chunk_timing.get('host_to_gpu_s', 0.0))}, "
                    f"gpu_sync_wait={format_seconds(chunk_timing.get('gpu_sync_wait_s', 0.0))}, "
                    f"segment_copy_back={format_seconds(chunk_timing.get('segment_copy_back_s', 0.0))}, "
                    f"segment_input_build={format_seconds(chunk_timing.get('segment_input_build_s', 0.0))}, "
                    f"threshold_mask={format_seconds(chunk_timing.get('threshold_mask_s', 0.0))}, "
                    f"segment_materialize={format_seconds(chunk_timing.get('segment_materialize_s', 0.0))}, "
                    f"segment_materialize_calls={int(chunk_timing.get('segment_materialize_calls', 0))}, "
                    f"segment_materialize_samples={int(chunk_timing.get('segment_materialize_samples', 0))}, "
                    f"segment_count={int(chunk_timing.get('segment_count', 0))}, "
                    f"segment_copy_calls={int(chunk_timing.get('segment_copy_calls', 0))}, "
                    f"segment_copy_samples={int(chunk_timing.get('segment_copy_samples', 0))}, "
                    f"parser_submit={format_seconds(chunk_timing.get('parser_submit_s', 0.0))}, "
                    f"overlap_reuse={int(chunk_timing.get('overlap_reuse_samples', 0))}, "
                    f"fused_target={format_seconds(chunk_timing.get('fused_target_s', 0.0))}, "
                    f"fused_target_reuse={int(chunk_timing.get('fused_target_reuse', 0))}, "
                    f"fused_target_batches={int(chunk_timing.get('fused_target_batches', 0))}, "
                    f"fused_target_batch_items={int(chunk_timing.get('fused_target_batch_items', 0))}, "
                    f"fused_target_shared_input_batches={int(chunk_timing.get('fused_target_shared_input_batches', 0))}, "
                    f"ble_parse={format_seconds(chunk_timing.get('ble_parse_s', 0.0))}, "
                    f"bredr_parse={format_seconds(chunk_timing.get('bredr_parse_s', 0.0))}, "
                    f"finalize={format_seconds(chunk_timing.get('finalize_s', 0.0))}, "
                    f"task_wall={format_seconds(chunk_timing.get('task_wall_s', 0.0))}, "
                    f"targets={int(chunk_timing.get('targets', 0))}"
                )
    finally:
        if executor is not None:
            executor.shutdown()
        if parse_executor is not None:
            parse_executor.shutdown()
        args._parse_executor = None

    stage_start = perf_counter()
    ble_rows.sort(key=lambda row: int(row["sample_index"]))
    bt_rows.sort(key=lambda row: int(row["sample_index"]))
    add_timing(timing_totals, "sort_s", perf_counter() - stage_start)
    stage_start = perf_counter()
    write_csv(output_dir / "ble_packets.csv", ble_rows, legacy.BLE_40M_FIELDS)
    write_csv(output_dir / "btclassic_packets.csv", bt_rows, legacy.BT_40M_FIELDS)
    save_packet_events(ble_rows, bt_rows, output_dir / "packet_events.csv")
    add_timing(timing_totals, "write_csv_s", perf_counter() - stage_start)
    add_timing(timing_totals, "total_wall_s", perf_counter() - parse_start)
    print(f"Saved BLE rows: {len(ble_rows)} -> {output_dir / 'ble_packets.csv'}")
    print(f"Saved BR/EDR rows: {len(bt_rows)} -> {output_dir / 'btclassic_packets.csv'}")
    print(f"Saved packet events -> {output_dir / 'packet_events.csv'}")
    if args.timing:
        print(
            "Timing summary: "
            f"chunks={chunk_count}, "
            f"pfb={format_seconds(timing_totals.get('pfb_s', 0.0))}, "
            f"group_tasks={format_seconds(timing_totals.get('group_tasks_s', 0.0))}, "
            f"shift={format_seconds(timing_totals.get('shift_s', 0.0))}, "
            f"cleanup={format_seconds(timing_totals.get('cleanup_s', 0.0))}, "
            f"threshold={format_seconds(timing_totals.get('threshold_s', 0.0))}, "
            f"read_chunk={format_seconds(timing_totals.get('read_chunk_s', 0.0))}, "
            f"host_to_gpu={format_seconds(timing_totals.get('host_to_gpu_s', 0.0))}, "
            f"gpu_sync_wait={format_seconds(timing_totals.get('gpu_sync_wait_s', 0.0))}, "
            f"segment_copy_back={format_seconds(timing_totals.get('segment_copy_back_s', 0.0))}, "
            f"segment_input_build={format_seconds(timing_totals.get('segment_input_build_s', 0.0))}, "
            f"threshold_mask={format_seconds(timing_totals.get('threshold_mask_s', 0.0))}, "
            f"segment_materialize={format_seconds(timing_totals.get('segment_materialize_s', 0.0))}, "
            f"segment_materialize_calls={int(timing_totals.get('segment_materialize_calls', 0))}, "
            f"segment_materialize_samples={int(timing_totals.get('segment_materialize_samples', 0))}, "
            f"segment_count={int(timing_totals.get('segment_count', 0))}, "
            f"segment_copy_calls={int(timing_totals.get('segment_copy_calls', 0))}, "
            f"segment_copy_samples={int(timing_totals.get('segment_copy_samples', 0))}, "
            f"parser_submit={format_seconds(timing_totals.get('parser_submit_s', 0.0))}, "
            f"overlap_reuse={int(timing_totals.get('overlap_reuse_samples', 0))}, "
            f"fused_target={format_seconds(timing_totals.get('fused_target_s', 0.0))}, "
            f"fused_target_reuse={int(timing_totals.get('fused_target_reuse', 0))}, "
            f"fused_target_batches={int(timing_totals.get('fused_target_batches', 0))}, "
            f"fused_target_batch_items={int(timing_totals.get('fused_target_batch_items', 0))}, "
            f"fused_target_shared_input_batches={int(timing_totals.get('fused_target_shared_input_batches', 0))}, "
            f"ble_parse={format_seconds(timing_totals.get('ble_parse_s', 0.0))}, "
            f"bredr_parse={format_seconds(timing_totals.get('bredr_parse_s', 0.0))}, "
            f"task_wall={format_seconds(timing_totals.get('task_wall_s', 0.0))}, "
            f"finalize={format_seconds(timing_totals.get('finalize_s', 0.0))}, "
            f"sort={format_seconds(timing_totals.get('sort_s', 0.0))}, "
            f"write_csv={format_seconds(timing_totals.get('write_csv_s', 0.0))}, "
            f"total_wall={format_seconds(timing_totals.get('total_wall_s', 0.0))}, "
            f"targets={int(timing_totals.get('targets', 0))}"
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "capture":
        legacy.run_capture(args)
    elif args.command == "parse":
        run_parse(args)


if __name__ == "__main__":
    main()
