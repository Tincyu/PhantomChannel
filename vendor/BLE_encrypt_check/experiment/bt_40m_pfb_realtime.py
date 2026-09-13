import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, sleep

import bt_40m_pfb_pipeline as pfb
import bt_40m_pipeline as legacy
from bt_pipeline.parsers import LockedUAPCache
from bt_pipeline.realtime_pipeline import (
    AsyncChunkWorker,
    AsyncFullyStagedChunkWorker,
    AsyncSinkWriter,
    AsyncStagedChunkWorker,
    AsyncTargetStagedChunkWorker,
    PipelineStats,
    SynchronousChunkWorker,
)
from bt_pipeline.realtime_sinks import CsvAppendSink, JsonlEventSink, MultiSink
from bt_pipeline.realtime_sources import (
    FileReplayIQSource,
    FramedUdpIQSource,
    RawPipeIQSource,
    RawPipeWithUdpTagsIQSource,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Realtime-style Bluetooth PFB parser. The first implementation uses "
            "file replay as the source and writes packet rows incrementally."
        )
    )
    parser.add_argument("--source", choices=("file", "stdin", "stdin-udp-tags", "udp-framed"), default="file")
    parser.add_argument("--input-bin", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--rx-tags-csv", default="")
    parser.add_argument("--udp-ip", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=9002)
    parser.add_argument("--udp-timeout", type=float, default=0.2)
    parser.add_argument("--udp-socket-rcvbuf", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--udp-max-packet-size", type=int, default=65535)
    parser.add_argument("--udp-payload-format", choices=("sc16",), default="sc16")
    parser.add_argument(
        "--udp-tag-wait",
        type=float,
        default=0.2,
        help="Seconds to wait for UDP rx_time tags to cover a raw stdin chunk.",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=["sample_index", "uhd_rx_time", "auto"],
        default="auto",
    )
    parser.add_argument("--sample-rate", type=float, default=legacy.WIDEBAND_SAMPLE_RATE)
    parser.add_argument("--center-freq", type=float, default=2420e6)
    parser.add_argument("--bandwidth", type=float, default=legacy.DEFAULT_BANDWIDTH)
    parser.add_argument("--subband-sample-rate", type=float, default=legacy.SUBBAND_SAMPLE_RATE)
    parser.add_argument("--iq-format", choices=["complex64", "int16"], default="int16")
    parser.add_argument("--output-dir", default="artifacts/40m_pfb/realtime")
    parser.add_argument("--jsonl-events", default="")
    parser.add_argument("--chunk-samples", type=int, default=16_000_000)
    parser.add_argument("--overlap-samples", type=int, default=200_000)
    parser.add_argument("--replay-rate", type=float, default=0.0)
    parser.add_argument(
        "--pinned-host-buffers",
        action="store_true",
        help="Use CUDA pinned host memory for stdin raw IQ reads. File replay keeps using mmap.",
    )
    parser.add_argument(
        "--source-prefetch",
        action="store_true",
        help="Read the next source chunk in a background thread while processing the current chunk.",
    )
    parser.add_argument(
        "--ring-slots",
        type=int,
        default=2,
        help="Configured source ring slots for future worker pipeline. Current prefetch uses one outstanding read.",
    )
    parser.add_argument(
        "--async-sink",
        action="store_true",
        help="Write CSV/JSONL rows from a background output writer thread.",
    )
    parser.add_argument(
        "--async-chunk-worker",
        action="store_true",
        help="Run chunk processing from a background worker thread while preserving output order.",
    )
    parser.add_argument(
        "--split-stage-workers",
        action="store_true",
        help=(
            "Split async chunk processing into channelize and parser/finalize worker "
            "threads. Requires --async-chunk-worker."
        ),
    )
    parser.add_argument(
        "--split-target-dsp-worker",
        action="store_true",
        help=(
            "Further split target DSP from CPU parser/finalize workers. Requires "
            "--split-stage-workers."
        ),
    )
    parser.add_argument(
        "--split-finalize-worker",
        action="store_true",
        help=(
            "Further split CPU parser and finalize into separate worker threads. "
            "Requires --split-target-dsp-worker."
        ),
    )
    parser.add_argument(
        "--max-queue-chunks",
        type=int,
        default=4,
        help=(
            "Default maximum queued chunks for async workers. Stage-specific queue "
            "options override this value."
        ),
    )
    parser.add_argument(
        "--worker-queue-chunks",
        type=int,
        default=0,
        help="Maximum queued chunks inside async chunk/stage workers. Default uses --max-queue-chunks.",
    )
    parser.add_argument(
        "--sink-queue-chunks",
        type=int,
        default=0,
        help="Maximum queued chunks for async sink/output worker. Default uses --max-queue-chunks.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after N chunks. Default 0 processes until the source is exhausted.",
    )
    parser.add_argument(
        "--reset-parser-state-per-chunk",
        action="store_true",
        help=(
            "Testing mode: reset parser state each chunk to mirror the offline parse "
            "baseline. Default keeps realtime parser state alive across chunks."
        ),
    )

    parser.add_argument("--freq-dev", type=float, default=250e3)
    parser.add_argument("--ble-threshold", type=float, default=0.01)
    parser.add_argument(
        "--ble-candidate-detector",
        choices=("fixed", "adaptive_hysteresis"),
        default="fixed",
        help="BLE candidate gate. The default preserves the legacy fixed-amplitude behavior.",
    )
    parser.add_argument("--ble-envelope-window-samples", type=int, default=8)
    parser.add_argument("--ble-start-noise-multiplier", type=float, default=3.5)
    parser.add_argument("--ble-hold-noise-multiplier", type=float, default=1.8)
    parser.add_argument("--ble-candidate-gap-tolerance-samples", type=int, default=0)
    parser.add_argument("--ble-candidate-prepad-samples", type=int, default=0)
    parser.add_argument("--ble-candidate-postpad-samples", type=int, default=0)
    parser.add_argument("--br-threshold", type=float, default=0.005)
    parser.add_argument("--ble-segment-min-len", type=int, default=150)
    parser.add_argument("--br-segment-min-len", type=int, default=200)
    parser.add_argument("--ble-score-threshold", type=float, default=3.0)
    parser.add_argument("--br-cutoff", type=float, default=0.5e6)
    parser.add_argument("--ble-lpf-cutoff", type=float, default=1.0e6)
    parser.add_argument("--bredr-lpf-cutoff", type=float, default=0.75e6)
    parser.add_argument("--pfb-cutoff", type=float, default=1.9e6)
    parser.add_argument("--pfb-numtaps", type=int, default=401)
    parser.add_argument("--cleanup-numtaps", type=int, default=31)
    parser.add_argument("--ble-freqs-mhz", default="")
    parser.add_argument("--bredr-freqs-mhz", default="")
    parser.add_argument("--skip-ble", action="store_true")
    parser.add_argument("--skip-bredr", action="store_true")

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parse-workers", type=int, default=1)
    parser.add_argument("--parse-worker-chunk-segments", type=int, default=32)
    parser.add_argument("--cpp-parser-threads", type=int, default=4)
    parser.add_argument("--ble-parser-backend", choices=("python", "cpp"), default="cpp")
    parser.add_argument(
        "--bredr-parser-backend",
        choices=("python", "cpp", "hybrid"),
        default="hybrid",
        help=(
            "BR/EDR protocol parser backend. cpp uses the native finalized segment "
            "path; hybrid uses the native path when possible and falls back for "
            "unsupported known-LAP modes."
        ),
    )
    parser.add_argument("--parallel-bredr-candidates", action="store_true")
    parser.add_argument("--native-br-batch-candidates", action="store_true")
    parser.add_argument("--known-ble-aa", default="")
    parser.add_argument("--known-bredr-lap", default="")
    parser.add_argument("--known-bredr-lap-fast-path", action="store_true")
    parser.add_argument("--known-packet-csv", action="append", default=[])
    parser.add_argument(
        "--learned-parser-fast-path",
        dest="learned_parser_fast_path",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-learned-parser-fast-path",
        dest="learned_parser_fast_path",
        action="store_false",
    )

    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true", default=True)
    parser.add_argument("--no-use-cuda", dest="use_cuda", action="store_false")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
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
        default="kernel_multi_float_phase",
    )
    parser.add_argument(
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
    )
    parser.add_argument("--cuda-batch-targets", dest="cuda_batch_targets", action="store_true", default=True)
    parser.add_argument("--no-cuda-batch-targets", dest="cuda_batch_targets", action="store_false")
    parser.add_argument("--cuda-fuse-target-dsp", dest="cuda_fuse_target_dsp", action="store_true", default=True)
    parser.add_argument("--no-cuda-fuse-target-dsp", dest="cuda_fuse_target_dsp", action="store_false")
    parser.add_argument("--cuda-threshold-detect", dest="cuda_threshold_detect", action="store_true", default=True)
    parser.add_argument("--no-cuda-threshold-detect", dest="cuda_threshold_detect", action="store_false")
    parser.add_argument(
        "--cuda-segment-copy-merge-gap-samples",
        type=int,
        default=4096,
        help=(
            "Maximum gap between CUDA threshold segments copied back in one host "
            "transfer. Default 4096 preserves the previous hard-coded behavior."
        ),
    )
    parser.add_argument(
        "--cuda-segment-copy-max-merged-samples",
        type=int,
        default=262144,
        help=(
            "Maximum span copied back in one merged CUDA threshold transfer. "
            "Default 262144 preserves the previous internal limit."
        ),
    )
    parser.add_argument(
        "--cuda-target-dsp-materialization",
        choices=("full", "threshold_then_segments"),
        default="full",
        help=(
            "CUDA fused target DSP materialization mode. full preserves the current "
            "cleaned-IQ matrix path; threshold_then_segments writes only threshold "
            "masks first and materializes IQ only for merged parser segments."
        ),
    )
    parser.add_argument(
        "--cuda-known-candidate-filter",
        dest="cuda_known_candidate_filter",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-cuda-known-candidate-filter",
        dest="cuda_known_candidate_filter",
        action="store_false",
    )
    parser.add_argument("--quiet", dest="quiet", action="store_true", default=True)
    parser.add_argument("--verbose", dest="quiet", action="store_false")
    parser.add_argument("--timing", dest="timing", action="store_true", default=True)
    parser.add_argument("--no-timing", dest="timing", action="store_false")
    parser.add_argument(
        "--cuda-sync-timing",
        action="store_true",
        default=False,
        help=(
            "Synchronize CUDA after timed GPU stages and report gpu_sync_wait_s. "
            "Disabled by default for low-overhead realtime runs."
        ),
    )
    parser.add_argument(
        "--native-segment-input",
        choices=("legacy", "compact"),
        default="legacy",
        help=(
            "Native parser segment input mode. compact keeps CUDA threshold output "
            "as merged host buffers plus descriptors for native parser backends."
        ),
    )
    parser.add_argument(
        "--target-dsp-profile",
        action="store_true",
        default=False,
        help=(
            "Collect per protocol/frequency/residual target DSP copy-back profile "
            "metrics and write target_dsp_profile.csv."
        ),
    )
    parser.add_argument(
        "--target-selection",
        choices=("full", "profile_topk"),
        default="full",
        help=(
            "Target frequency selection mode. full preserves the complete target "
            "list; profile_topk ranks targets from --target-profile-csv and keeps "
            "the requested top BLE/BR/EDR frequencies."
        ),
    )
    parser.add_argument(
        "--target-profile-csv",
        default="",
        help="target_dsp_profile.csv used by --target-selection profile_topk.",
    )
    parser.add_argument(
        "--target-profile-metric",
        choices=pfb.TARGET_SELECTION_METRICS,
        default="segment_copy_samples",
        help="Profile metric used to rank targets for profile_topk selection.",
    )
    parser.add_argument(
        "--target-profile-top-ble",
        type=int,
        default=0,
        help="Keep top N BLE targets in profile_topk mode. Default 0 keeps all BLE targets.",
    )
    parser.add_argument(
        "--target-profile-top-bredr",
        type=int,
        default=0,
        help="Keep top N BR/EDR targets in profile_topk mode. Default 0 keeps all BR/EDR targets.",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=0.0,
        help="Print realtime throughput stats every N seconds. Default 0 disables periodic stats.",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=0.0,
        help="Warn when one chunk takes longer than this many milliseconds. Default 0 disables warnings.",
    )
    return parser


def print_chunk_timing(chunk_timing):
    print(
        "Timing chunk: "
        f"pfb={pfb.format_seconds(chunk_timing.get('pfb_s', 0.0))}, "
        f"shift={pfb.format_seconds(chunk_timing.get('shift_s', 0.0))}, "
        f"cleanup={pfb.format_seconds(chunk_timing.get('cleanup_s', 0.0))}, "
        f"threshold={pfb.format_seconds(chunk_timing.get('threshold_s', 0.0))}, "
        f"read_chunk={pfb.format_seconds(chunk_timing.get('read_chunk_s', 0.0))}, "
        f"gpu_sync_wait={pfb.format_seconds(chunk_timing.get('gpu_sync_wait_s', 0.0))}, "
        f"segment_copy_back={pfb.format_seconds(chunk_timing.get('segment_copy_back_s', 0.0))}, "
        f"segment_input_build={pfb.format_seconds(chunk_timing.get('segment_input_build_s', 0.0))}, "
        f"threshold_mask={pfb.format_seconds(chunk_timing.get('threshold_mask_s', 0.0))}, "
        f"segment_materialize={pfb.format_seconds(chunk_timing.get('segment_materialize_s', 0.0))}, "
        f"segment_materialize_calls={int(chunk_timing.get('segment_materialize_calls', 0))}, "
        f"segment_materialize_samples={int(chunk_timing.get('segment_materialize_samples', 0))}, "
        f"segment_count={int(chunk_timing.get('segment_count', 0))}, "
        f"segment_copy_calls={int(chunk_timing.get('segment_copy_calls', 0))}, "
        f"segment_copy_samples={int(chunk_timing.get('segment_copy_samples', 0))}, "
        f"ble_parse={pfb.format_seconds(chunk_timing.get('ble_parse_s', 0.0))}, "
        f"bredr_parse={pfb.format_seconds(chunk_timing.get('bredr_parse_s', 0.0))}, "
        f"target_dsp_wall={pfb.format_seconds(chunk_timing.get('target_dsp_wall_s', 0.0))}, "
        f"parser_wall={pfb.format_seconds(chunk_timing.get('parser_wall_s', 0.0))}, "
        f"task_wall={pfb.format_seconds(chunk_timing.get('task_wall_s', 0.0))}, "
        f"finalize={pfb.format_seconds(chunk_timing.get('finalize_s', 0.0))}, "
        f"targets={int(chunk_timing.get('targets', 0))}"
    )


def run_realtime(args):
    output_dir = Path(args.output_dir)
    if args.source == "file" and not args.input_bin:
        raise ValueError("--source file requires --input-bin")
    if args.ring_slots < 1:
        raise ValueError("--ring-slots must be at least 1")
    if args.max_queue_chunks < 1:
        raise ValueError("--max-queue-chunks must be at least 1")
    if args.worker_queue_chunks < 0:
        raise ValueError("--worker-queue-chunks must be non-negative")
    if args.sink_queue_chunks < 0:
        raise ValueError("--sink-queue-chunks must be non-negative")
    if args.pinned_host_buffers and args.source not in ("stdin", "stdin-udp-tags"):
        raise ValueError("--pinned-host-buffers currently requires --source stdin or stdin-udp-tags")
    if args.pinned_host_buffers and args.source_prefetch and args.ring_slots < 2:
        raise ValueError(
            "--source-prefetch with --pinned-host-buffers requires --ring-slots >= 2."
        )
    if args.async_chunk_worker and args.reset_parser_state_per_chunk:
        raise ValueError(
            "--async-chunk-worker cannot be combined with "
            "--reset-parser-state-per-chunk."
        )
    if args.split_stage_workers and not args.async_chunk_worker:
        raise ValueError("--split-stage-workers requires --async-chunk-worker")
    if args.split_target_dsp_worker and not args.split_stage_workers:
        raise ValueError("--split-target-dsp-worker requires --split-stage-workers")
    if args.split_finalize_worker and not args.split_target_dsp_worker:
        raise ValueError("--split-finalize-worker requires --split-target-dsp-worker")
    worker_queue_chunks = args.worker_queue_chunks or args.max_queue_chunks
    sink_queue_chunks = args.sink_queue_chunks or args.max_queue_chunks
    if args.source in ("udp-framed", "stdin-udp-tags") and args.timestamp_mode == "uhd_rx_time" and not args.rx_tags_csv:
        args.timestamp_mode = "auto"
    runtime = pfb.build_pfb_runtime_config(args, realtime=True)

    decode_int16 = not (args.use_cuda and args.iq_format == "int16")
    if args.source == "stdin":
        source = RawPipeIQSource(
            iq_format=args.iq_format,
            chunk_samples=args.chunk_samples,
            overlap_samples=args.overlap_samples,
            decode_int16=decode_int16,
            pinned_host_buffer=args.pinned_host_buffers,
            pinned_slots=args.ring_slots,
        )
    elif args.source == "stdin-udp-tags":
        source = RawPipeWithUdpTagsIQSource(
            iq_format=args.iq_format,
            chunk_samples=args.chunk_samples,
            overlap_samples=args.overlap_samples,
            decode_int16=decode_int16,
            pinned_host_buffer=args.pinned_host_buffers,
            pinned_slots=args.ring_slots,
            udp_ip=args.udp_ip,
            udp_port=args.udp_port,
            udp_timeout_s=args.udp_timeout,
            udp_socket_rcvbuf=args.udp_socket_rcvbuf,
            udp_max_packet_size=args.udp_max_packet_size,
            tag_wait_s=args.udp_tag_wait,
        )
        runtime.timestamp_mode = "uhd_rx_time"
        runtime.rx_tags_csv = f"stdin+udp-tags://{args.udp_ip}:{args.udp_port}"
        runtime.rx_time_tags = source.rx_time_lookup()
    elif args.source == "udp-framed":
        source = FramedUdpIQSource(
            udp_ip=args.udp_ip,
            udp_port=args.udp_port,
            iq_format=args.iq_format,
            chunk_samples=args.chunk_samples,
            overlap_samples=args.overlap_samples,
            decode_int16=decode_int16,
            socket_rcvbuf=args.udp_socket_rcvbuf,
            max_packet_size=args.udp_max_packet_size,
            timeout_s=args.udp_timeout,
            expected_payload_format=args.udp_payload_format,
        )
        runtime.timestamp_mode = "uhd_rx_time"
        runtime.rx_tags_csv = f"udp://{args.udp_ip}:{args.udp_port}"
        runtime.rx_time_tags = source.rx_time_lookup()
    else:
        source = FileReplayIQSource(
            args.input_bin,
            args.iq_format,
            args.chunk_samples,
            args.overlap_samples,
            decode_int16=decode_int16,
        )
    pfb.print_pfb_runtime_summary(runtime)
    print(f"Realtime source: {args.source}")
    if args.source == "udp-framed":
        print(f"Realtime UDP framed input: {args.udp_ip}:{args.udp_port}")
    if args.source == "stdin-udp-tags":
        print(f"Realtime raw stdin IQ + UDP rx_time tags: {args.udp_ip}:{args.udp_port}")
    print(f"Realtime sink: incremental CSV -> {output_dir}")
    sinks = [CsvAppendSink(output_dir, legacy.BLE_40M_FIELDS, legacy.BT_40M_FIELDS)]
    if args.jsonl_events:
        sinks.append(JsonlEventSink(args.jsonl_events))
    sink = MultiSink(sinks)
    sink_writer = AsyncSinkWriter(sink, sink_queue_chunks) if args.async_sink else None

    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    parse_executor = (
        ProcessPoolExecutor(max_workers=args.parse_workers)
        if args.parse_workers > 1
        else None
    )
    source_executor = ThreadPoolExecutor(max_workers=1) if args.source_prefetch else None
    source_future = None
    args._parse_executor = parse_executor
    chunk_worker = SynchronousChunkWorker(
        pfb.process_pfb_chunk,
        runtime,
        executor=executor,
    )
    if args.split_finalize_worker:
        chunk_worker = AsyncFullyStagedChunkWorker(
            pfb.channelize_pfb_chunk,
            pfb.build_pfb_chunk_tasks,
            pfb.run_pfb_target_dsp_tasks,
            pfb.run_pfb_parser_tasks,
            pfb.finalize_pfb_chunk_results,
            runtime,
            executor=executor,
            max_queue_chunks=worker_queue_chunks,
        )
    elif args.split_target_dsp_worker:
        chunk_worker = AsyncTargetStagedChunkWorker(
            pfb.channelize_pfb_chunk,
            pfb.build_pfb_chunk_tasks,
            pfb.run_pfb_target_dsp_tasks,
            pfb.run_pfb_parser_tasks,
            pfb.finalize_pfb_chunk_results,
            runtime,
            executor=executor,
            max_queue_chunks=worker_queue_chunks,
        )
    elif args.split_stage_workers:
        chunk_worker = AsyncStagedChunkWorker(
            pfb.channelize_pfb_chunk,
            pfb.build_pfb_chunk_tasks,
            pfb.run_pfb_chunk_tasks,
            pfb.finalize_pfb_chunk_results,
            runtime,
            executor=executor,
            max_queue_chunks=worker_queue_chunks,
        )
    elif args.async_chunk_worker:
        chunk_worker = AsyncChunkWorker(chunk_worker, worker_queue_chunks)

    timing_totals = {}
    chunk_count = 0
    ble_count = 0
    bt_count = 0
    start = perf_counter()
    next_replay_deadline = start
    stats = PipelineStats(sample_rate=args.sample_rate)
    pending_chunks = []

    def handle_chunk_result(pending, chunk_result):
        nonlocal ble_count, bt_count
        chunk = pending["chunk"]
        chunk_timing = pending["timing"]
        for key, seconds in chunk_result.timing.items():
            pfb.add_timing(chunk_timing, key, seconds)
        ble_rows = chunk_result.ble_rows
        bt_rows = chunk_result.bt_rows
        if sink_writer is not None:
            sink_writer.write_rows(ble_rows, bt_rows)
            output_queue_depth = sink_writer.queue.qsize()
        else:
            sink.write_rows(ble_rows, bt_rows)
            output_queue_depth = 0
        ble_count += len(ble_rows)
        bt_count += len(bt_rows)
        chunk_wall_s = perf_counter() - pending["wall_start"]
        source_queue_depth = 1 if source_future is not None else 0
        worker_queue_depth = (
            chunk_worker.queue_depth() if args.async_chunk_worker else 0
        )
        worker_queue_detail = (
            chunk_worker.queue_detail() if args.async_chunk_worker else ""
        )
        sink_queue_detail = (
            f"sink={output_queue_depth}" if sink_writer is not None else "sink=0"
        )
        source_detail = source.stats_detail() if hasattr(source, "stats_detail") else ""
        queue_detail = (
            f"source={source_queue_depth},"
            f"{worker_queue_detail},"
            f"{sink_queue_detail}"
        )
        if source_detail:
            queue_detail = f"{queue_detail},{source_detail}"
        stats.record_chunk(
            chunk.core_end - chunk.core_start,
            len(ble_rows),
            len(bt_rows),
            chunk_wall_s=chunk_wall_s,
            source_wait_s=pending["source_wait_s"],
            buffer_wait_s=pending["buffer_wait_s"],
            queue_depth=source_queue_depth + worker_queue_depth + output_queue_depth,
            queue_detail=queue_detail,
        )
        for key, seconds in chunk_timing.items():
            pfb.add_timing(timing_totals, key, seconds)
        if args.timing:
            print_chunk_timing(chunk_timing)
        latency_warning = stats.latency_warning(args.max_latency_ms)
        if latency_warning:
            print(latency_warning)
        if stats.should_report(args.stats_interval):
            print(stats.format_report())

    try:
        if source_executor is not None:
            source_future = source_executor.submit(source.read_chunk)
        while True:
            if args.max_chunks and chunk_count >= args.max_chunks:
                break
            chunk_wall_start = perf_counter()
            chunk_timing = {}
            read_start = perf_counter()
            if source_future is not None:
                chunk = source_future.result()
            else:
                chunk = source.read_chunk()
            source_wait_s = perf_counter() - read_start
            buffer_wait_s = 0.0
            pfb.add_timing(chunk_timing, "read_chunk_s", source_wait_s)
            if chunk is None:
                break
            if hasattr(source, "rx_time_lookup"):
                runtime.rx_time_tags = source.rx_time_lookup()
                runtime.timestamp_mode = "uhd_rx_time"
            next_chunk_count = chunk_count + 1
            can_prefetch_next = not (
                args.max_chunks and next_chunk_count >= args.max_chunks
            )
            if source_executor is not None and can_prefetch_next:
                source_future = source_executor.submit(source.read_chunk)
            elif source_executor is not None:
                source_future = None

            chunk_count = next_chunk_count
            chunk_samples = (
                chunk.raw_iq.size // 2 if chunk.raw_iq.dtype.name == "int16" else chunk.raw_iq.size
            )
            print(
                f"Realtime chunk {chunk.chunk_seq}: read_start={chunk.chunk_start}, "
                f"core=[{chunk.core_start}, {chunk.core_end}), samples={chunk_samples}"
            )
            if args.reset_parser_state_per_chunk:
                args._uap_cache = LockedUAPCache()
            pending = {
                "chunk": chunk,
                "timing": chunk_timing,
                "wall_start": chunk_wall_start,
                "source_wait_s": source_wait_s,
                "buffer_wait_s": buffer_wait_s,
            }
            if args.async_chunk_worker:
                chunk_worker.submit(chunk)
                pending_chunks.append(pending)
                if len(pending_chunks) >= worker_queue_chunks:
                    handle_chunk_result(
                        pending_chunks.pop(0),
                        chunk_worker.read_result(),
                    )
            else:
                handle_chunk_result(pending, chunk_worker.process(chunk))

            if args.replay_rate > 0:
                next_replay_deadline += (chunk.core_end - chunk.core_start) / args.replay_rate
                delay = next_replay_deadline - perf_counter()
                if delay > 0:
                    sleep(delay)
        while pending_chunks:
            handle_chunk_result(
                pending_chunks.pop(0),
                chunk_worker.read_result(),
            )
    finally:
        source.close()
        if sink_writer is not None:
            sink_writer.close()
        else:
            sink.close()
        if args.async_chunk_worker:
            chunk_worker.close()
        if executor is not None:
            executor.shutdown()
        if parse_executor is not None:
            parse_executor.shutdown()
        if source_executor is not None:
            source_executor.shutdown()
        args._parse_executor = None

    pfb.add_timing(timing_totals, "total_wall_s", perf_counter() - start)
    print(f"Realtime chunks processed: {chunk_count}")
    print(f"Appended BLE rows: {ble_count} -> {output_dir / 'ble_packets.csv'}")
    print(f"Appended BR/EDR rows: {bt_count} -> {output_dir / 'btclassic_packets.csv'}")
    print(f"Appended packet events -> {output_dir / 'packet_events.csv'}")
    if args.timing:
        print(
            "Timing summary: "
            f"chunks={chunk_count}, "
            f"pfb={pfb.format_seconds(timing_totals.get('pfb_s', 0.0))}, "
            f"group_tasks={pfb.format_seconds(timing_totals.get('group_tasks_s', 0.0))}, "
            f"shift={pfb.format_seconds(timing_totals.get('shift_s', 0.0))}, "
            f"cleanup={pfb.format_seconds(timing_totals.get('cleanup_s', 0.0))}, "
            f"threshold={pfb.format_seconds(timing_totals.get('threshold_s', 0.0))}, "
            f"read_chunk={pfb.format_seconds(timing_totals.get('read_chunk_s', 0.0))}, "
            f"gpu_sync_wait={pfb.format_seconds(timing_totals.get('gpu_sync_wait_s', 0.0))}, "
            f"segment_copy_back={pfb.format_seconds(timing_totals.get('segment_copy_back_s', 0.0))}, "
            f"segment_input_build={pfb.format_seconds(timing_totals.get('segment_input_build_s', 0.0))}, "
            f"threshold_mask={pfb.format_seconds(timing_totals.get('threshold_mask_s', 0.0))}, "
            f"segment_materialize={pfb.format_seconds(timing_totals.get('segment_materialize_s', 0.0))}, "
            f"segment_materialize_calls={int(timing_totals.get('segment_materialize_calls', 0))}, "
            f"segment_materialize_samples={int(timing_totals.get('segment_materialize_samples', 0))}, "
            f"segment_count={int(timing_totals.get('segment_count', 0))}, "
            f"segment_copy_calls={int(timing_totals.get('segment_copy_calls', 0))}, "
            f"segment_copy_samples={int(timing_totals.get('segment_copy_samples', 0))}, "
            f"ble_parse={pfb.format_seconds(timing_totals.get('ble_parse_s', 0.0))}, "
            f"bredr_parse={pfb.format_seconds(timing_totals.get('bredr_parse_s', 0.0))}, "
            f"target_dsp_wall={pfb.format_seconds(timing_totals.get('target_dsp_wall_s', 0.0))}, "
            f"parser_wall={pfb.format_seconds(timing_totals.get('parser_wall_s', 0.0))}, "
            f"task_wall={pfb.format_seconds(timing_totals.get('task_wall_s', 0.0))}, "
            f"finalize={pfb.format_seconds(timing_totals.get('finalize_s', 0.0))}, "
            f"total_wall={pfb.format_seconds(timing_totals.get('total_wall_s', 0.0))}, "
            f"targets={int(timing_totals.get('targets', 0))}"
        )
    if args.target_dsp_profile:
        profile_path = output_dir / "target_dsp_profile.csv"
        rows = pfb.write_target_dsp_profile_csv(profile_path, timing_totals)
        print(f"Target DSP profile rows: {len(rows)} -> {profile_path}")
        print(pfb.format_target_dsp_profile_summary(timing_totals))


def main():
    args = build_parser().parse_args()
    run_realtime(args)


if __name__ == "__main__":
    main()
