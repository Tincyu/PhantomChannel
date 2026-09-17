from dataclasses import dataclass, field
from queue import Queue
from threading import Lock, Thread
from time import perf_counter


@dataclass
class PipelineStats:
    sample_rate: float
    start_time: float = field(default_factory=perf_counter)
    last_report_time: float = field(default_factory=perf_counter)
    chunks: int = 0
    core_samples: int = 0
    ble_rows: int = 0
    bt_rows: int = 0
    dropped_chunks: int = 0
    last_chunk_wall_s: float = 0.0
    max_chunk_wall_s: float = 0.0
    last_source_wait_s: float = 0.0
    max_source_wait_s: float = 0.0
    last_buffer_wait_s: float = 0.0
    max_buffer_wait_s: float = 0.0
    queue_depth: int = 0
    max_queue_depth: int = 0
    queue_detail: str = ""

    def record_chunk(
        self,
        core_samples,
        ble_count,
        bt_count,
        chunk_wall_s=0.0,
        source_wait_s=0.0,
        buffer_wait_s=0.0,
        queue_depth=0,
        queue_detail="",
    ):
        self.chunks += 1
        self.core_samples += int(core_samples)
        self.ble_rows += int(ble_count)
        self.bt_rows += int(bt_count)
        self.last_chunk_wall_s = float(chunk_wall_s)
        self.max_chunk_wall_s = max(self.max_chunk_wall_s, self.last_chunk_wall_s)
        self.last_source_wait_s = float(source_wait_s)
        self.max_source_wait_s = max(self.max_source_wait_s, self.last_source_wait_s)
        self.last_buffer_wait_s = float(buffer_wait_s)
        self.max_buffer_wait_s = max(self.max_buffer_wait_s, self.last_buffer_wait_s)
        self.queue_depth = int(queue_depth)
        self.max_queue_depth = max(self.max_queue_depth, self.queue_depth)
        self.queue_detail = str(queue_detail)

    def should_report(self, interval_s):
        return bool(interval_s and interval_s > 0 and perf_counter() - self.last_report_time >= interval_s)

    def format_report(self):
        now = perf_counter()
        elapsed = max(now - self.start_time, 1e-9)
        interval = max(now - self.last_report_time, 1e-9)
        msps = self.core_samples / elapsed / 1e6
        realtime_ratio = (
            (self.core_samples / self.sample_rate) / elapsed
            if self.sample_rate > 0
            else 0.0
        )
        self.last_report_time = now
        return (
            "Realtime stats: "
            f"chunks={self.chunks}, "
            f"samples={self.core_samples}, "
            f"throughput={msps:.3f} Msps, "
            f"realtime_ratio={realtime_ratio:.2f}x, "
            f"ble_rows={self.ble_rows}, "
            f"bt_rows={self.bt_rows}, "
            f"dropped_chunks={self.dropped_chunks}, "
            f"last_chunk_wall={self.last_chunk_wall_s * 1000.0:.3f}ms, "
            f"max_chunk_wall={self.max_chunk_wall_s * 1000.0:.3f}ms, "
            f"source_wait={self.last_source_wait_s * 1000.0:.3f}ms, "
            f"max_source_wait={self.max_source_wait_s * 1000.0:.3f}ms, "
            f"buffer_wait={self.last_buffer_wait_s * 1000.0:.3f}ms, "
            f"max_buffer_wait={self.max_buffer_wait_s * 1000.0:.3f}ms, "
            f"queue_depth={self.queue_depth}, "
            f"max_queue_depth={self.max_queue_depth}, "
            f"queue_detail={self.queue_detail}, "
            f"interval={interval:.3f}s"
        )

    def latency_warning(self, max_latency_ms):
        if not max_latency_ms or max_latency_ms <= 0:
            return ""
        latency_ms = self.last_chunk_wall_s * 1000.0
        if latency_ms <= max_latency_ms:
            return ""
        return (
            "WARNING realtime latency: "
            f"last_chunk_wall={latency_ms:.3f}ms exceeds "
            f"max_latency_ms={float(max_latency_ms):.3f}"
        )


@dataclass
class RealtimeChunkResult:
    chunk_seq: int
    ble_rows: list
    bt_rows: list
    timing: dict


@dataclass
class RealtimeChannelizedChunk:
    chunk: object
    tasks: list
    timing: dict


def _add_timing(timings, key, seconds):
    timings[key] = timings.get(key, 0.0) + seconds


class QueueWaitMetrics:
    def __init__(self, name):
        self.name = name
        self.lock = Lock()
        self.put_wait_s = 0.0
        self.get_wait_s = 0.0
        self.max_put_wait_s = 0.0
        self.max_get_wait_s = 0.0
        self.put_count = 0
        self.get_count = 0
        self.put_block_count = 0
        self.get_block_count = 0
        self.max_depth = 0

    def record_put(self, wait_s, depth):
        with self.lock:
            self.put_count += 1
            self.put_wait_s += wait_s
            self.max_put_wait_s = max(self.max_put_wait_s, wait_s)
            if wait_s > 1e-3:
                self.put_block_count += 1
            self.max_depth = max(self.max_depth, int(depth))

    def record_get(self, wait_s, depth):
        with self.lock:
            self.get_count += 1
            self.get_wait_s += wait_s
            self.max_get_wait_s = max(self.max_get_wait_s, wait_s)
            if wait_s > 1e-3:
                self.get_block_count += 1
            self.max_depth = max(self.max_depth, int(depth))

    def snapshot(self):
        with self.lock:
            return {
                "put_wait_s": self.put_wait_s,
                "get_wait_s": self.get_wait_s,
                "max_put_wait_s": self.max_put_wait_s,
                "max_get_wait_s": self.max_get_wait_s,
                "put_block_count": self.put_block_count,
                "get_block_count": self.get_block_count,
                "max_depth": self.max_depth,
            }


def _timed_put(queue, item, metrics):
    stage_start = perf_counter()
    queue.put(item)
    metrics.record_put(perf_counter() - stage_start, queue.qsize())


def _timed_get(queue, metrics):
    stage_start = perf_counter()
    item = queue.get()
    metrics.record_get(perf_counter() - stage_start, queue.qsize())
    return item


def _format_queue_detail(items):
    queue_parts = []
    wait_parts = []
    block_parts = []
    for name, queue, metrics in items:
        snapshot = metrics.snapshot()
        queue_parts.append(f"{name}={queue.qsize()}/{snapshot['max_depth']}")
        wait_parts.append(
            f"{name}:p{snapshot['put_wait_s'] * 1000.0:.3f}/"
            f"{snapshot['max_put_wait_s'] * 1000.0:.3f},"
            f"g{snapshot['get_wait_s'] * 1000.0:.3f}/"
            f"{snapshot['max_get_wait_s'] * 1000.0:.3f}"
        )
        block_parts.append(
            f"{name}:p{snapshot['put_block_count']},g{snapshot['get_block_count']}"
        )
    return (
        f"{','.join(queue_parts)},"
        f"stage_wait_ms={';'.join(wait_parts)},"
        f"stage_blocks={';'.join(block_parts)}"
    )


class SynchronousChunkWorker:
    def __init__(self, process_chunk_func, runtime, executor=None):
        self.process_chunk_func = process_chunk_func
        self.runtime = runtime
        self.executor = executor

    def process(self, chunk):
        ble_rows, bt_rows, timing = self.process_chunk_func(
            chunk.raw_iq,
            chunk.chunk_start,
            chunk.core_start,
            chunk.core_end,
            self.runtime,
            executor=self.executor,
        )
        return RealtimeChunkResult(
            chunk_seq=chunk.chunk_seq,
            ble_rows=ble_rows,
            bt_rows=bt_rows,
            timing=timing,
        )


class AsyncChunkWorker:
    def __init__(self, chunk_worker, max_queue_chunks=1):
        self.chunk_worker = chunk_worker
        self.queue = Queue(maxsize=max(1, int(max_queue_chunks)))
        self.results = Queue(maxsize=max(1, int(max_queue_chunks)))
        self.queue_metrics = QueueWaitMetrics("chunk_in")
        self.results_metrics = QueueWaitMetrics("chunk_out")
        self._error = None
        self.thread = Thread(target=self._run, name="realtime-chunk-worker", daemon=True)
        self.thread.start()

    def process(self, chunk):
        self.submit(chunk)
        return self.read_result()

    def submit(self, chunk):
        self._raise_if_failed()
        _timed_put(self.queue, chunk, self.queue_metrics)
        self._raise_if_failed()

    def read_result(self):
        result = _timed_get(self.results, self.results_metrics)
        try:
            self._raise_if_failed()
            return result
        finally:
            self.results.task_done()

    def close(self):
        _timed_put(self.queue, None, self.queue_metrics)
        self.thread.join()
        self._raise_if_failed()

    def queue_depth(self):
        return self.queue.qsize()

    def queue_detail(self):
        return _format_queue_detail(
            (
                ("chunk_in", self.queue, self.queue_metrics),
                ("chunk_out", self.results, self.results_metrics),
            )
        )

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("async chunk worker failed") from self._error

    def _run(self):
        while True:
            chunk = _timed_get(self.queue, self.queue_metrics)
            try:
                if chunk is None:
                    return
                _timed_put(
                    self.results,
                    self.chunk_worker.process(chunk),
                    self.results_metrics,
                )
            except BaseException as exc:
                self._error = exc
                return
            finally:
                self.queue.task_done()


class AsyncStagedChunkWorker:
    def __init__(
        self,
        channelize_func,
        build_tasks_func,
        run_tasks_func,
        finalize_func,
        runtime,
        executor=None,
        max_queue_chunks=1,
    ):
        queue_size = max(1, int(max_queue_chunks))
        self.channelize_func = channelize_func
        self.build_tasks_func = build_tasks_func
        self.run_tasks_func = run_tasks_func
        self.finalize_func = finalize_func
        self.runtime = runtime
        self.executor = executor
        self.input_queue = Queue(maxsize=queue_size)
        self.channelized_queue = Queue(maxsize=queue_size)
        self.results = Queue(maxsize=queue_size)
        self.input_metrics = QueueWaitMetrics("chunk_in")
        self.channelized_metrics = QueueWaitMetrics("channelized")
        self.results_metrics = QueueWaitMetrics("chunk_out")
        self._error = None
        self.channelize_thread = Thread(
            target=self._run_channelize,
            name="realtime-channelize-worker",
            daemon=True,
        )
        self.parser_thread = Thread(
            target=self._run_parser_finalize,
            name="realtime-parser-finalize-worker",
            daemon=True,
        )
        self.channelize_thread.start()
        self.parser_thread.start()

    def submit(self, chunk):
        self._raise_if_failed()
        _timed_put(self.input_queue, chunk, self.input_metrics)
        self._raise_if_failed()

    def read_result(self):
        result = _timed_get(self.results, self.results_metrics)
        try:
            self._raise_if_failed()
            return result
        finally:
            self.results.task_done()

    def close(self):
        _timed_put(self.input_queue, None, self.input_metrics)
        self.channelize_thread.join()
        self.parser_thread.join()
        self._raise_if_failed()

    def queue_depth(self):
        return self.input_queue.qsize() + self.channelized_queue.qsize()

    def queue_detail(self):
        return _format_queue_detail(
            (
                ("chunk_in", self.input_queue, self.input_metrics),
                ("channelized", self.channelized_queue, self.channelized_metrics),
                ("chunk_out", self.results, self.results_metrics),
            )
        )

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("async staged chunk worker failed") from self._error

    def _run_channelize(self):
        while True:
            chunk = _timed_get(self.input_queue, self.input_metrics)
            try:
                if chunk is None:
                    _timed_put(
                        self.channelized_queue,
                        None,
                        self.channelized_metrics,
                    )
                    return
                timing = {}
                channel_bank, stage_timing = self.channelize_func(
                    chunk.raw_iq,
                    chunk.chunk_start,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                tasks, stage_timing = self.build_tasks_func(
                    channel_bank,
                    chunk.chunk_start,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.channelized_queue,
                    RealtimeChannelizedChunk(
                        chunk=chunk,
                        tasks=tasks,
                        timing=timing,
                    ),
                    self.channelized_metrics,
                )
            except BaseException as exc:
                self._error = exc
                _timed_put(self.channelized_queue, None, self.channelized_metrics)
                return
            finally:
                self.input_queue.task_done()

    def _run_parser_finalize(self):
        while True:
            item = _timed_get(self.channelized_queue, self.channelized_metrics)
            try:
                if item is None:
                    return
                timing = dict(item.timing)
                task_results, stage_timing = self.run_tasks_func(
                    item.tasks,
                    executor=self.executor,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                ble_rows, bt_rows, stage_timing = self.finalize_func(
                    task_results,
                    item.chunk.chunk_start,
                    item.chunk.core_start,
                    item.chunk.core_end,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.results,
                    RealtimeChunkResult(
                        chunk_seq=item.chunk.chunk_seq,
                        ble_rows=ble_rows,
                        bt_rows=bt_rows,
                        timing=timing,
                    ),
                    self.results_metrics,
                )
            except BaseException as exc:
                self._error = exc
                return
            finally:
                self.channelized_queue.task_done()


class AsyncTargetStagedChunkWorker:
    def __init__(
        self,
        channelize_func,
        build_tasks_func,
        target_dsp_func,
        parser_func,
        finalize_func,
        runtime,
        executor=None,
        max_queue_chunks=1,
    ):
        queue_size = max(1, int(max_queue_chunks))
        self.channelize_func = channelize_func
        self.build_tasks_func = build_tasks_func
        self.target_dsp_func = target_dsp_func
        self.parser_func = parser_func
        self.finalize_func = finalize_func
        self.runtime = runtime
        self.executor = executor
        self.input_queue = Queue(maxsize=queue_size)
        self.channelized_queue = Queue(maxsize=queue_size)
        self.target_dsp_queue = Queue(maxsize=queue_size)
        self.results = Queue(maxsize=queue_size)
        self.input_metrics = QueueWaitMetrics("chunk_in")
        self.channelized_metrics = QueueWaitMetrics("channelized")
        self.target_dsp_metrics = QueueWaitMetrics("target_dsp")
        self.results_metrics = QueueWaitMetrics("chunk_out")
        self._error = None
        self.channelize_thread = Thread(
            target=self._run_channelize,
            name="realtime-channelize-worker",
            daemon=True,
        )
        self.target_dsp_thread = Thread(
            target=self._run_target_dsp,
            name="realtime-target-dsp-worker",
            daemon=True,
        )
        self.parser_thread = Thread(
            target=self._run_parser_finalize,
            name="realtime-parser-finalize-worker",
            daemon=True,
        )
        self.channelize_thread.start()
        self.target_dsp_thread.start()
        self.parser_thread.start()

    def submit(self, chunk):
        self._raise_if_failed()
        _timed_put(self.input_queue, chunk, self.input_metrics)
        self._raise_if_failed()

    def read_result(self):
        result = _timed_get(self.results, self.results_metrics)
        try:
            self._raise_if_failed()
            return result
        finally:
            self.results.task_done()

    def close(self):
        _timed_put(self.input_queue, None, self.input_metrics)
        self.channelize_thread.join()
        self.target_dsp_thread.join()
        self.parser_thread.join()
        self._raise_if_failed()

    def queue_depth(self):
        return (
            self.input_queue.qsize()
            + self.channelized_queue.qsize()
            + self.target_dsp_queue.qsize()
        )

    def queue_detail(self):
        return _format_queue_detail(
            (
                ("chunk_in", self.input_queue, self.input_metrics),
                ("channelized", self.channelized_queue, self.channelized_metrics),
                ("target_dsp", self.target_dsp_queue, self.target_dsp_metrics),
                ("chunk_out", self.results, self.results_metrics),
            )
        )

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("async target staged chunk worker failed") from self._error

    def _run_channelize(self):
        while True:
            chunk = _timed_get(self.input_queue, self.input_metrics)
            try:
                if chunk is None:
                    _timed_put(
                        self.channelized_queue,
                        None,
                        self.channelized_metrics,
                    )
                    return
                timing = {}
                channel_bank, stage_timing = self.channelize_func(
                    chunk.raw_iq,
                    chunk.chunk_start,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                tasks, stage_timing = self.build_tasks_func(
                    channel_bank,
                    chunk.chunk_start,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.channelized_queue,
                    RealtimeChannelizedChunk(
                        chunk=chunk,
                        tasks=tasks,
                        timing=timing,
                    ),
                    self.channelized_metrics,
                )
            except BaseException as exc:
                self._error = exc
                _timed_put(self.channelized_queue, None, self.channelized_metrics)
                return
            finally:
                self.input_queue.task_done()

    def _run_target_dsp(self):
        while True:
            item = _timed_get(self.channelized_queue, self.channelized_metrics)
            try:
                if item is None:
                    _timed_put(
                        self.target_dsp_queue,
                        None,
                        self.target_dsp_metrics,
                    )
                    return
                timing = dict(item.timing)
                target_dsp_results, stage_timing = self.target_dsp_func(
                    item.tasks,
                    executor=self.executor,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.target_dsp_queue,
                    {
                        "chunk": item.chunk,
                        "target_dsp_results": target_dsp_results,
                        "timing": timing,
                    },
                    self.target_dsp_metrics,
                )
            except BaseException as exc:
                self._error = exc
                _timed_put(self.target_dsp_queue, None, self.target_dsp_metrics)
                return
            finally:
                self.channelized_queue.task_done()

    def _run_parser_finalize(self):
        while True:
            item = _timed_get(self.target_dsp_queue, self.target_dsp_metrics)
            try:
                if item is None:
                    return
                chunk = item["chunk"]
                timing = dict(item["timing"])
                task_results, stage_timing = self.parser_func(
                    item["target_dsp_results"]
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                ble_rows, bt_rows, stage_timing = self.finalize_func(
                    task_results,
                    chunk.chunk_start,
                    chunk.core_start,
                    chunk.core_end,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.results,
                    RealtimeChunkResult(
                        chunk_seq=chunk.chunk_seq,
                        ble_rows=ble_rows,
                        bt_rows=bt_rows,
                        timing=timing,
                    ),
                    self.results_metrics,
                )
            except BaseException as exc:
                self._error = exc
                return
            finally:
                self.target_dsp_queue.task_done()


class AsyncFullyStagedChunkWorker:
    def __init__(
        self,
        channelize_func,
        build_tasks_func,
        target_dsp_func,
        parser_func,
        finalize_func,
        runtime,
        executor=None,
        max_queue_chunks=1,
    ):
        queue_size = max(1, int(max_queue_chunks))
        self.channelize_func = channelize_func
        self.build_tasks_func = build_tasks_func
        self.target_dsp_func = target_dsp_func
        self.parser_func = parser_func
        self.finalize_func = finalize_func
        self.runtime = runtime
        self.executor = executor
        self.input_queue = Queue(maxsize=queue_size)
        self.channelized_queue = Queue(maxsize=queue_size)
        self.target_dsp_queue = Queue(maxsize=queue_size)
        self.parsed_queue = Queue(maxsize=queue_size)
        self.results = Queue(maxsize=queue_size)
        self.input_metrics = QueueWaitMetrics("chunk_in")
        self.channelized_metrics = QueueWaitMetrics("channelized")
        self.target_dsp_metrics = QueueWaitMetrics("target_dsp")
        self.parsed_metrics = QueueWaitMetrics("parsed")
        self.results_metrics = QueueWaitMetrics("chunk_out")
        self._error = None
        self.channelize_thread = Thread(
            target=self._run_channelize,
            name="realtime-channelize-worker",
            daemon=True,
        )
        self.target_dsp_thread = Thread(
            target=self._run_target_dsp,
            name="realtime-target-dsp-worker",
            daemon=True,
        )
        self.parser_thread = Thread(
            target=self._run_parser,
            name="realtime-parser-worker",
            daemon=True,
        )
        self.finalize_thread = Thread(
            target=self._run_finalize,
            name="realtime-finalize-worker",
            daemon=True,
        )
        self.channelize_thread.start()
        self.target_dsp_thread.start()
        self.parser_thread.start()
        self.finalize_thread.start()

    def submit(self, chunk):
        self._raise_if_failed()
        _timed_put(self.input_queue, chunk, self.input_metrics)
        self._raise_if_failed()

    def read_result(self):
        result = _timed_get(self.results, self.results_metrics)
        try:
            self._raise_if_failed()
            return result
        finally:
            self.results.task_done()

    def close(self):
        _timed_put(self.input_queue, None, self.input_metrics)
        self.channelize_thread.join()
        self.target_dsp_thread.join()
        self.parser_thread.join()
        self.finalize_thread.join()
        self._raise_if_failed()

    def queue_depth(self):
        return (
            self.input_queue.qsize()
            + self.channelized_queue.qsize()
            + self.target_dsp_queue.qsize()
            + self.parsed_queue.qsize()
        )

    def queue_detail(self):
        return _format_queue_detail(
            (
                ("chunk_in", self.input_queue, self.input_metrics),
                ("channelized", self.channelized_queue, self.channelized_metrics),
                ("target_dsp", self.target_dsp_queue, self.target_dsp_metrics),
                ("parsed", self.parsed_queue, self.parsed_metrics),
                ("chunk_out", self.results, self.results_metrics),
            )
        )

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("async fully staged chunk worker failed") from self._error

    def _run_channelize(self):
        while True:
            chunk = _timed_get(self.input_queue, self.input_metrics)
            try:
                if chunk is None:
                    _timed_put(
                        self.channelized_queue,
                        None,
                        self.channelized_metrics,
                    )
                    return
                timing = {}
                channel_bank, stage_timing = self.channelize_func(
                    chunk.raw_iq,
                    chunk.chunk_start,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                tasks, stage_timing = self.build_tasks_func(
                    channel_bank,
                    chunk.chunk_start,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.channelized_queue,
                    RealtimeChannelizedChunk(
                        chunk=chunk,
                        tasks=tasks,
                        timing=timing,
                    ),
                    self.channelized_metrics,
                )
            except BaseException as exc:
                self._error = exc
                _timed_put(self.channelized_queue, None, self.channelized_metrics)
                return
            finally:
                self.input_queue.task_done()

    def _run_target_dsp(self):
        while True:
            item = _timed_get(self.channelized_queue, self.channelized_metrics)
            try:
                if item is None:
                    _timed_put(
                        self.target_dsp_queue,
                        None,
                        self.target_dsp_metrics,
                    )
                    return
                timing = dict(item.timing)
                target_dsp_results, stage_timing = self.target_dsp_func(
                    item.tasks,
                    executor=self.executor,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.target_dsp_queue,
                    {
                        "chunk": item.chunk,
                        "target_dsp_results": target_dsp_results,
                        "timing": timing,
                    },
                    self.target_dsp_metrics,
                )
            except BaseException as exc:
                self._error = exc
                _timed_put(self.target_dsp_queue, None, self.target_dsp_metrics)
                return
            finally:
                self.channelized_queue.task_done()

    def _run_parser(self):
        while True:
            item = _timed_get(self.target_dsp_queue, self.target_dsp_metrics)
            try:
                if item is None:
                    _timed_put(self.parsed_queue, None, self.parsed_metrics)
                    return
                timing = dict(item["timing"])
                task_results, stage_timing = self.parser_func(
                    item["target_dsp_results"]
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.parsed_queue,
                    {
                        "chunk": item["chunk"],
                        "task_results": task_results,
                        "timing": timing,
                    },
                    self.parsed_metrics,
                )
            except BaseException as exc:
                self._error = exc
                _timed_put(self.parsed_queue, None, self.parsed_metrics)
                return
            finally:
                self.target_dsp_queue.task_done()

    def _run_finalize(self):
        while True:
            item = _timed_get(self.parsed_queue, self.parsed_metrics)
            try:
                if item is None:
                    return
                chunk = item["chunk"]
                timing = dict(item["timing"])
                ble_rows, bt_rows, stage_timing = self.finalize_func(
                    item["task_results"],
                    chunk.chunk_start,
                    chunk.core_start,
                    chunk.core_end,
                    self.runtime,
                )
                for key, seconds in stage_timing.items():
                    _add_timing(timing, key, seconds)
                _timed_put(
                    self.results,
                    RealtimeChunkResult(
                        chunk_seq=chunk.chunk_seq,
                        ble_rows=ble_rows,
                        bt_rows=bt_rows,
                        timing=timing,
                    ),
                    self.results_metrics,
                )
            except BaseException as exc:
                self._error = exc
                return
            finally:
                self.parsed_queue.task_done()


class AsyncSinkWriter:
    def __init__(self, sink, max_queue_chunks=4):
        self.sink = sink
        self.queue = Queue(maxsize=max(1, int(max_queue_chunks)))
        self._error = None
        self.thread = Thread(target=self._run, name="realtime-output-writer", daemon=True)
        self.thread.start()

    def write_rows(self, ble_rows, bt_rows):
        self._raise_if_failed()
        self.queue.put((ble_rows, bt_rows))
        self._raise_if_failed()

    def close(self):
        self.queue.put(None)
        self.thread.join()
        try:
            self._raise_if_failed()
        finally:
            self.sink.close()

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("async sink writer failed") from self._error

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                ble_rows, bt_rows = item
                self.sink.write_rows(ble_rows, bt_rows)
            except BaseException as exc:
                self._error = exc
                return
            finally:
                self.queue.task_done()
