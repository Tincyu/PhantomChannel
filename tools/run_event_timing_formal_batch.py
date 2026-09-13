#!/usr/bin/env python3
"""Run the randomized §6 event-level timing formal matrix.

Each condition has five independent sessions.  The 52833 support image is
flashed before the matching 52840 image for connected captures.  Flashing is
delegated to ``program_event_timing_firmware.py``, which uses the nrfutil
method frozen in the PIP/CIS firmware plan.  Each capture is staged under the
project NVMe testdata directory and archived by the single-session capture
coordinator before the NVMe source is removed.

The batch manifest is written after every task so an interrupted run can be
resumed.  Existing archived sessions with a usable session_manifest.json and
pcap are retained and counted; no existing capture is deleted or overwritten.
Failed tasks are retained in the manifest for diagnosis and are not silently
replaced by a retry.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_ARCHIVE_ROOT = Path("/path/to/PhantomChannel/experiments/event_timing/formal_20260809")
DEFAULT_SNIFFER = Path("/path/to/.nrfutil/bin/nrfutil-ble-sniffer")
EVENT_FLASHER = PROJECT_ROOT / "tools/program_event_timing_firmware.py"
CAPTURE = PROJECT_ROOT / "tools/run_event_timing_capture.py"

SERIAL_52840 = "NRF52840_SERIAL"
SERIAL_52833 = "NRF52833_SERIAL"

TARGET_ADV = "d1:22:33:44:55:66"
TARGET_CENTRAL = "c0:de:52:83:00:33"
TARGET_PERIPHERAL = "c0:de:52:84:00:03"

TASKS = [
    {"traffic": "advertising", "condition": "matched-nRF52840-benign", "image": "adv_normal", "target": TARGET_ADV, "support": None, "prefix": "timing_formal_adv_normal"},
    {"traffic": "advertising", "condition": "append-last-239B", "image": "adv_append_last", "target": TARGET_ADV, "support": None, "prefix": "timing_formal_adv_append_last"},
    {"traffic": "advertising", "condition": "append-every-239B", "image": "adv_append_every", "target": TARGET_ADV, "support": None, "prefix": "timing_formal_adv_append_every"},
    # The §6 six-condition matrix has one connected normal cell.  Use the
    # central-direction normal image for that cell; the peripheral-side tail
    # remains a separate formal cell with its HRS central support image.
    {"traffic": "connection", "condition": "normal", "image": "conn_central_normal", "target": TARGET_CENTRAL, "support": "conn_sink_52833", "prefix": "timing_formal_conn_normal"},
    {"traffic": "connection", "condition": "central-side-8B", "image": "conn_central_8b", "target": TARGET_CENTRAL, "support": "conn_sink_52833", "prefix": "timing_formal_conn_central_8b"},
    {"traffic": "connection", "condition": "peripheral-side-240B", "image": "conn_peripheral_240b", "target": TARGET_PERIPHERAL, "support": "conn_hrs_central_52833", "prefix": "timing_formal_conn_peripheral_240b"},
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def task_key(task: dict[str, Any]) -> str:
    return str(task["prefix"])


def archive_dir(args: argparse.Namespace, task: dict[str, Any], rep: int) -> Path:
    group = "advertising" if task["traffic"] == "advertising" else (
        "connection_central" if task["image"].startswith("conn_central") else "connection_peripheral"
    )
    return args.archive_root / group / f"{task['prefix']}_rep{rep}"


def existing_usable(path: Path, minimum_duration: float) -> tuple[bool, dict[str, Any]]:
    manifest_path = path / "session_manifest.json"
    pcap_path = path / "monitor/monitor.pcapng"
    if not manifest_path.is_file() or not pcap_path.is_file() or pcap_path.stat().st_size == 0:
        return False, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, {}
    usable = (
        manifest.get("status") == "archived"
        and float(manifest.get("duration_s", 0)) >= minimum_duration
        and bool(manifest.get("nvme_deleted"))
    )
    return usable, manifest


def run(command: list[str]) -> dict[str, Any]:
    started = time.time_ns()
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "started_epoch_ns": started,
        "completed_epoch_ns": time.time_ns(),
    }


def flash(condition: str, serial: str) -> dict[str, Any]:
    return run([sys.executable, str(EVENT_FLASHER), "--condition", condition, "--serial-number", serial])


def capture(args: argparse.Namespace, task: dict[str, Any], rep: int) -> dict[str, Any]:
    run_id = f"{task['prefix']}_rep{rep}"
    archive = archive_dir(args, task, rep)
    command = [
        sys.executable,
        str(CAPTURE),
        "--traffic", task["traffic"],
        "--condition", task["condition"],
        "--target-address", task["target"],
        "--duration-s", str(args.duration_s),
        "--run-id", run_id,
        "--sniffer-bin", str(args.sniffer_bin),
        "--sniffer-port", args.sniffer_port,
        "--sniffer-timeout-ms", str(args.sniffer_timeout_ms),
        "--nvme-root", str(args.nvme_root),
        "--archive-root", str(archive.parent),
        "--setup-s", str(args.setup_s),
    ]
    return run(command)


def build_tasks(seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for rep in range(1, 6):
        for task in TASKS:
            tasks.append({**task, "rep": rep, "task_id": f"{task_key(task)}::rep{rep}"})
    random.Random(seed).shuffle(tasks)
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--setup-s", type=float, default=2.0)
    parser.add_argument("--sniffer-bin", type=Path, default=DEFAULT_SNIFFER)
    parser.add_argument("--sniffer-port", default="/dev/ttyACM4")
    parser.add_argument("--sniffer-timeout-ms", type=int, default=500)
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_s < 60.0:
        raise SystemExit("§6 formal capture requires at least 60 seconds per session")
    args.sniffer_bin = args.sniffer_bin.expanduser().resolve()
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.archive_root = args.archive_root.expanduser().resolve()
    args.archive_root.mkdir(parents=True, exist_ok=True)
    batch_path = args.archive_root / "batch_manifest.json"
    tasks = build_tasks(args.seed)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "detector_roc_experiment_redesign.md §6 formal event-level timing",
        "seed": args.seed,
        "duration_s": args.duration_s,
        "conditions": {task_key(task): 5 for task in TASKS},
        "randomized_task_order": [task["task_id"] for task in tasks],
        "archive_root": str(args.archive_root),
        "nvme_root": str(args.nvme_root),
        "flashing_method": "program_event_timing_firmware.py -> pip_program_common -> nrfutil device program",
        "tasks": [],
    }
    if batch_path.is_file():
        try:
            previous = json.loads(batch_path.read_text(encoding="utf-8"))
            if previous.get("seed") == args.seed and previous.get("tasks"):
                manifest["tasks"] = previous["tasks"]
        except (OSError, json.JSONDecodeError):
            pass
    completed_ids = {item.get("task_id") for item in manifest["tasks"] if item.get("status") in {"archived", "existing"}}
    if args.dry_run:
        for task in tasks:
            print(json.dumps({"task_id": task["task_id"], "condition": task["condition"], "image": task["image"], "support": task["support"], "archive": str(archive_dir(args, task, task["rep"]))}, ensure_ascii=False))
        return 0

    for task in tasks:
        if task["task_id"] in completed_ids:
            print(f"SKIP recorded {task['task_id']}", flush=True)
            continue
        target_archive = archive_dir(args, task, task["rep"])
        usable, existing_manifest = existing_usable(target_archive, args.duration_s)
        if usable:
            record = {**task, "status": "existing", "archive": str(target_archive), "session_manifest": existing_manifest}
            manifest["tasks"].append(record)
            completed_ids.add(task["task_id"])
            write_json(batch_path, manifest)
            print(f"SKIP usable archive {target_archive}", flush=True)
            continue
        record: dict[str, Any] = {**task, "status": "running", "archive": str(target_archive), "started_epoch_ns": time.time_ns(), "steps": []}
        manifest["tasks"].append(record)
        write_json(batch_path, manifest)
        if task["support"]:
            step = flash(task["support"], SERIAL_52833)
            record["steps"].append({"role": "support_52833", **step})
            if step["returncode"] != 0:
                record.update(status="flash_failed", completed_epoch_ns=time.time_ns())
                write_json(batch_path, manifest)
                continue
        step = flash(task["image"], SERIAL_52840)
        record["steps"].append({"role": "primary_52840", **step})
        if step["returncode"] != 0:
            record.update(status="flash_failed", completed_epoch_ns=time.time_ns())
            write_json(batch_path, manifest)
            continue
        step = capture(args, task, task["rep"])
        record["steps"].append({"role": "capture", **step})
        usable, session_manifest = existing_usable(target_archive, args.duration_s)
        record["session_manifest"] = session_manifest
        record.update(status="archived" if usable else "capture_failed", completed_epoch_ns=time.time_ns())
        write_json(batch_path, manifest)
        if not usable:
            print(f"FORMAL TASK FAILED/RETAINED: {task['task_id']}", file=sys.stderr, flush=True)
    counts: dict[str, int] = {}
    for item in manifest["tasks"]:
        counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
    manifest["summary"] = counts
    manifest["completed_epoch_ns"] = time.time_ns()
    write_json(batch_path, manifest)
    print(json.dumps({"batch_manifest": str(batch_path), "summary": counts}, indent=2, ensure_ascii=False))
    return 0 if counts.get("archived", 0) + counts.get("existing", 0) == 30 else 2


if __name__ == "__main__":
    raise SystemExit(main())
