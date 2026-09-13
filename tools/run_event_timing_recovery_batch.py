#!/usr/bin/env python3
"""Recover §6 sessions whose first static-follow batch lacked coverage.

The original 30-session batch is retained as a diagnostic archive.  This
batch uses the already validated dynamic extcap follow path and resets the
boards only after the monitor is running, so the sniffer can learn the target
advertising/connection context.  Firmware programming is still delegated to
the PIP-plan nrfutil wrapper.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASH = PROJECT_ROOT / "tools/program_event_timing_firmware.py"
CAPTURE = PROJECT_ROOT / "tools/run_cis_x310_evidence_capture.py"
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_ARCHIVE_ROOT = Path("/path/to/PhantomChannel/experiments/event_timing/formal_20260809/recovery")
S40 = "NRF52840_SERIAL"
S33 = "NRF52833_SERIAL"

TASKS = [
    {"group": "advertising", "condition": "append-last-239B", "image": "adv_append_last", "support": None, "target": "d1:22:33:44:55:66", "prefix": "timing_formal_recovery_adv_append_last"},
    {"group": "connection_central", "condition": "normal", "image": "conn_central_normal", "support": "conn_sink_52833", "target": "c0:de:52:83:00:33", "prefix": "timing_formal_recovery_conn_normal"},
    {"group": "connection_central", "condition": "central-side-8B", "image": "conn_central_8b", "support": "conn_sink_52833", "target": "c0:de:52:83:00:33", "prefix": "timing_formal_recovery_conn_central_8b"},
    {"group": "connection_peripheral", "condition": "peripheral-side-240B", "image": "conn_peripheral_240b", "support": "conn_hrs_central_52833", "target": "c0:de:52:84:00:03", "prefix": "timing_formal_recovery_conn_peripheral_240b"},
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(command: list[str]) -> dict[str, Any]:
    started = time.time_ns()
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    return {"command": command, "returncode": result.returncode, "started_epoch_ns": started, "completed_epoch_ns": time.time_ns()}


def archive_path(args: argparse.Namespace, task: dict[str, Any], rep: int) -> Path:
    return args.archive_root / task["group"] / f"{task['prefix']}_rep{rep}"


def usable(path: Path) -> bool:
    manifest = path / "session_manifest.json"
    pcap = path / "monitor/monitor.pcapng"
    if not manifest.is_file() or not pcap.is_file() or pcap.stat().st_size == 0:
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("status") == "captured" and data.get("nvme_deleted") is True and pcap.stat().st_size > 0


def flash(condition: str, serial: str) -> dict[str, Any]:
    return run([sys.executable, str(FLASH), "--condition", condition, "--serial-number", serial])


def capture(args: argparse.Namespace, task: dict[str, Any], rep: int) -> dict[str, Any]:
    run_id = f"{task['prefix']}_rep{rep}"
    target = archive_path(args, task, rep)
    return run([
        sys.executable, str(CAPTURE),
        "--trace-kind", "acl",
        "--duration-s", str(args.duration_s),
        "--target-address", task["target"],
        "--sniffer-port", args.sniffer_port,
        "--sniffer-timeout-ms", str(args.sniffer_timeout_ms),
        "--run-id", run_id,
        "--sniffer-only",
        "--reset-boards",
        "--nvme-root", str(args.nvme_root),
        "--archive-root", str(target.parent),
        "--setup-s", str(args.setup_s),
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--setup-s", type=float, default=2.0)
    parser.add_argument("--sniffer-port", default="/dev/ttyACM4")
    parser.add_argument("--sniffer-timeout-ms", type=int, default=5000)
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_s < 60.0:
        raise SystemExit("formal recovery sessions require at least 60 seconds")
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.archive_root = args.archive_root.expanduser().resolve()
    args.archive_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for task in TASKS:
        start = 1 if task["prefix"].endswith("adv_append_last") else 1
        for rep in range(start, 6):
            tasks.append({**task, "rep": rep, "task_id": f"{task['prefix']}::rep{rep}"})
    random.Random(args.seed).shuffle(tasks)
    manifest_path = args.archive_root / "recovery_batch_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "recover static-follow-invalid §6 sessions",
        "seed": args.seed,
        "duration_s": args.duration_s,
        "flashing_method": "program_event_timing_firmware.py -> pip_program_common -> nrfutil device program",
        "tasks": [],
    }
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("seed") == args.seed:
                manifest["tasks"] = previous.get("tasks", [])
        except (OSError, json.JSONDecodeError):
            pass
    done = {item.get("task_id") for item in manifest["tasks"] if item.get("status") == "archived"}
    if args.dry_run:
        for task in tasks:
            print(json.dumps({"task_id": task["task_id"], "archive": str(archive_path(args, task, task["rep"]))}, ensure_ascii=False))
        return 0

    for task in tasks:
        if task["task_id"] in done:
            print(f"SKIP {task['task_id']}", flush=True)
            continue
        target = archive_path(args, task, task["rep"])
        if usable(target):
            manifest["tasks"].append({**task, "status": "archived", "archive": str(target), "preexisting": True})
            done.add(task["task_id"])
            write_json(manifest_path, manifest)
            continue
        record: dict[str, Any] = {**task, "status": "running", "archive": str(target), "steps": []}
        manifest["tasks"].append(record)
        write_json(manifest_path, manifest)
        if task["support"]:
            step = flash(task["support"], S33)
            record["steps"].append({"role": "support_52833", **step})
            if step["returncode"] != 0:
                record["status"] = "flash_failed"
                write_json(manifest_path, manifest)
                continue
        step = flash(task["image"], S40)
        record["steps"].append({"role": "primary_52840", **step})
        if step["returncode"] != 0:
            record["status"] = "flash_failed"
            write_json(manifest_path, manifest)
            continue
        step = capture(args, task, task["rep"])
        record["steps"].append({"role": "dynamic_capture", **step})
        record["status"] = "archived" if usable(target) else "capture_failed"
        record["completed_epoch_ns"] = time.time_ns()
        write_json(manifest_path, manifest)
    counts: dict[str, int] = {}
    for item in manifest["tasks"]:
        counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
    manifest["summary"] = counts
    manifest["completed_epoch_ns"] = time.time_ns()
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "summary": counts}, indent=2, ensure_ascii=False))
    return 0 if counts.get("archived", 0) == 20 else 2


if __name__ == "__main__":
    raise SystemExit(main())
