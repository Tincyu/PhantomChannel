#!/usr/bin/env python3
"""Forced-command SSH gateway for the frozen, offline AE test profile.

No shell, user-supplied paths, package installation, network calls or hardware
operations are performed. Deployment and account isolation are still required.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path


RUN_ID = re.compile(r"^[0-9a-f]{32}$")
COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
MAX_LOG_CHARS = 262_144
TIMEOUT_SECONDS = 600


def parse_command(raw: str) -> tuple[str, str | None]:
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        raise ValueError("invalid command") from exc
    if parts == ["help"]:
        return "help", None
    if parts == ["run", "offline"]:
        return "run", "offline"
    if len(parts) == 2 and parts[0] in {"status", "fetch"} and RUN_ID.fullmatch(parts[1]):
        return parts[0], parts[1]
    raise ValueError("command not allowed")


def load_config(path: Path) -> dict[str, Path | str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"repo", "python", "results", "expected_commit"}:
        raise ValueError("config must contain exactly repo, python, results, expected_commit")
    if not COMMIT.fullmatch(data["expected_commit"]):
        raise ValueError("expected_commit must be a full Git commit hash")
    config: dict[str, Path | str] = {"expected_commit": data["expected_commit"]}
    for name in ("repo", "python", "results"):
        value = Path(data[name])
        if not value.is_absolute():
            raise ValueError(f"{name} must be absolute")
        config[name] = value
    return config


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
        text=True, timeout=10,
    )
    return result.stdout.strip()


def verify_checkout(repo: Path, expected_commit: str) -> None:
    if not repo.is_dir():
        raise ValueError("frozen checkout missing")
    actual = git_output(repo, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ValueError("checkout commit does not match configured frozen commit")
    if git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("checkout is not clean")


def report_path(results: Path, run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run id")
    return results / run_id / "result.json"


def run_offline(config: dict[str, Path | str]) -> dict[str, object]:
    repo = config["repo"]
    python = config["python"]
    results = config["results"]
    expected = config["expected_commit"]
    assert isinstance(repo, Path) and isinstance(python, Path)
    assert isinstance(results, Path) and isinstance(expected, str)
    if not python.is_file() or not results.is_dir():
        raise ValueError("Python environment or result directory missing")
    verify_checkout(repo, expected)

    lock_path = results / ".offline.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("offline evaluator is busy") from exc
        run_id = uuid.uuid4().hex
        run_dir = results / run_id
        run_dir.mkdir(mode=0o700)
        started = time.time()
        environment = {
            "HOME": str(results),
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        try:
            completed = subprocess.run(
                [str(python), "-m", "pytest", "-q", "tests"],
                cwd=repo, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS, env=environment, check=False,
            )
            exit_code = completed.returncode
            log = completed.stdout + completed.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            log = str(exc)
            timed_out = True
        log = log[:MAX_LOG_CHARS]
        (run_dir / "test.log").write_text(log, encoding="utf-8")
        report: dict[str, object] = {
            "run_id": run_id,
            "profile": "offline",
            "commit": expected,
            "started_at_unix": started,
            "duration_seconds": round(time.time() - started, 3),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "passed": exit_code == 0,
        }
        report_path(results, run_id).write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    raw = os.environ.get("SSH_ORIGINAL_COMMAND")
    if raw is None:
        print("SSH forced-command context required", file=sys.stderr)
        return 2
    try:
        operation, argument = parse_command(raw)
        if operation == "help":
            print("Allowed: run offline | status <run-id> | fetch <run-id> | help")
            return 0
        config = load_config(args.config)
        results = config["results"]
        assert isinstance(results, Path)
        if operation == "run":
            report = run_offline(config)
            print(json.dumps(report, sort_keys=True))
            return 0 if report["passed"] else 1
        assert isinstance(argument, str)
        path = report_path(results, argument)
        if not path.is_file():
            raise ValueError("run not found")
        if operation == "status":
            print(path.read_text(encoding="utf-8"), end="")
        elif operation == "fetch":
            print((path.parent / "test.log").read_text(encoding="utf-8")[:MAX_LOG_CHARS], end="")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"AE request rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
