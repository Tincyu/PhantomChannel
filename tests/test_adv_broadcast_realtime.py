from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.adv_broadcast_realtime import (
    BLE_ADV_CHANNEL,
    BLE_ADV_FREQUENCY_HZ,
    DEFAULT_AA_TOLERANCE,
    DEFAULT_MAX_SEGMENT_LEN,
    DEFAULT_MIN_SEGMENT_LEN,
    DEFAULT_STRIP_HARDWARE_CRC,
    DEFAULT_THRESHOLD,
    build_arg_parser,
    compare_payload,
    extract_pc_frame,
    _command_for,
    parse_payload,
    run,
    unwrap_pc_frame,
)


def test_payload_accepts_priority_lengths_and_rejects_truncation() -> None:
    assert len(parse_payload(covert_hex=(bytes(range(64))).hex(), covert_data=None)) == 64
    assert len(parse_payload(covert_hex=(bytes(range(239))).hex(), covert_data=None)) == 239
    with pytest.raises(ValueError):
        parse_payload(covert_hex="00", covert_data="also supplied")
    with pytest.raises(ValueError):
        parse_payload(covert_hex="00 0", covert_data=None)
    with pytest.raises(ValueError):
        parse_payload(covert_hex=(bytes(range(240))).hex(), covert_data=None)


def test_platform_commands_keep_the_full_runtime_cli_path() -> None:
    args = build_arg_parser().parse_args(["--platform", "nrf52840"])
    assert _command_for(args, b"\x00\x01") == ["phantom", "covert-hex", "0001"]
    args = build_arg_parser().parse_args(["--platform", "cc2340r5"])
    assert _command_for(args, b"\x00\x01") == ["covert-hex", "0001"]
    args = build_arg_parser().parse_args(["--platform", "efr32mg24"])
    assert _command_for(args, b"\x00 \xff") == ["setAdvTailHex", "0020ff"]


def test_custom_hex_uart_template_does_not_decode_binary_payload() -> None:
    args = build_arg_parser().parse_args(
        ["--platform", "efr32mg24", "--uart-command", "setAdvTailHex {hex}"]
    )
    assert _command_for(args, b"\x00\xff") == ["setAdvTailHex 00ff"]


def test_live_parser_defaults_keep_weak_cc_scan_response_tail() -> None:
    args = build_arg_parser().parse_args(["--platform", "cc2340r5"])
    assert args.threshold == DEFAULT_THRESHOLD == 0.01
    assert args.min_len == DEFAULT_MIN_SEGMENT_LEN == 100
    assert args.max_len == DEFAULT_MAX_SEGMENT_LEN == 40000
    assert args.aa_tolerance == DEFAULT_AA_TOLERANCE == 2
    assert args.strip_hw_crc is DEFAULT_STRIP_HARDWARE_CRC is True


def test_compare_payload_keeps_partial_and_overflow_quality() -> None:
    expected = bytes(range(64))
    observed = expected[:31] + bytes([expected[31] ^ 0x01])
    result = compare_payload(expected, observed)
    assert result["status"] == "partial"
    assert result["matched_bytes"] == 31
    assert result["unknown_bytes"] == 32
    assert result["overflow_bytes"] == 0
    assert result["bit_errors"] == 1
    assert result["ber"] == pytest.approx(1 / (32 * 8))

    overflow = compare_payload(expected, expected + b"\xaa\xbb")
    assert overflow["status"] == "partial"
    assert overflow["unknown_bytes"] == 0
    assert overflow["overflow_bytes"] == 2


def test_unwrap_pc_frame_requires_valid_xor() -> None:
    payload = b"runtime-64"
    frame = bytearray(b"PC\x01\x00" + bytes([len(payload)]) + payload + b"\x00")
    frame[-1] = 0
    for byte in frame[:-1]:
        frame[-1] ^= byte
    assert unwrap_pc_frame(bytes(frame)) == payload
    assert extract_pc_frame(bytes(frame)) == payload
    frame[-1] ^= 0x01
    assert unwrap_pc_frame(bytes(frame)) == bytes(frame)
    assert extract_pc_frame(bytes(frame)) is None


def test_dry_run_persists_contract_for_64_bytes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = build_arg_parser().parse_args(
        [
            "--platform", "nrf52840",
            "--covert-hex", bytes(range(64)).hex(),
            "--source", "replay",
            "--input-iq", str(tmp_path / "empty.npy"),
            "--output-root", str(tmp_path),
            "--run-id", "dry-64",
            "--dry-run",
        ]
    )
    # The dry-run path records the input contract before a source is opened.
    result = run(args)
    assert result == 0
    root = tmp_path / "dry-64"
    manifest = json.loads((root / "run_manifest.json").read_text())
    usrp = json.loads((root / "usrp_config.json").read_text())
    summary = json.loads((root / "result_summary.json").read_text())
    assert manifest["input"]["len_bytes"] == 64
    assert manifest["mode"] == "broadcast_only"
    assert "PIP" in manifest["scope_exclusions"]
    assert usrp["channel"] == BLE_ADV_CHANNEL
    assert usrp["center_frequency_hz"] == BLE_ADV_FREQUENCY_HZ
    assert summary["status"] == "dry_run"
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
