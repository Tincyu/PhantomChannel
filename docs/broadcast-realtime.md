# Broadcast realtime path

For the cross-platform device matrix and the 2026-09-18 live results, see
[Multi-device BLE advertising demo](broadcast-multi-device.md). This page
documents the common runner contract and its output files.

`tools/adv_broadcast_realtime.py` is the host-side entry point for the first
handoff goal. It intentionally covers BLE advertising only. Connection state,
PIP, whole-fragment reassembly and strict TIFS are out of scope.

Every run creates a unique directory containing:

- `input.json` — the exact runtime covert data;
- `uart_ack.json` and `logs/uart_transcript.log` — the host command and device
  acknowledgement;
- `usrp_config.json` — B210 settings, including actual UHD values when the
  device reports them;
- `parser_config.json` — detector and tolerant-tail options;
- `realtime_parser.jsonl` — one normalized record per candidate, including
  sample offset, recovered data, candidate/hit counters, BER, byte hit rate,
  unknown range and overflow-related quality context; and
- `result_summary.json` — the run-level status and capture quality.

The RF contract is fixed to BLE advertising channel 39, center frequency
2480 MHz, 4 MS/s and 4 MHz bandwidth. UHD overflow is retained as a quality
warning. A run is `passed` when at least one exact candidate is recovered and
`partial` when only a usable partial candidate is recovered.

Examples:

```bash
python3 tools/adv_broadcast_realtime.py \
  --platform nrf52840 --uart /dev/ttyACM2 \
  --covert-hex "$(python3 -c 'print(bytes(range(64)).hex())')" \
  --source uhd --duration-s 12 --save-iq
```

For CC2340R5 the default runtime command is `covert-hex <hex>`. For nRF52840
and DA14695 it is `phantom covert-hex <hex>`. EFR32MG24 uses the binary-safe
`setAdvTailHex <hex>` command after selecting channel 39 and setting the legal
visible name. Its default path accepts arbitrary byte values, including NUL and
bytes above `0x7f`, with a length of 1 through 239 bytes. The runner performs the
required continuous-TX preflight automatically. Use `--uart-command` when a
board image exposes a different runtime command.

For example, the EFR32MG24 path accepts binary data that cannot be represented
by the former ASCII-only tail argument:

```bash
python3 tools/adv_broadcast_realtime.py \
  --platform efr32mg24 --uart /dev/ttyACM3 \
  --covert-hex 00ff807f0102 \
  --source uhd --duration-s 12 --save-iq
```

An existing SC16 capture can be replayed through the same output contract:

```bash
python3 tools/adv_broadcast_realtime.py \
  --platform cc2340r5 --no-uart --source replay \
  --input-iq /path/to/capture.sc16 --iq-format sc16 \
  --covert-hex 00010203 \
  --threshold 0.01 --aa-tolerance 2 --strip-hw-crc --no-drop-last-extra
```

The tool does not build or flash firmware; the firmware must already expose
the platform's runtime input command and must be in broadcast mode.

## nRF52840 demo shortcuts

The public demo has two small entry points. They assume the nRF52840 already
runs the advertising firmware:

```bash
./scripts/demo_nrf52840_tx.sh --uart /dev/ttyACM2 "HELLO_AE"
./scripts/demo_usrp_rx.sh
```

The transmitter also accepts binary input with `--hex`. The receiver listens
continuously on channel 39 and prints only tails with a valid PhantomChannel
frame checksum; use `--once` to stop after the first result. Run either script
with `--help` for UART, advertiser-address, and USRP overrides.
