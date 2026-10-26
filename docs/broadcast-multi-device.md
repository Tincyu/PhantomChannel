# Multi-device BLE advertising demo

This document records the public, reproducible path for the BLE advertising
demo validated on 2026-09-18. It covers five transmitter platforms and one
common live receiver contract. The demo is advertising-only: the RF receiver
is fixed to BLE advertising channel 39 (2480 MHz), and the normal BLE payload
remains valid while the PhantomChannel frame is recovered after the BLE CRC.

## Platforms and hardware topology

Each run uses the same logical chain:

```text
host UART command -> device advertising firmware
                         |
                         +--> ordinary BLE sniffer (legal payload check)
                         |
                         +--> USRP B210, channel 39, live PhantomChannel parser
```

The validated platform matrix is:

| Platform | Device-side control | Optional legal-payload check | Runtime command | Validated lengths |
| --- | --- | --- | --- | --- |
| nRF52840 DK | J-Link VCOM | Nordic nRF52840 Dongle | `phantom covert-hex <hex>` | 64B, 239B |
| DA14695 DK | FTDI/UART and J-Link | Nordic nRF Sniffer | `phantom covert-hex <hex>` | 64B, 239B |
| EFR32MG24 BRD2703A | J-Link OB VCOM | Nordic nRF Sniffer | `setAdvTailHex <hex>` | 64B, 239B |
| CC2340R5 | XDS110 UART | Nordic nRF Sniffer | `covert-hex <hex>` | 64B, 128B, 200B, 239B |
| STM32L476RG + SX1280 | STM32 application | optional BLE observer | application-defined input | runtime |

The sniffer is a diagnostic path: it should see only the legal BLE payload and
must not be used as the covert-data receiver. The B210 is the path that
recovers the post-CRC PhantomChannel frame.

## Common runner

The public entry point is [`tools/adv_broadcast_realtime.py`](../tools/adv_broadcast_realtime.py).
It does not build or flash firmware. The selected image must already be
running and expose the platform's runtime command.

Example for a 64-byte nRF52840 run:

```bash
python3 tools/adv_broadcast_realtime.py \
  --platform nrf52840 \
  --uart /dev/ttyACM2 \
  --covert-hex "$(python3 -c 'print(bytes(range(64)).hex())')" \
  --source uhd \
  --usrp-args 'type=b200,serial=<B200_SERIAL>,master_clock_rate=16e6' \
  --duration-s 12 \
  --save-iq \
  --output-root artifacts/broadcast-runs
```

Change only `--platform`, `--uart`, and the device-specific RF setup when
switching boards. For EFR32MG24 the runner uses the binary-safe
`setAdvTailHex` command; for CC2340R5 it uses `covert-hex`. A custom firmware
command can be supplied with `--uart-command`, using `{hex}` or `{data}` as the
placeholder.

Every run writes a self-contained result directory under
`artifacts/broadcast-runs/`:

```text
input.json                  exact input bytes and length
uart_ack.json               device acknowledgement
logs/uart_transcript.log    raw host/device UART exchange
usrp_config.json            requested and actual RF settings
parser_config.json          detector and recovery settings
realtime_parser.jsonl       one record per decoded candidate
result_summary.json         exact/partial counts and capture quality
capture.fc32                optional IQ saved with --save-iq
```

The parser keeps partial candidates, BER, unknown bytes, overflow and dropped
chunk counters. `passed` means at least one exact candidate was recovered;
`partial` means the run recovered a usable but incomplete candidate.

## 2026-09-18 live result summary

These are the real hardware results recorded by the workstation test logs. The
raw IQ, PCAP and UART logs remain outside this source repository; this table is
the compact public record used to reproduce and audit the runs.

| Platform | Input | Exact | Partial | Candidates | Capture warning | Workstation run ID |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| nRF52840 | 64B | 126 | 41 | 174 | 1 overflow | `goal1-live-nrf-64-20260918` |
| nRF52840 | 239B | 114 | 53 | 197 | 2 overflows | `goal1-live-nrf-239-20260918` |
| DA14695 | 64B | 173 | 172 | 803 | 3 overflows, 10 dropped chunks | `goal1-live-da-64-20260918-r3` |
| DA14695 | 239B | 144 | 230 | 954 | 3 overflows, 10 dropped chunks | `goal1-live-da-239-20260918` |
| EFR32MG24 | 64B | 42 | 119 | 161 | 9 overflows, 71 dropped chunks | `goal1-live-efr32-64-20260918-r2` |
| EFR32MG24 | 239B | 4 | 134 | 139 | 5 overflows, 71 dropped chunks | `goal1-live-efr32-239-20260918` |
| CC2340R5 | 64B | 4 | 3 | 66 | 6 overflows, 13 dropped chunks | `cc-live-unified-64-r5-20260918` |
| CC2340R5 | 239B | 24 | 6 | 153 | 2 overflows, 13 dropped chunks | `cc-live-unified-239-r7-20260918` |

All eight live runs produced a complete candidate with BER=0. The quality
warnings are retained in each summary and do not turn a complete recovery
into a failure.

The nRF52840-specific two-command shortcut remains available:

```bash
./scripts/demo_nrf52840_tx.sh --uart /dev/ttyACM2 HELLO_AE
./scripts/demo_usrp_rx.sh \
  --usrp-args 'type=b200,serial=<B200_SERIAL>,master_clock_rate=16e6' \
  --duration 8 --once
```

For the full runtime-input and result contract, use the common runner above.

## Scope and exclusions

- The device firmware must be built and flashed separately; this document does
  not embed vendor SDKs or board-specific build trees.
- Raw IQ/PCAP captures, serial transcripts and local hardware paths are not
  committed to `gitpublic`; use `--save-iq` and an external output directory
  when preserving a new run.
- The live UART runner covers the Nordic, DA14695, EFR32MG24 and CC2340R5
  paths. The STM32/SX1280 direct-tail image is documented in
  `firmware/images/FIRMWARE_GUIDE.md` and uses its board application input
  path rather than the common UART runner.
- This demo does not cover connected traffic, PIP, whole-fragment reassembly
  or strict connected-event timing.
