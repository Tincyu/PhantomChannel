# PhantomChannel

**Accepted at ACM CCS 2026**

*PhantomChannel: A Parasitic Covert Channel via a Parsing-Boundary Blind Spot
in BLE Link-Layer Traffic*

PhantomChannel is an open-source research prototype for transmitting and
recovering an auxiliary frame alongside Bluetooth Low Energy traffic. The
transmitter places project data after the normal BLE CRC, while an ordinary
BLE connection continues to carry the application notification. The receiver
captures the radio signal and recovers the additional frame with the supplied
DSP and parser pipeline.

This repository contains the PhantomChannel source needed to inspect the
design, run the offline regression suite, and reproduce the multi-platform
transmitter and host-side receiver setup. Large vendor SDKs, toolchains and raw
IQ captures are kept outside Git.

## Repository contents

- `firmware/` contains the Nordic, DA14695, EFR32MG24, CC2340R5 and SX1280
  platform sources and board notes.
- `firmware/images/` contains the corresponding release images and their
  platform guide.
- `patches/zephyr/` contains the focused controller patch and its exact base
  revision.
- `receiver/` contains the receiver, DSP pipeline and packet parser.
- `tools/` and `configs/` contain experiment and analysis entry points.
- `tests/` contains hardware-independent regression tests.
- `reproduction/` contains the offline figure, table, tail-energy and
  auxiliary-experiment analysis code and the one-command package runner.

## Quick start

The offline suite is the simplest way to check a fresh clone. It does not
require an SDR, development board, CUDA or a vendor SDK.

```bash
git clone https://github.com/Tincyu/PhantomChannel.git
cd PhantomChannel
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
```

The source regression suite contains 158 tests.

## Offline experiment reproduction

Install the analysis dependencies and run against the released experiment ZIP:

```bash
.venv/bin/python -m pip install -r reproduction/requirements.txt
.venv/bin/python reproduction/run.py \
  --bundle /path/to/PhantomChannel-experiment-reproduction-2026.09.19-ae.zip \
  --output-dir artifacts/offline-reproduction
```

The runner uses the analysis code in this checkout with the analysis-ready
inputs in the ZIP. It regenerates Figures 10–12, Tables 3–5, tail-energy
outputs, BPL/CAF NLOS results and the 60 m antenna comparison. The antenna
scorer recomputes PSR/goodput from all 19 retained parser records:
default 55.0% / 53.94 kbps; SAS-571 88.6% / 86.79 kbps.

See [reproduction/README.md](reproduction/README.md) for inputs, outputs and
paper correspondence. This workflow requires Python and the experiment
bundle; radio hardware is optional.

## Hardware reproduction

The reference transmitter uses an nRF52840 DK. Apply the patch in
`patches/zephyr/` to the documented Zephyr revision, then build the application
in `firmware/nrf52840dk/phantomchannel_peripheral/`. The other platform image
paths and board notes are under `firmware/`; their vendor SDKs and toolchains
are not included.

Receiver and experiment paths are configured locally. Copy the example file
before running hardware commands:

```bash
cp config/local.env.example config/local.env
```

Edit the copied file for the workstation and review the selected YAML profile
under `configs/`. Setup notes are in [docs/setup.md](docs/setup.md).

For the broadcast-only runtime path, use
[`tools/adv_broadcast_realtime.py`](tools/adv_broadcast_realtime.py). It accepts
hex data at run time, fixes the B210 to BLE channel 39 (2480 MHz), and writes
the input, UART acknowledgement, requested/actual USRP configuration, JSONL
candidate stream and quality summary into a new run directory. Its contract
and platform examples are documented in
[docs/broadcast-realtime.md](docs/broadcast-realtime.md). The platform matrix
and the 2026-09-18 four-platform live result summary are in
[docs/broadcast-multi-device.md](docs/broadcast-multi-device.md).

The 14 firmware image files for Nordic, DA14695, EFR32MG24, CC2340R5 and
SX1280 are collected in [firmware/images/](firmware/images/). Start with
[FIRMWARE_GUIDE.md](firmware/images/FIRMWARE_GUIDE.md) for programming tools,
commands and board behavior. Hardware evaluation is optional for offline
functional evaluation.

## Source and generated data

The repository tracks project source, focused SDK patches, tests, small
configuration files and explicitly selected release images under
`firmware/images/`. Other build output, SDK copies, raw captures, device
serials, credentials and evaluation results must remain outside Git. See
[docs/publishing.md](docs/publishing.md) before publishing changes.

PhantomChannel host-side source is released under the [MIT License](LICENSE).
Firmware files with separate license headers and external dependencies are
listed in [docs/third-party.md](docs/third-party.md). The release-specific
third-party source, firmware, SDK and runtime notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
