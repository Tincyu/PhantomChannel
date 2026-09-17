# PhantomChannel

PhantomChannel is an open-source research prototype for transmitting and
recovering an auxiliary frame alongside Bluetooth Low Energy traffic. The
transmitter places project data after the normal BLE CRC, while an ordinary
BLE connection continues to carry the application notification. The receiver
captures the radio signal and recovers the additional frame with the supplied
DSP and parser pipeline.

This repository contains the PhantomChannel source needed to inspect the
design, run the offline regression suite, and reproduce the Nordic transmitter
and host-side receiver setup. Large vendor SDKs, toolchains and raw IQ captures
are kept outside Git.

## Repository contents

- `firmware/` contains the nRF52840 peripheral application.
- `patches/zephyr/` contains the focused controller patch and its exact base
  revision.
- `receiver/` contains the receiver, DSP pipeline and packet parser.
- `tools/` and `configs/` contain experiment and analysis entry points.
- `tests/` contains hardware-independent regression tests.
- `ae/` contains the restricted SSH entry point for artifact evaluation.

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

The current baseline is 155 passing tests on Python 3.12.

## Hardware reproduction

The reference transmitter uses an nRF52840 DK. Apply the patch in
`patches/zephyr/` to the documented Zephyr revision, then build the application
in `firmware/nrf52840dk/phantomchannel_peripheral/`. The complete Nordic SDK is
not included.

Receiver and experiment paths are configured locally. Copy the example file
before running hardware commands:

```bash
cp config/local.env.example config/local.env
```

Edit the copied file for the workstation and review the selected YAML profile
under `configs/`. Setup notes are in [docs/setup.md](docs/setup.md); the AE
workflow is described in [docs/ae-evaluation.md](docs/ae-evaluation.md).

## Source and generated data

The repository tracks project source, focused SDK patches, tests and small
configuration files. Build output, SDK copies, raw captures, device serials,
credentials and evaluation results must remain outside Git. See
[docs/publishing.md](docs/publishing.md) before publishing changes.

PhantomChannel host-side source is released under the [MIT License](LICENSE).
Firmware files with separate license headers and external dependencies are
listed in [docs/third-party.md](docs/third-party.md).
