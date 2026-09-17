# PhantomChannel

PhantomChannel is the source repository for the transmitter experiments, BLE
receiver/parser, and reproducible offline analyses. The repository is based on
the original PhantomChannel and `BLE_encrypt_check` projects, both owned by the
project author. SDKs, toolchains, captures, device-specific configuration, and
evaluation results are intentionally kept outside Git.

## Layout

| Path | Purpose |
|---|---|
| `firmware/`, `patches/` | Transmitter sample and the focused Zephyr controller patch; external SDKs are not vendored |
| `tools/`, `native/` | Host-side analysis, build, capture, and parser code |
| `vendor/BLE_encrypt_check/` | Author-owned receiver/parser source snapshot |
| `configs/`, `config/local.env.example` | Example experiment settings and private-path template |
| `tests/` | Offline regression and AE gateway tests (no SDR or board required) |
| `ae/` | Versioned restricted SSH evaluation gateway and config template |
| `scripts/`, `docs/` | Verification entry points and public setup/evaluation guidance |

## Offline test

Use Python 3.12. On a fresh clone:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q tests
```

The expected current baseline is 155 passing offline tests. These tests do not
claim that a transmitter board was flashed or that a new RF capture passed.
For optional CUDA parsing, install `requirements-cuda.txt` in a separate
environment and run the checks in [setup](docs/setup.md).

## Hardware and evaluation

Copy `config/local.env.example` to the ignored `config/local.env` and set paths
and device identifiers for the target workstation. Review each YAML file before
running any hardware command; example paths and identifiers are deliberately
non-operational. See [setup](docs/setup.md) and [AE evaluation](docs/ae-evaluation.md).

The AE SSH gateway source is in `ae/`; its machine-local config, keys and
results are deployed outside this Git repository. The public offline test
command is the same command used by the gateway; hardware profiles require
separate authorization and evidence.

## Publication boundary

This checkout contains source and small configuration files only. Do not commit
SDK copies, `.venv*`, `config/local.env`, raw IQ, board serials, logs, generated
firmware, or AE credentials. Follow the [safe publishing guide](docs/publishing.md),
run `scripts/check_public.py`, and check unpushed history with
`scripts/check_push_size.py --base origin/main` before each push.

The project-owned host source is released under the [MIT license](LICENSE).
Some firmware files retain their own license headers; see
[third-party notes](docs/third-party.md) for those exceptions, external
dependencies and the author-owned `BLE_encrypt_check` snapshot.
