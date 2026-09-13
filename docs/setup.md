# Setup and verification

The offline acceptance gate needs only Python 3.12 and the packages in
`requirements-test.txt`. Run from a fresh clone:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/check_public.py
```

The test suite is independent of hardware. CUDA is optional; to check that
runtime separately, install `requirements-cuda.txt` in a dedicated Python
environment, run `python -m pip check`, and verify that
`cupy.cuda.runtime.getDeviceCount()` is nonzero. Native parser builds use the
external CMake, compiler and `pybind11` toolchain.

NCS/Zephyr, nRF5 SDK, UHD, J-Link, TI and Silicon Labs packages are external
dependencies. `config/local.env.example` lists private path variables. The
`configs/` examples also contain placeholders for the original lab; replace
them and verify exact board type and serial before running a flash or capture.
Never treat an offline test pass as hardware or RF acceptance.

The project-owned Zephyr controller changes are exported as a focused patch in
[`patches/zephyr`](../patches/zephyr/README.md). Its base is the official
`v4.0.99-ncs1-1` tag, **not** the `v4.0.99-ncs1-rc1` revision in the NCS
`3.0.0-rc1` manifest. Do not apply it to a different revision without a
separate compatibility check. Firmware build and flash acceptance on a clean
clone remains a separate gate.
