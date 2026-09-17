# bt_native

`bt_native` is the experimental C++ extension for CPU parser hot paths.

Current scope:

- build a minimal importable extension
- keep the Python parser as the default path
- provide smoke-test APIs before moving BLE/BR parser logic into C++

Build example:

```bash
cmake -S native -B build-native \
  -DPython3_EXECUTABLE="$(pwd)/.venv-cuda/bin/python"
cmake --build build-native -j
PYTHONPATH=experiment:ble_fun_test:build-native ./.venv-cuda/bin/python \
  native/tests/test_bt_native_smoke.py
```

If `pybind11` is not installed in the active Python environment, configure with
`-DBT_NATIVE_PYBIND11_ROOT=/path/to/pybind11`.
