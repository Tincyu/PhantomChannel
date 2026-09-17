# PhantomChannel receiver

This directory contains the project-owned BLE receiver and parser
implementation. It is developed as part of PhantomChannel and is not a renamed
copy of another project, a third-party dependency, or a Git submodule.

The published source includes:

- the PFB and realtime DSP entry points under `experiment/`;
- shared channelizer, parser and streaming components under
  `experiment/bt_pipeline/`;
- BLE packet matching helpers under `ble_fun_test/`; and
- the optional C++ parser extension under `native/`, buildable as `bt_native`.

`SOURCE_MANIFEST.sha256` records the integrity of the published receiver source.
Generated native libraries, virtual environments, captures and build output
are deliberately excluded. Build output belongs under the ignored project
`artifacts/` directory.

Repository tools resolve this directory automatically. External integrations
may set `PHANTOM_RECEIVER_ROOT`; the former generic `PHANTOM_BLE_ROOT`
environment variable remains accepted as a compatibility alias. The obsolete
`vendor/...` path is not retained.

To rebuild the optional C++ parser:

```bash
python3 tools/build_ble_native_backend.py
```

The resulting `bt_native*.so` is generated output and is not committed.
