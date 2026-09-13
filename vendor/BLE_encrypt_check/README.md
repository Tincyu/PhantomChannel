# BLE_encrypt_check parser snapshot

This directory is the small, project-local source snapshot required by the
PhantomChannel receiver. It is intentionally not a Git submodule and it does
not contain the original project's virtual environment, captures, artifacts,
or build products.

Snapshot provenance (read from the source workstation):

- source root: `/path/to/BLE_encrypt_check`
- Git HEAD: `b973d3c75e38fa3d3f8292424126e83bf31e8065`
- source worktree had pre-existing local changes when copied; the snapshot is
  therefore a working-tree snapshot, not a claim of a clean upstream release
- copied parser path: `experiment/bt_40m_pfb_realtime.py`
- copied native path: `native/` (buildable as `bt_native`)

The public copy anonymizes the original workstation path in this README; its
checksum in `SOURCE_MANIFEST.sha256` was updated accordingly. Source-code
checksums remain unchanged from the imported snapshot.

The snapshot contains only the files imported by the PFB/realtime parser and
the native extension build. Keep it read-only during experiments. New-device
build output belongs under `PhantomChannel/artifacts/`.

To rebuild the optional C++ parser from this snapshot:

```bash
python3 tools/build_ble_native_backend.py
```

The resulting `bt_native*.so` is generated output and is not part of this
source snapshot.
