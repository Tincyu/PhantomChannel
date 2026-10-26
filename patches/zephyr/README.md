# Focused Zephyr controller patch

`phantomchannel-controller.patch` contains the project-related Bluetooth
controller/host and `samples/bluetooth/phantomchannel_*` changes exported from
the local `covert-notify` branch. It deliberately omits the rest of the Zephyr
SDK, unrelated mode/documentation differences, and sample READMEs containing
old workstation paths.

| Item | Git revision |
|---|---|
| Upstream repository | `https://github.com/nrfconnect/sdk-zephyr.git` |
| Exact patch base | tag `v4.0.99-ncs1-1`, commit `77f865b8f8d0cb3d19002bfe713e9dd46e6f71b7` |
| Patch-stack source checkout | `a013917082dce3d6d9287e848a0b45e353444158` |

In a **separate disposable checkout** of the exact base, use
`git apply --check /path/to/phantomchannel-controller.patch` before applying
it. Do not apply the patch in-place to an SDK used by other projects. The
original local checkout passes `git apply --check --reverse`, confirming the
export matches its changed files. This is a source snapshot, not a claim that
the patch has been rebuilt or hardware-validated from a fresh public clone.

For the central runtime path, apply the patches in this order:

1. `phantomchannel-controller.patch`
2. `phantomchannel-goal2-runtime.patch`
3. `phantomchannel-central-runtime-tail.patch`
4. `phantomchannel-encrypted-direct-tail.patch`

The final incremental patch keeps the GATT write at one legal prefix byte,
adds the `bt_phantomchannel_central_set_covert()` controller API and
controller-owned runtime buffer, and appends the generated PC frame after the
standard BLE CRC. The runtime shell accepts 1--239 bytes of arbitrary hex
without rebuilding the application.

The fourth patch contains the source changes for encrypted direct-tail
transmission: the legal inner PDU is sent through Nordic hardware CCM before
the software CRC and post-CRC tail are appended. It does not enable encrypted
PIP; that controller path remains explicitly rejected. The patch is
reverse-checked against local source commit `9116a1378e6` (parent
`67b838faaba9`).

The NCS `3.0.0-rc1` manifest points at `v4.0.99-ncs1-rc1`, which is **not**
the patch base. Reproducible firmware builds must pin the actual Zephyr base
above and separately verify compatibility with the remaining NCS modules.
