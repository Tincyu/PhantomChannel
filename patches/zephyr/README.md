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
| Local source checkout | `a013917082dce3d6d9287e848a0b45e353444158` |

In a **separate disposable checkout** of the exact base, use
`git apply --check /path/to/phantomchannel-controller.patch` before applying
it. Do not apply the patch in-place to an SDK used by other projects. The
original local checkout passes `git apply --check --reverse`, confirming the
export matches its changed files. This is a source snapshot, not a claim that
the patch has been rebuilt or hardware-validated from a fresh public clone.

The NCS `3.0.0-rc1` manifest points at `v4.0.99-ncs1-rc1`, which is **not**
the patch base. Reproducible firmware builds must pin the actual Zephyr base
above and separately verify compatibility with the remaining NCS modules.
