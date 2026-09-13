# Event-level timing detector firmware matrix

These overlays are consumed by `tools/prepare_event_timing_firmware.py`.
They keep the base Zephyr sample `prj.conf` files unchanged and make the six
conditions in §6 explicit.

The connected-state `peripheral-side: 240 B` label refers to the historical
on-air post-CRC segment.  The current HRS application uses
`CONFIG_PHANTOMCHANNEL_COVERT_LEN=231`; its fixed 6-byte Phantom frame plus
the controller's 3-byte CRC-like prefix gives the historical 240-byte
monitor-visible tail.  A literal 240-byte application covert field would not
fit in the 251-byte Link Layer PDU.

No file in this directory flashes hardware.  Build artifacts are placed under
`artifacts/firmware/event_timing/`; flashing remains a separate explicit step.
