# CC2340R5 firmware

`../images/cc2340r5-basic-ble-dynamic-tail-239b.hex` is the supplied
SimpleLink Basic BLE advertising image. Its host-side runtime command is
`covert-hex <hex>`; the common receiver and platform settings are documented
in `../../docs/broadcast-realtime.md` and
`../../docs/broadcast-multi-device.md`.

The TI SimpleLink SDK and XDS110 tools are not part of this artifact. Use the
board vendor tools to program the `.hex` file.
