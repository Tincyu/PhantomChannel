# EFR32MG24 firmware

`../images/efr32mg24-railtest-adv-tail-239b.hex` is the supplied BRD2703A
RAILtest advertising image. Its host-side runtime command is
`setAdvTailHex <hex>`; the common receiver and platform settings are
documented in `../../docs/broadcast-realtime.md` and
`../../docs/broadcast-multi-device.md`.

The Silicon Labs SDK and Commander tool are not part of this artifact. Use
the board vendor tools to program the `.hex` file.
