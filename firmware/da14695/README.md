# DA14695 firmware

`../images/da14695-blehr-pip-8b.img` is the supplied DA14695 PIP/HRS image.
The host-side broadcast runner uses the DA UART command
`phantom covert-hex <hex>`; the common receiver and platform settings are
documented in `../../docs/broadcast-realtime.md` and
`../../docs/broadcast-multi-device.md`.

The DA14695 SDK and flashing toolchain are not part of this artifact. Use the
board vendor tools to program the `.img` file.
