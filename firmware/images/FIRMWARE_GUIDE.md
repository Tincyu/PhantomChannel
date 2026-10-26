# Firmware guide

The source archive places all 14 prebuilt images in this directory.  Hardware
evaluation is optional; the offline experiment package does not require a
board, an SDR, or a vendor SDK.  The commands below use the versions used by
the corresponding build/test paths.  Replace serial numbers and local tool
paths with the values reported by the programmer on your machine.

## Image matrix

| Image | Platform / board | Role | Strategy | Payload |
| --- | --- | --- | --- | --- |
| `nrf52840-peripheral-direct-tail-231b-plain.hex` | nRF52840 DK | peripheral HRS source | direct-tail | 231B, plain |
| `nrf52840-peripheral-direct-tail-231b-encrypted.hex` | nRF52840 DK | peripheral HRS source | direct-tail | 231B, encrypted |
| `nrf52840-peripheral-pip-200b-plain.hex` | nRF52840 DK | peripheral HRS source | PIP | 200B, plain |
| `nrf52840-central-runtime-tail-8b-plain.hex` | nRF52840 DK | central GATT writer | runtime tail | runtime, plain |
| `nrf52833-hrs-central-231b-plain.hex` | nRF52833 DK | HRS sink/central | ordinary peer | peer for plain 231B |
| `nrf52833-hrs-central-231b-encrypted.hex` | nRF52833 DK | HRS sink/central | ordinary peer | peer for encrypted 231B |
| `nrf52833-pip-central-200b-plain.hex` | nRF52833 DK | PIP HRS central | PIP | 200B, plain |
| `nrf52833-sink-plain.hex` | nRF52833 DK | GATT sink | ordinary peer | no covert tail |
| `da14695-blehr-pip-8b.img` | DA14695 DK | PIP/HRS transmitter | PIP | 8B |
| `efr32mg24-railtest-adv-tail-239b.hex` | EFR32MG24 BRD2703A | advertising transmitter | direct-tail | 239B |
| `cc2340r5-basic-ble-dynamic-tail-239b.hex` | CC2340R5 | advertising transmitter | direct-tail | 239B |
| `sx1280-gatt-counter-direct-tail.hex` | STM32L476RG + SX1280 | advertising transmitter | direct-tail | runtime |
| `sx1280-gatt-counter-direct-tail.bin` | STM32L476RG + SX1280 | programming image | direct-tail | runtime |
| `sx1280-gatt-counter-direct-tail.elf` | STM32L476RG + SX1280 | debug/build image | direct-tail | runtime |

## Nordic nRF52840 / nRF52833

Build inputs: nRF Connect SDK `3.0.0-rc1`, Zephyr base
`v4.0.99-ncs1-1`; the field logs used J-Link `9.28` and nrfutil device
plugin `2.20.1` (the installed nrfutil bundle reports `8.2.1`).

List boards and program an image with read-back verification and reset:

```bash
nrfutil device list --json
nrfutil device program --serial-number <DEVICE_SERIAL> \
  --firmware firmware/images/<IMAGE>.hex \
  --jlink-dll /path/to/libjlinkarm.so \
  --options verify=VERIFY_READ,reset=RESET_DEFAULT
```

For the runtime advertising image, open the DK VCOM at 115200 baud and send:

```text
phantom covert-hex <hex-bytes>
```

The expected response is `len=<N>/status=0` (the live runner also accepts
`COVERT OK len=N`).  A successful board run continues normal BLE advertising
or HRS traffic while the live USRP parser prints the target address, normal
payload, and recovered covert data.

## DA14695

Build input: Apache Mynewt/NimBLE workspace with Apache Newt `1.15.0-dev`.
Program the image through the J-Link probe (the runner uses speed 1000):

```bash
cd <MYNEWT_WORKSPACE>
newt load blehr_da14695 \
  --imgfile <SOURCE_ARCHIVE>/firmware/images/da14695-blehr-pip-8b.img \
  --extrajtagcmd "-select usb=<JLINK_SERIAL> -speed 1000" -v
```

At the DA UART, send `phantom covert-hex <hex-bytes>` (up to 239 bytes).  The
expected acknowledgement is `PHANTOM_PIP_ACK len=<N>` or the corresponding
`COVERT OK len=<N>` runtime acknowledgement.  Normal HRS traffic remains
visible to a BLE observer and the USRP parser reports the recovered PIP data.

## EFR32MG24

Build input: Silicon Labs Simplicity SDK `2025.6.3`; tested Commander version
`1v23p0b1960`.  For BRD2703A (`EFR32MG24B210F1536IM48`), list the adapter and
flash the image:

```bash
commander adapter list
commander flash firmware/images/efr32mg24-railtest-adv-tail-239b.hex \
  --device EFR32MG24B210F1536IM48 --serialno <JLINK_SERIAL>
```

At the J-Link OB VCOM, configure the tail with `setAdvTailHex <hex-bytes>`.
The board reports the configured `CovertLen` and transmit length, then
continues advertising.  The expected live result is a normal legal payload to
the BLE sniffer and the configured covert tail in the USRP parser.

## CC2340R5

Build input: TI SimpleLink Low Power F3 SDK `8.20`, TI ARM compiler profile
`5.1.0`, and UniFlash `9.4.1` with an XDS110.  UniFlash can generate a target
`.ccxml`; the equivalent non-interactive flash/verify/reset command is:

```bash
/opt/embedded/toolchains/ti/uniflash/9.4.1/dslite.sh \
  --config <CC2340R5_XDS110.ccxml> \
  --flash --verify \
  firmware/images/cc2340r5-basic-ble-dynamic-tail-239b.hex
```

After the command reports successful programming and verification, reset the
board (or power-cycle it).  On the XDS110 UART at 115200 baud, send
`covert-hex <hex-bytes>` or
`covert-clear`.  The expected response is `COVERT OK len=<N>` followed by
`Adv status: Started`; the USRP parser recovers the exact bytes while the BLE
sniffer sees only the legal payload.

## STM32L476RG + SX1280

The SX1280 image is built from the compact BTstack-based source in
`firmware/sx1280/source/`.  The `.hex` or `.bin` can be programmed with
STM32CubeProgrammer and the `.elf` is for debugging:

```bash
STM32_Programmer_CLI -c port=SWD -w \
  firmware/images/sx1280-gatt-counter-direct-tail.hex -v -rst
```

The application input format is `legal advertising data || AA AA || covert
tail`.  A successful run emits the normal advertising packet and the
configured tail on the last physical advertising channel; the single-channel
USRP parser reports the normal payload and recovered covert data.

## Host-side receiver

The common broadcast runner and live USRP parser are documented in
`docs/broadcast-realtime.md` and `docs/broadcast-multi-device.md`.  A live
receiver normally uses BLE channel 39 (2480 MHz); keep the receiver bandwidth
to one channel for the demo.

## Binary redistribution and vendor terms

The image files are supplied as evaluation artifacts alongside their build
inputs and notices.  `THIRD_PARTY_NOTICES.md` records the upstream source,
version, and license for each platform.  Those notices are attribution and do
not grant additional vendor rights: before redistributing a prebuilt Nordic,
DA14695, EFR32MG24, or CC2340R5 image beyond this artifact, verify the
applicable SDK/EULA terms.  If those terms do not permit redistribution,
rebuild locally with the listed SDK or obtain the image directly under the
vendor's terms.
