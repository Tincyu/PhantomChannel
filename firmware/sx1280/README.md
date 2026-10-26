# STM32L476RG + SX1280 firmware

This directory contains the compact BTstack direct-tail source for the
STM32L476RG Nucleo and SX1280 radio. The source is under `source/`; the
corresponding flashable files are in `../images/`:

- `sx1280-gatt-counter-direct-tail.hex`
- `sx1280-gatt-counter-direct-tail.bin`
- `sx1280-gatt-counter-direct-tail.elf`

The application input format is `legal advertising data || AA AA || covert
tail`. Build and board details are described in `source/Makefile` and the
source comments. The STM32/SX1280 SDK and toolchain are not part of this
artifact.
