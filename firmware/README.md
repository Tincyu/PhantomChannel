# Firmware collection

All board firmware supplied with this artifact is grouped under `firmware/`.
Flashable images are collected in `firmware/images/`; the platform directories
contain the small project sources or board-specific notes needed to identify
each image.

## Platforms

- `nrf52840dk/` — Nordic transmitter application source;
- `da14695/` — DA14695 board notes for the included PIP image;
- `efr32mg24/` — EFR32MG24 board notes for the included RAILtest image;
- `cc2340r5/` — CC2340R5 board notes for the included SimpleLink image;
- `sx1280/` — STM32L476RG/SX1280 direct-tail source.

The corresponding image files are in `images/`. `images/FIRMWARE_GUIDE.md`
maps every image to its board and runtime path. The complete Nordic and
SX1280 programming details remain in the platform documentation and source
bundle READMEs.
