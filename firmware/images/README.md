# Release firmware images

This directory collects the flashable firmware images for every hardware path
included in the artifact:

- Nordic nRF52840/nRF52833 connected-state and PIP images;
- DA14695 PIP/HRS image;
- EFR32MG24 RAILtest advertising image;
- CC2340R5 SimpleLink advertising image;
- STM32L476RG/SX1280 direct-tail image files.

Use `FIRMWARE_GUIDE.md` to select an image for a board and to identify its
runtime role. Build directories, SDK copies, serial logs, and captures are not
part of this release directory.

The SX1280 ELF retains its symbols with machine-specific debug information
removed. Its loadable contents match the supplied BIN image.
