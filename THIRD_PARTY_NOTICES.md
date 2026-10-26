# Third-party notices

This source snapshot contains PhantomChannel code together with focused
patches, board notes, and selected firmware images.  The project-owned host
code is released under the MIT License in [`LICENSE`](LICENSE).  The notices
below apply to third-party source, firmware build inputs, and runtime tools
used by the corresponding paths.

The complete vendor SDK trees are not included in this snapshot.  Components
described as external must be installed from the linked upstream source; use
the license files shipped with that version.  Copyright and SPDX
headers in included source files are retained and are not replaced by the
project MIT license.

This document is attribution and license guidance, not an additional grant of
vendor rights.  The vendor SDK license for a platform does not automatically
authorize general redistribution of every prebuilt image.  Before distributing
the Nordic, DA14695, EFR32MG24, or CC2340R5 images beyond this artifact,
verify the applicable SDK/EULA terms.  If redistribution is not permitted,
rebuild locally with the listed SDK or obtain the image directly under the
vendor's terms.

## Included source and focused patches

| Path in this snapshot | Author / copyright | Version or base | Source | License |
| --- | --- | --- | --- | --- |
| `patches/zephyr/*.patch` and `firmware/nrf52840dk/phantomchannel_peripheral/` | Zephyr Project contributors; Nordic contributors; project additions retain file-level headers | Patch base `nrfconnect/sdk-zephyr` tag `v4.0.99-ncs1-1`, commit `77f865b8f8d0cb3d19002bfe713e9dd46e6f71b7` | [nrfconnect/sdk-zephyr](https://github.com/nrfconnect/sdk-zephyr) | Apache-2.0 for the Zephyr-derived files; see the `SPDX-License-Identifier: Apache-2.0` headers and [Zephyr licensing](https://github.com/zephyrproject-rtos/zephyr/blob/main/README.license) |
| `firmware/sx1280/source/gatt_counter.c` | BlueKitchen GmbH (2014); PhantomChannel modifications | BTstack `1.6.2`, source commit `565100c1cae73dc1c5f9b239953dbdd14ae283d3` | [BlueKitchen BTstack v1.6.2](https://github.com/bluekitchen/btstack/tree/v1.6.2) | BlueKitchen BTstack license reproduced below; it includes a personal/non-commercial-use condition |
| `firmware/sx1280/source/ll_sx1280.c` | BlueKitchen GmbH (2020); PhantomChannel modifications | BTstack `1.6.2`, source commit `565100c1cae73dc1c5f9b239953dbdd14ae283d3` | [BlueKitchen BTstack v1.6.2](https://github.com/bluekitchen/btstack/tree/v1.6.2) | BlueKitchen BTstack license reproduced below; it includes a personal/non-commercial-use condition |

The original copyright blocks remain at the beginning of the two SX1280 C
files.  The compact source directory intentionally does not contain the full
BTstack repository; the notices here provide the corresponding license text
and upstream source reference for the included derivative files.

## Firmware images whose SDK source is external

The following images are distributed as selected build artifacts.  Their full
vendor SDKs and build workspaces are not part of this archive.

| Image / path | Vendor code and version used for the build | Source / license reference |
| --- | --- | --- |
| `firmware/images/da14695-blehr-pip-8b.img` | Apache Mynewt core commit `bf87c1f0599cd41f86b64dcf58ab11fa40ef78ad`; Apache Mynewt NimBLE commit `605c7585408bc3674818eeb7b6f478a8aefe9746`; mcumgr commit `06c87b5fc5ce938810d1c2196a87bdf95c0d423d`; MCUboot `2.3.0` | [apache/mynewt-core](https://github.com/apache/mynewt-core), [apache/mynewt-nimble](https://github.com/apache/mynewt-nimble), and [apache/mynewt-mcumgr](https://github.com/apache/mynewt-mcumgr). Apache-2.0 and component notices are supplied by the vendor repositories; verify the DA SDK/image redistribution terms before further distribution. |
| `firmware/images/efr32mg24-railtest-adv-tail-239b.hex` | Silicon Labs Simplicity SDK `2025.6.3` / RAILtest | [SiliconLabs/simplicity_sdk](https://github.com/SiliconLabs/simplicity_sdk). The SDK uses multiple licenses; consult the MSLA and component notices before redistributing the image. |
| `firmware/images/cc2340r5-basic-ble-dynamic-tail-239b.hex` | Texas Instruments SimpleLink Low Power F3 SDK `8.20` | [SimpleLink Low Power F3 SDK](https://www.ti.com/tool/download/SIMPLELINK-LOWPOWER-F3-SDK) and its [8.20 documentation](https://software-dl.ti.com/simplelink/esd/simplelink_lowpower_f3_sdk/8.20.00.119/exports/docs/Documentation_Overview.html). The TI license/EULA governs the SDK and any image built from it; verify redistribution rights separately. |
| `firmware/images/nrf52840-*.hex`, `firmware/images/nrf52833-*.hex` | Nordic nRF Connect SDK `3.0.0-rc1` and the focused Zephyr base identified above | [nRF Connect SDK](https://github.com/nrfconnect/sdk-nrf) and [Zephyr licensing](https://github.com/zephyrproject-rtos/zephyr/blob/main/README.license). The complete NCS/toolchain is external; verify Nordic terms before further binary distribution. |

## SX1280 external build inputs

`firmware/sx1280/source/Makefile` references the following source trees.  They
are required only when rebuilding the SX1280 image; they are not copied into
this snapshot.

| Component | Author | Version / revision | Source | License |
| --- | --- | --- | --- | --- |
| BTstack core and embedded platform | BlueKitchen GmbH and contributors | `1.6.2` / `565100c1cae73dc1c5f9b239953dbdd14ae283d3` | [bluekitchen/btstack](https://github.com/bluekitchen/btstack/tree/v1.6.2) | BlueKitchen BTstack license below; non-commercial-use condition applies |
| SX1280 C driver | Semtech S.A. | BTstack `v1.6.2` driver tree | [BTstack SX128x driver](https://github.com/bluekitchen/btstack/tree/v1.6.2/chipset/sx128x/SMTC_Drivers/sx1280-driver-c) | Revised BSD License (3-Clause) |
| micro-ecc (`uECC.c`) | Kenneth MacKay | BTstack third-party revision `e4d264b5` | [kmackay/micro-ecc](https://github.com/kmackay/micro-ecc) | BSD-2-Clause |
| Rijndael (`rijndael.c`) | Original Rijndael authors | BTstack revision `20020903` | [BTstack Rijndael directory](https://github.com/bluekitchen/btstack/tree/v1.6.2/3rd-party/rijndael) | Public domain notice in the upstream source |
| SEGGER RTT (`SEGGER_RTT*.c`) | SEGGER Microcontroller GmbH | BTstack revision `V786a` | [SEGGER RTT](https://www.segger.com/products/debug-probes/j-link/technology/about-real-time-transfer/) | SEGGER redistribution terms in the upstream `License.txt` |
| STM32L4 HAL | STMicroelectronics | STM32CubeL4 HAL used by the external port | [STMicroelectronics STM32CubeL4](https://github.com/STMicroelectronics/STM32CubeL4) | BSD-3-Clause for the HAL; component notices are included in the STM32Cube package |
| CMSIS Cortex-M4 headers | Arm Limited | CMSIS headers used by the external STM32 port | [ARM-software/CMSIS_5](https://github.com/ARM-software/CMSIS_5) | Apache-2.0 |

The SX1280 firmware image is therefore not relicensed as MIT merely because
the surrounding PhantomChannel project is MIT-licensed.  The applicable
upstream notices remain those of each component above.

## Host-side runtime dependencies

These packages are installed by the requirements files rather than shipped as
source in this archive.

| Requirements file | Direct packages and versions | License/source |
| --- | --- | --- |
| `requirements-test.txt` | `numpy==2.4.6`, `scipy==1.17.1`, `PyYAML==6.0.3`, `pytest==8.4.2` | NumPy and SciPy use BSD-style licenses; PyYAML and pytest use MIT. Use the package license files for the exact wheel selected by the evaluator: [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), [PyYAML](https://github.com/yaml/pyyaml), [pytest](https://github.com/pytest-dev/pytest). |
| `requirements-cuda.txt` and `receiver/requirements-cuda.txt` | CuPy, `pybind11==3.0.1`, CUDA Toolkit, NVIDIA CUDA runtime libraries, NumPy/SciPy, Matplotlib and their pinned versions listed in those files | CuPy and the Python libraries retain their upstream licenses; pybind11 is BSD-3-Clause; CUDA Toolkit and NVIDIA wheels are distributed under NVIDIA license terms included with the selected packages. |
| Separate experiment reproduction archive (`requirements.txt`) | `numpy>=1.26`, `scipy>=1.11`, `matplotlib>=3.8` | Install from PyPI and retain each package's license metadata. These ranges intentionally do not identify a single transitive dependency set. |

## SDR and programming tools

UHD, nrfutil, J-Link, Silicon Labs Commander, TI UniFlash, Apache Newt, the
Nordic Sniffer package, and the ARM/TI compilers are external tools.  They are
not distributed in this source archive.  The PhantomChannel receiver uses
UHD through its Python/C++ interfaces; UHD is offered by Ettus under GPLv3
and an alternative license ([Ettus licensing](https://www.ettus.com/sdr-software/licenses/)).

## License texts for included derivative source

### BlueKitchen BTstack license

Copyright (C) 2009 BlueKitchen GmbH. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holders nor the names of contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
4. Any redistribution, use, or modification is done solely for personal
   benefit and not for any commercial purpose or for monetary gain.

THIS SOFTWARE IS PROVIDED BY BLUEKITCHEN GMBH AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL BLUEKITCHEN GMBH OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please inquire about commercial licensing options at
contact@bluekitchen-gmbh.com.

### Other included-source notices

The Semtech SX1280 driver, micro-ecc, SEGGER RTT, STM32 HAL, and CMSIS license
terms remain those of their upstream distributions listed above.  Their source
trees are external build inputs for this compact snapshot; do not remove or
replace their original headers when assembling a full rebuild workspace.
