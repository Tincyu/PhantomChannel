# Source and dependency boundary

`receiver/` is project-owned PhantomChannel source, not an unrelated third-party
SDK or a renamed copy of another project. Its
`SOURCE_MANIFEST.sha256` records the integrity of the published receiver and
parser implementation.

The Zephyr controller patch is based on `nrfconnect/sdk-zephyr` tag
`v4.0.99-ncs1-1`. The `firmware/nrf52840dk/phantomchannel_peripheral/`
sample retains `SPDX-License-Identifier: Apache-2.0` headers.
The legacy `nonzephyr/` copy is not in this public repository because it
contains Nordic SDK-derived configuration with separate redistribution terms.
It remains in the local review area until separately cleared. File-level
terms are not replaced by the root MIT license.

NCS/Zephyr, nRF5 SDK, Nordic tooling, UHD, Silicon Labs and TI SDK/tooling,
Python packages, and system libraries are external dependencies. Their source
trees and binary installers are not included in this repository. Follow each
upstream project's own licensing and installation terms. No claim of a clean
upstream Zephyr or controller checkout is made by this source repository.
