# PhantomChannel nRF52840DK peripheral sample

This is the small application sample copied from the local NCS workspace. It
is bundled so the new workstation has the exact application source, while the
full NCS workspace and Zephyr toolchain remain external dependencies.

Expected build context:

- nRF Connect SDK `v3.0.0-rc1`
- Zephyr `v4.0.99-ncs1-rc1`
- nRF Connect toolchain bundle `7cbc0036f4`
- board: `nrf52840dk/nrf52840`

The sample uses the PhantomChannel GATT notification characteristic and RTT
logging expected by the host receiver. Review the address and serial number in
the portable YAML before flashing a board.

