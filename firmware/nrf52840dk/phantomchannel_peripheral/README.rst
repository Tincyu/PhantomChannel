PhantomChannel Peripheral
#########################

This sample is a controlled BLE peripheral for PhantomChannel range and
bandwidth experiments on ``nrf52840dk/nrf52840``.

After a central connects and enables notifications, the sample periodically
sends a normal GATT notification whose value contains a ``0xaa 0xaa 0x00``
marker followed by a deterministic PhantomChannel frame. The modified Nordic
LLL connection TX path rewrites that marker into the normal BLE CRC and places
the following bytes after the CRC.

The main experiment knobs are Kconfig options:

* ``CONFIG_PHANTOMCHANNEL_COVERT_LEN``
* ``CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS``
* ``CONFIG_PHANTOMCHANNEL_STATIC_ADDRESS``

Build example:

.. code-block:: console

   west build -b nrf52840dk/nrf52840 samples/bluetooth/phantomchannel_peripheral

The firmware prints one machine-readable ``PHANTOM_TX`` JSON record per
notification attempt. ``covert_hex`` is the payload bytes used for recovery
statistics. ``frame_hex`` is the complete post-CRC PhantomChannel frame.

