# AE offline SSH gateway

`gateway.py` is the versioned, forced-command entry point for evaluation on a
project workstation. `config.example.json` documents its four required settings.
The gateway accepts only `help`, `run offline`, `status <run-id>` and
`fetch <run-id>`; it does not provide a shell, hardware access or package
installation. Its unit tests are in `tests/test_ae_gateway.py` and run with the
ordinary offline suite.

The gateway program may be invoked from any directory. Its `--config` argument
must point to an operator-owned JSON file **outside this repository**. That file
sets absolute paths to a separate, clean checkout at a frozen commit, a Python
test environment, and a writable results directory. The AE account must not be
able to modify the checkout or configuration. Do not commit SSH keys, real
configuration, result logs or captures.

An example forced command is:

```text
restrict,command="/usr/bin/python3 /absolute/path/to/gateway.py --config /absolute/path/to/config.json" ssh-ed25519 ...
```

This is a deployment template, not a claim that workstation SSH isolation is
active. Configure and test the dedicated account, key restrictions, filesystem
permissions and network rules before giving access. See
[`docs/ae-evaluation.md`](../docs/ae-evaluation.md).
