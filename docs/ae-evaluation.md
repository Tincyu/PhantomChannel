# AE evaluation

The public offline profile is `python -m pytest -q tests` from a frozen clean
checkout. It requires no USB, SDR, CUDA or network access during execution.
An evaluator can reproduce it from a fresh clone with the commands in
[setup](setup.md).

For SSH evaluation on the project workstation, the separately deployed gateway
in the local `AE/` directory accepts only `run offline`, `status <run-id>`,
`fetch <run-id>` and `help` through a dedicated non-privileged SSH account.
It records the exact Git revision, exit status, timing and test log for each
run. The gateway must point to a clean, frozen checkout; it does not execute
arbitrary commands, accept client-provided paths or access hardware. The
project operator provides the SSH address and authorized key out of band.

Controller smoke, legal-peer reception and RF capture remain separate hardware
gates. They require a reserved board, exact device verification, logs, hashes
and explicit operator approval; they are not enabled by the offline gateway.
