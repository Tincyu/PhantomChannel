# Artifact appendix

`phantomchannel-appendix.pdf` is the submission-oriented appendix, with editable
source in `phantomchannel-appendix.tex`. It follows the CCS artifact template
and focuses on access, environment setup, execution and output inspection.
It does not repeat experimental result values or fixed test/sample counts.

Build with an installation of Tectonic, or a LaTeX environment containing
`acmart`, `paralist`, `xurl` and the ACM class's font dependencies:

```bash
tectonic appendix/phantomchannel-appendix.tex --outdir appendix
```

See [`../reproduction/ENVIRONMENT.md`](../reproduction/ENVIRONMENT.md) for the
verified analysis environment and pinned dependencies. The archived experiment
ZIP fixes replay inputs and scripts. Repository supplements should be recorded
by commit when used with that archive.

After running the archived experiment replay, the optional antenna helper can
be invoked with the extracted experiment root as its positional argument:

```bash
python3 appendix/verify_antenna_correctness.py /path/to/extracted-experiment
```

The helper reads generated candidate records without modifying them. Its output
separates candidate recovery from exact-pattern and integrity-accepted useful
payload delivery.
