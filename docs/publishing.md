# Safe GitHub publishing

This repository publishes project source, focused SDK patches, configuration
examples, tests and small result summaries. It must not publish complete SDKs,
toolchains, generated build trees, firmware binaries, raw captures or local
evaluation state.

## Publication model

Keep vendor SDKs outside this repository. Preserve an embedding implementation
with all of the following instead:

1. the exact upstream repository, release and commit (or vendor SDK version and
   original-file hashes);
2. a focused patch containing only the project changes;
3. project-owned applications, overlays and build configuration;
4. a script or documented command that checks the base before applying the
   patch; and
5. small, sanitized test summaries and hashes, while raw IQ and full logs stay
   in the outer local `test/` area.

Do not use Git LFS as a way to publish an SDK. LFS changes storage mechanics;
it does not resolve redistribution rights, reviewability or reproducibility.

## Never stage these paths

- SDK and toolchain directories, including NCS, Zephyr checkouts, Mynewt build
  repositories, Simplicity SDK and SimpleLink SDK;
- `build/`, `artifacts/`, `.west/`, `.venv/`, generated `autogen/` trees and
  compiler output;
- `.hex`, `.bin`, `.elf`, `.out`, `.map`, `.obj`, archives and installers;
- raw `.sc16`, IQ, PCAP, complete UART/RTT logs and hardware result directories;
- real board serial numbers, private paths, credentials, SSH keys and AE config.

Do not run `git add -A`, `git add .`, `git add -f`, `git push --all` or
`git push --mirror` as part of the publication workflow. Stage an explicit
allowlist of reviewed paths.

## Safe staging and commit workflow

Run all commands from the `gitpublic/` checkout, not from its outer workspace.
First verify the repository identity and inspect the worktree:

```bash
git rev-parse --show-toplevel
git remote -v
git status --short --branch
```

Stage only the reviewed source, patch, documentation, test or configuration
paths. For example:

```bash
git add -- patches/firmware-change.patch
git add -- firmware/project-owned-sample/
git add -- configs/example.yaml docs/embedding-strategy.md
```

Before committing, inspect both the path list and the size summary:

```bash
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```

If an SDK root, build directory, binary, capture or unexpected generated file
appears, unstage that exact path with `git restore --staged -- <path>` and fix
the ignore rule. Do not delete the local SDK merely to unstage it.

Run the gates, then create a local checkpoint:

```bash
python -m pytest -q tests
python scripts/check_public.py
git commit -m "Describe the reviewed source change"
```

`check_public.py` rejects individual files larger than 8 MiB, more than 5,000
Git-visible files, or more than 64 MiB of total Git-visible content. These are
deliberately conservative project limits, not GitHub service limits.

## Check unpushed history

Deleting a mistakenly committed SDK from the latest worktree does not remove
it from earlier unpushed commits. Before every push, check all objects that the
remote does not yet have:

```bash
git fetch origin
python scripts/check_push_size.py --base origin/main
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

The outbound check rejects a blob larger than 8 MiB, more than 5,000 new blobs,
or more than 64 MiB of unique outbound blob content. If it fails, do not push.
Rewrite only the unpushed local commits after reviewing the exact offending
paths; adding the path to `.gitignore` or deleting it in a later commit is not
enough.

## Push a review branch first

Use a named review branch for a multi-platform or SDK-patch release:

```bash
git switch -c release/multiplatform-embedding
git push --dry-run -u origin release/multiplatform-embedding
git push -u origin release/multiplatform-embedding
```

Review the GitHub diff before merging it into `main`. A dry run checks the ref
operation but does not replace the content and history checks above.

## Current SDK strategy

The Nordic implementation is represented by the focused patch under
`patches/zephyr/`; the complete Zephyr/NCS tree remains external. DA14695,
EFR32MG24 and CC2340R5 should follow the same patch-and-overlay model before
their SDK-resident embedding changes are declared published. Never copy their
entire validated workstation SDK or build workspace into this repository.
