#!/usr/bin/env python3
"""Convert an X310 20 MS/s SC16 ch39 capture into per-packet complex64 .npy
files at 4 MS/s for the PhantomChannel_nRF_PoC phantom_rx_cli.py receiver.

The X310 IQ (SC16 int16, 20 MS/s, centered on 2480 MHz) is decimated to
4 MS/s, threshold-segmented into bursts, and each burst is written as
``iq_packet_*.npy`` (np.complex64) with the pre-samples padding the PoC
receiver expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import decimate


SR_IN = 20_000_000
SR_OUT = 4_000_000
DECIM = int(SR_IN / SR_OUT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq", type=Path, required=True, help="capture.sc16 (20 MS/s SC16)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="absolute magnitude threshold; auto if omitted")
    parser.add_argument("--min-len", type=int, default=400)
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--pre-samples", type=int, default=80)
    parser.add_argument("--merge-gap", type=int, default=3000, help="merge gaps below this many 4 MS/s samples")
    args = parser.parse_args(argv)

    iq = args.iq.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.memmap(iq, dtype=np.int16, mode="r")
    total = int(raw.size / 2)
    print(f"input complex samples: {total} ({total / SR_IN:.1f} s @ {SR_IN/1e6:.0f} MS/s)")

    # Decimate chunk by chunk (avoid holding the whole 1.6 GB complex64).
    dec_chunks: list[np.ndarray] = []
    chunk = 20_000_000
    for off in range(0, total, chunk):
        end = min(off + chunk, total)
        seg = raw[off * 2:end * 2].reshape(-1, 2).astype(np.float32)
        z = (seg[:, 0] + 1j * seg[:, 1]) / 32768.0
        dec_chunks.append(decimate(z, DECIM, ftype="fir", zero_phase=False))
    del raw
    dec = np.concatenate(dec_chunks)
    print(f"decimated: {dec.size} samples @ {SR_OUT/1e6:.0f} MS/s ({dec.size / SR_OUT:.1f} s)")

    mag = np.abs(dec)
    if args.threshold is None:
        sample = mag[:1_000_000]
        med = float(np.median(sample))
        mad = float(np.median(np.abs(sample - med)))
        thr = med + 8.0 * 1.4826 * mad
    else:
        thr = float(args.threshold)
    print(f"threshold: {thr:.6f}")

    # Threshold segmentation with gap merging.
    above = mag > thr
    idx = np.flatnonzero(above)
    if idx.size == 0:
        print("no samples above threshold")
        return 1
    splits = np.flatnonzero(np.diff(idx) > args.merge_gap)
    starts = np.concatenate([[idx[0]], idx[splits + 1]])
    ends = np.concatenate([idx[splits], [idx[-1]]])

    count = 0
    for s, e in zip(starts, ends):
        dur = int(e) - int(s) + 1
        if dur < args.min_len or dur > args.max_len:
            continue
        lo = max(0, int(s) - args.pre_samples)
        hi = min(dec.size, int(e) + 1)
        packet = dec[lo:hi].astype(np.complex64)
        np.save(out_dir / f"iq_packet_{count:05d}.npy", packet)
        count += 1
    print(f"wrote {count} packets to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
