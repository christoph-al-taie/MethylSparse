"""Simulate low-coverage (e.g. nanopore) measurements from array beta values.

A low-pass nanopore run observes only a random subset of CpGs, typically with a
single read each, so the value at an observed CpG is 0 or 1 rather than a
fraction. We mimic this by
  1. keeping each CpG independently with probability `keep_frac`
     (missing -> NaN), and
  2. optionally replacing each kept beta value by a Bernoulli(beta) draw
     ("one read per CpG").
"""
from __future__ import annotations

import numpy as np


def simulate(X: np.ndarray, keep_frac: float, binarize: bool,
             rng: np.random.Generator, row_block: int = 512) -> np.ndarray:
    """Return a float32 copy of X with unobserved CpGs set to NaN."""
    X = np.asarray(X, dtype=np.float32)
    out = np.empty_like(X)
    for s in range(0, X.shape[0], row_block):
        blk = X[s:s + row_block].copy()
        if binarize:
            blk = (rng.random(blk.shape, dtype=np.float32) < blk).astype(np.float32)
        if keep_frac < 1.0:
            miss = rng.random(blk.shape, dtype=np.float32) >= keep_frac
            # guarantee at least one observed CpG per sample
            rows = np.flatnonzero(miss.all(axis=1))
            if rows.size:
                miss[rows, rng.integers(0, blk.shape[1], rows.size)] = False
            blk[miss] = np.nan
        out[s:s + row_block] = blk
    return out


def describe(keep_frac: float, binarize: bool, n_probes: int) -> str:
    kind = "binary reads" if binarize else "continuous beta"
    return f"{keep_frac:.3%} of {n_probes} CpGs (~{keep_frac * n_probes:.0f}), {kind}"
