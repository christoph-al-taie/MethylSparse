"""Streaming reader for large probe x sample beta-value text matrices.

Handles the typical GEO 'processed data' layout

    ID_REF  <sample 1>  Detection Pval  <sample 2>  Detection Pval  ...
    cg0000  0.83        0.00            0.12        0.00

as well as plain matrices (probe id + one column per sample), tab or comma
separated, optionally gzipped. Several files with identical probe order (e.g.
batches written by scripts/preprocess_idat.R) can be streamed side by side.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

PVAL_PATTERN = re.compile(r"(detection|p[\s_.-]?val)", re.IGNORECASE)


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


@dataclass
class BetaFile:
    path: Path
    sep: str
    header: list[str]
    beta_idx: list[int]   # column positions holding beta values

    @property
    def sample_names(self) -> list[str]:
        return [self.header[i] for i in self.beta_idx]


def open_beta_file(path: Path | str) -> BetaFile:
    path = Path(path)
    with _open_text(path) as fh:
        first = fh.readline().rstrip("\n").rstrip("\r")
    sep = "\t" if first.count("\t") >= first.count(",") else ","
    header = [h.strip().strip('"') for h in first.split(sep)]
    beta_idx = [i for i, h in enumerate(header) if i > 0 and not PVAL_PATTERN.search(h)]
    if not beta_idx:
        raise ValueError(f"{path}: no beta columns detected in header")
    return BetaFile(path, sep, header, beta_idx)


SAMPLE_NO = re.compile(r"\bsample[\s_#-]*0*(\d+)\b", re.IGNORECASE)


def match_by_sample_number(names: list[str], anno: pd.DataFrame) -> list[str] | None:
    """Match 'SAMPLE <n>' matrix columns to the annotation column that carries
    'sample <n>' in its text (unique numbers on both sides). None if no fit."""
    nums = [SAMPLE_NO.search(n) for n in names]
    if not all(nums):
        return None
    nums = [int(m.group(1)) for m in nums]
    if len(set(nums)) != len(nums):
        return None
    for col in anno.columns:
        found = anno[col].astype(str).map(lambda s: SAMPLE_NO.search(s))
        if not found.map(bool).all():
            continue
        col_nums = found.map(lambda m: int(m.group(1)))
        if col_nums.duplicated().any():
            continue
        lookup = dict(zip(col_nums, anno.index))
        if all(n in lookup for n in nums):
            ids = [lookup[n] for n in nums]
            ex = ", ".join(f"'{names[i]}' -> {ids[i]} ('{anno.at[ids[i], col]}')"
                           for i in range(min(2, len(names))))
            print(f"[map ] matched {len(names)} matrix columns by sample number via "
                  f"annotation column '{col}', e.g. {ex}")
            return ids
    return None


def map_samples(names: list[str], anno: pd.DataFrame, assume_order: bool = False) -> list[str]:
    """Map matrix column names to GSM accessions in `anno` (index = GSM).

    Tries, in this order: names already are GSM ids; names match one of the
    annotation's unique text columns (title, description, source name ...).
    Raises with a diagnostic message if nothing fits.
    """
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()  # noqa: E731
    nset = [norm(n) for n in names]

    gsm_like = [re.search(r"GSM\d+", n) for n in names]
    if all(gsm_like):
        ids = [m.group(0) for m in gsm_like]
        missing = [i for i in ids if i not in anno.index]
        if not missing:
            return ids

    best_col, best_hits = None, -1
    for col in anno.columns:
        vals = anno[col].astype(str).map(norm)
        if vals.duplicated().any():
            continue
        hits = sum(n in set(vals) for n in nset)
        if hits > best_hits:
            best_col, best_hits = col, hits
    if best_col is not None and best_hits == len(names):
        lookup = dict(zip(anno[best_col].astype(str).map(norm), anno.index))
        print(f"[map ] matched {len(names)} matrix columns via annotation column '{best_col}'")
        return [lookup[n] for n in nset]

    # e.g. matrix 'SAMPLE 17' <-> title 'MB, G4, sample 17 [reference set]' (GSE90496)
    ids = match_by_sample_number(names, anno)
    if ids is not None:
        return ids

    if assume_order:
        if len(names) != len(anno):
            raise ValueError(
                f"--assume-order: {len(names)} matrix columns but {len(anno)} annotated samples")
        print("[map ] WARNING: mapping columns to samples by ORDER (--assume-order). "
              "Check a few samples by hand!")
        return list(anno.index)

    examples = "\n".join(f"    {n}" for n in names[:5])
    cand = "\n".join(
        f"    {c}: {list(anno[c].astype(str).head(3))}"
        for c in anno.columns[:12])
    raise ValueError(
        "Could not map matrix columns to GSM ids.\n"
        f"  First matrix columns:\n{examples}\n"
        f"  Annotation columns (first values):\n{cand}\n"
        f"  Best partial match: column '{best_col}' with {best_hits}/{len(names)} hits.\n"
        "  Options: run `methylsparse inspect`, or pass --assume-order if the matrix "
        "columns are in the same order as the series matrix samples.")


def iter_chunks(files: list[BetaFile], chunksize: int = 20000
                ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (probe_ids, values[probes, samples]) chunks, columns concatenated
    across files in the given order. Probe order must match across files."""
    readers = []
    for bf in files:
        usecols = [0] + bf.beta_idx
        readers.append(pd.read_csv(
            bf.path, sep=bf.sep, usecols=usecols, index_col=0, chunksize=chunksize,
            na_values=["", "NA", "NaN", "nan", "null"], engine="c", low_memory=False))
    for parts in zip(*readers):
        ids = parts[0].index.astype(str).to_numpy()
        for p in parts[1:]:
            if len(p) != len(ids) or not np.array_equal(p.index.astype(str).to_numpy(), ids):
                raise ValueError("Probe order differs between beta files, cannot stream side by side")
        vals = np.hstack([p.to_numpy(dtype=np.float32, copy=False) for p in parts])
        yield ids, vals
