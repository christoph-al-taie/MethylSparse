"""Turn a raw GEO beta matrix + series matrix into a compact training file.

Pass 1 streams the (multi-GB) text matrix once, drops non-cg / excluded
probes, writes float32 chunks to a scratch dir and collects per-probe stats.
Pass 2 picks the top-K most variable probes (or a given probe list) and
assembles a samples x K matrix. Output: one .npz with X, y, ids, probes.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .beta_io import iter_chunks, map_samples, open_beta_file
from .geo import parse_series_matrix, read_probe_list


def prepare(
    beta_paths: list[str],
    series_paths: list[str],
    out: str,
    n_top: int = 20000,
    label_col: str = "methylation class",
    exclude_lists: list[str] | None = None,
    probes_from: str | None = None,
    max_nan_frac: float = 0.05,
    assume_order: bool = False,
    chunksize: int = 20000,
    keep_unlabeled: bool = False,
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    anno = pd.concat([parse_series_matrix(p) for p in series_paths])
    anno = anno[~anno.index.duplicated()]
    if label_col not in anno.columns:
        raise KeyError(f"label column '{label_col}' not in annotation. Available: {list(anno.columns)}")

    files = [open_beta_file(p) for p in beta_paths]
    names = [n for bf in files for n in bf.sample_names]
    gsm = map_samples(names, anno, assume_order=assume_order)
    print(f"[data] {len(gsm)} samples in beta matrix, {len(anno)} in annotation")

    exclude = read_probe_list(exclude_lists) if exclude_lists else set()
    wanted = None
    if probes_from:
        ref = np.load(probes_from, allow_pickle=True)
        wanted = list(ref["probes"])
        ref_median = dict(zip(wanted, ref["probe_median"]))
        print(f"[data] extracting the {len(wanted)} probes used in {probes_from}")

    scratch = Path(tempfile.mkdtemp(prefix="methylsparse_", dir=out.parent))
    try:
        chunk_files, all_ids, var, nanf = [], [], [], []
        n_seen = 0
        wset = set(wanted) if wanted is not None else None
        for k, (ids, vals) in enumerate(iter_chunks(files, chunksize)):
            n_seen += len(ids)
            keep = np.array([i.startswith("cg") and i not in exclude for i in ids])
            if wset is not None:
                keep &= np.array([i in wset for i in ids])
            ids, vals = ids[keep], vals[keep]
            if len(ids) == 0:
                continue
            f = scratch / f"chunk_{k:05d}.npy"
            np.save(f, vals)
            chunk_files.append((f, len(all_ids)))
            all_ids.extend(ids.tolist())
            with np.errstate(invalid="ignore"):
                var.append(np.nanvar(vals, axis=1))
            nanf.append(np.isnan(vals).mean(axis=1))
            print(f"\r[pass1] {n_seen:>9,d} probes read, {len(all_ids):>9,d} kept", end="")
        print()
        all_ids = np.array(all_ids)
        var = np.concatenate(var)
        nanf = np.concatenate(nanf)

        if wanted is not None:
            pos = {p: i for i, p in enumerate(all_ids)}
            sel = [pos.get(p, -1) for p in wanted]
            n_missing = sum(s < 0 for s in sel)
            if n_missing:
                print(f"[warn] {n_missing} reference probes missing here, filled with training median")
            probes = np.array(wanted)
        else:
            ok = (nanf <= max_nan_frac) & np.isfinite(var)
            order = np.argsort(-np.where(ok, var, -np.inf))
            sel = np.sort(order[: min(n_top, ok.sum())])
            probes = all_ids[sel]
            print(f"[pass2] selected {len(sel)} most variable probes "
                  f"(var range {var[sel].min():.4f} .. {var[sel].max():.4f})")

        sel = np.asarray(sel)
        X = np.full((len(gsm), len(sel)), np.nan, dtype=np.float32)
        starts = [s for _, s in chunk_files] + [len(all_ids)]
        for j, (path, start) in enumerate(chunk_files):
            m = (sel >= start) & (sel < starts[j + 1])
            if m.any():
                X[:, m] = np.load(path)[sel[m] - start].T
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # impute remaining NaNs per probe (median over samples / reference median)
    if wanted is not None:
        med = np.array([ref_median[p] for p in probes], dtype=np.float32)
    else:
        med = np.nanmedian(X, axis=0).astype(np.float32)
    nan_r, nan_c = np.where(np.isnan(X))
    X[nan_r, nan_c] = med[nan_c]
    if nan_r.size:
        print(f"[data] imputed {nan_r.size:,d} missing values ({nan_r.size / X.size:.4%})")

    labels = anno.reindex(gsm)[label_col]
    has_label = labels.notna().to_numpy() & (labels.astype(str).str.len() > 0).to_numpy()
    if not keep_unlabeled and not has_label.all():
        print(f"[data] dropping {np.sum(~has_label)} samples without '{label_col}'")
        X, gsm, labels = X[has_label], [g for g, h in zip(gsm, has_label) if h], labels[has_label]

    meta_cols = [c for c in anno.columns if c in ("title", "material", "source_name_ch1")]
    np.savez_compressed(
        out, X=X, y=labels.astype(str).to_numpy(), samples=np.array(gsm), probes=probes,
        probe_median=med)
    anno.reindex(gsm)[[label_col] + meta_cols].to_csv(out.with_suffix(".samples.csv"))
    info = dict(n_samples=int(X.shape[0]), n_probes=int(X.shape[1]), label_col=label_col,
                n_classes=int(pd.Series(labels).nunique()), beta=[str(p) for p in beta_paths])
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"[done] {out}  X={X.shape}  classes={info['n_classes']}")
    return out
