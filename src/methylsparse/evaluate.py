"""Cross-validated evaluation of classifiers across simulated sparsity levels."""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from .models import make_model
from .sparsity import simulate

FIELDS = ["split", "fold", "model", "keep_frac", "binarize", "repeat", "n_test",
          "balanced_accuracy", "accuracy", "macro_f1", "ece", "mean_confidence", "fit_seconds"]


def load_dataset(path, classes_regex=None, min_class_size=10):
    d = np.load(path, allow_pickle=True)
    X, y = d["X"].astype(np.float32), d["y"].astype(str)
    keep = np.ones(len(y), bool)
    if classes_regex:
        keep &= np.array([bool(re.search(classes_regex, c)) for c in y])
    labels, counts = np.unique(y[keep], return_counts=True)
    big = set(labels[counts >= min_class_size])
    keep &= np.isin(y, list(big))
    print(f"[data] {keep.sum()} samples, {len(big)} classes "
          f"(regex={classes_regex!r}, min_class_size={min_class_size}), {X.shape[1]} CpGs")
    return X[keep], y[keep], d["samples"][keep], d["probes"]


def ece_score(proba, y_true_idx, n_bins=15):
    conf = proba.max(1)
    correct = proba.argmax(1) == y_true_idx
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def score(model, Xte, yte):
    proba = model.predict_proba(Xte)
    classes = np.asarray(model.classes_)
    pred = classes[proba.argmax(1)]
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(c, -1) for c in yte])
    return dict(
        balanced_accuracy=balanced_accuracy_score(yte, pred),
        accuracy=accuracy_score(yte, pred),
        macro_f1=f1_score(yte, pred, average="macro", zero_division=0),
        ece=ece_score(proba, yi),
        mean_confidence=float(proba.max(1).mean()),
    )


def parse_levels(spec: str):
    """'1,0.05b,0.01b' -> [(1.0, False), (0.05, True), (0.01, True)]
    A trailing 'b' means binarised reads; 'c' or nothing means continuous."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        b = tok.endswith("b")
        out.append((float(tok.rstrip("bc")), b))
    return out


def _eval_levels(writer, fh, base, model, Xte, yte, levels, repeats, rng):
    for keep, binz in levels:
        reps = 1 if (keep >= 1 and not binz) else repeats
        for r in range(reps):
            Xs = simulate(Xte, keep, binz, rng)
            row = dict(base, keep_frac=keep, binarize=int(binz), repeat=r, n_test=len(yte),
                       **score(model, Xs, yte))
            writer.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            print(f"      {base['model']:<7s}keep={keep:<7g} {'bin ' if binz else 'cont'} rep={r}  "
                  f"bal_acc={row['balanced_accuracy']:.3f}  f1={row['macro_f1']:.3f}  "
                  f"ece={row['ece']:.3f}")


def run(data, outdir, models=("rf", "logreg", "mlp"), levels="1,0.2b,0.05b,0.01b,0.005b,0.002b",
        folds=5, repeats=3, classes_regex=None, min_class_size=10, external=None,
        quick=False, seed=0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    X, y, _, probes = load_dataset(data, classes_regex, min_class_size=max(min_class_size, folds))
    lv = parse_levels(levels)
    (outdir / "config.json").write_text(json.dumps(dict(
        data=str(data), models=list(models), levels=levels, folds=folds, repeats=repeats,
        classes_regex=classes_regex, min_class_size=min_class_size, external=external,
        quick=quick, seed=seed, n_samples=int(len(y)), n_classes=int(len(set(y))),
        n_probes=int(X.shape[1])), indent=2))

    res = outdir / "results.csv"
    with open(res, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(X, y)):
            print(f"[cv  ] fold {k + 1}/{folds}  train={len(tr)} test={len(te)}")
            for name in models:
                t0 = time.time()
                m = make_model(name, quick=quick, seed=seed + k).fit(X[tr], y[tr])
                dt = time.time() - t0
                print(f"    {name}: fitted in {dt:.1f}s")
                rng = np.random.default_rng(seed * 1000 + k)  # same masks for every model
                _eval_levels(w, fh, dict(split="cv", fold=k, model=name, fit_seconds=round(dt, 1)),
                             m, X[te], y[te], lv, repeats, rng)

        if external:
            d = np.load(external, allow_pickle=True)
            if not np.array_equal(d["probes"], probes):
                raise ValueError("external set has different probes; build it with "
                                 "`prepare --probes-from <training npz>`")
            Xe, ye = d["X"].astype(np.float32), d["y"].astype(str)
            known = np.isin(ye, np.unique(y))
            print(f"[ext ] {known.sum()}/{len(ye)} external samples have a class seen in training")
            Xe, ye = Xe[known], ye[known]
            for name in models:
                t0 = time.time()
                m = make_model(name, quick=quick, seed=seed).fit(X, y)
                dt = time.time() - t0
                rng = np.random.default_rng(seed * 1000 + 999)
                _eval_levels(w, fh, dict(split="external", fold=-1, model=name, fit_seconds=round(dt, 1)),
                             m, Xe, ye, lv, repeats, rng)
    print(f"[done] {res}")
    return res
