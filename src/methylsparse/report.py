"""Markdown summary table from results.csv."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ORDER = ["logreg", "mlp", "rf"]
LABEL = {"logreg": "Logistic regression", "mlp": "Masked MLP", "rf": "Random forest"}


def summary(results, split="cv") -> str:
    df = pd.read_csv(results)
    df = df[df["split"] == split]
    n_probes = None
    cfg = Path(results).with_name("config.json")
    if cfg.exists():
        n_probes = json.loads(cfg.read_text()).get("n_probes")

    g = (df.groupby(["model", "keep_frac"])
           .agg(bal=("balanced_accuracy", "mean"), sd=("balanced_accuracy", "std"),
                f1=("macro_f1", "mean"), ece=("ece", "mean"), conf=("mean_confidence", "mean"))
           .reset_index())
    levels = sorted(g["keep_frac"].unique(), reverse=True)

    def lvl(k):
        if k >= 1:
            return "100% (array)"
        s = f"{k:.1%}".replace(".0%", "%")
        return f"{s} (~{k * n_probes:,.0f})" if n_probes else s

    lines = ["| CpGs observed | " + " | ".join(LABEL.get(m, m) for m in ORDER if m in set(g.model)) + " |",
             "|---|" + "---:|" * len([m for m in ORDER if m in set(g.model)])]
    for k in levels:
        cells = []
        for m in ORDER:
            r = g[(g.model == m) & (g.keep_frac == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells.append(f"{r.bal:.3f} ± {r.sd:.3f}")
        lines.append(f"| {lvl(k)} | " + " | ".join(cells) + " |")
    bal = "\n".join(lines)

    lines = ["| CpGs observed | Model | Balanced acc. | Macro F1 | Mean confidence | ECE |",
             "|---|---|---:|---:|---:|---:|"]
    for k in levels:
        for m in ORDER:
            r = g[(g.model == m) & (g.keep_frac == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"| {lvl(k)} | {LABEL.get(m, m)} | {r.bal:.3f} | {r.f1:.3f} | "
                         f"{r.conf:.3f} | {r.ece:.3f} |")
    detail = "\n".join(lines)
    return f"Balanced accuracy, mean ± SD over folds and repeats:\n\n{bal}\n\nDetail:\n\n{detail}\n"
