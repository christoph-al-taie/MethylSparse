"""Core figure: performance vs. fraction of observed CpGs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# categorical slots 1-3 (validated colour-blind safe as a set)
COLORS = {"mlp": "#2a78d6", "rf": "#eb6834", "logreg": "#1baf7a"}
NAMES = {"mlp": "Masked MLP (sparsity-trained)", "rf": "Random forest (array-trained, mean-imputed)", "logreg": "Logistic regression"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b68", "#e6e6e3", "#fcfcfb"


def plot(results, out=None, metric="balanced_accuracy", split="cv"):
    df = pd.read_csv(results)
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"no rows with split == {split!r} in {results}")
    agg = (df.groupby(["model", "keep_frac", "binarize"])[metric]
             .agg(["mean", "std"]).reset_index().fillna(0))
    n_probes = None
    cfg = Path(results).with_name("config.json")
    if cfg.exists():
        import json
        n_probes = json.loads(cfg.read_text()).get("n_probes")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    order = [m for m in COLORS if m in set(agg["model"])] + \
            sorted(set(agg["model"]) - set(COLORS))
    for model in order:
        g = agg[agg["model"] == model]
        g = g.sort_values("keep_frac")
        c = COLORS.get(model, INK)
        ax.fill_between(g["keep_frac"], g["mean"] - g["std"], g["mean"] + g["std"],
                        color=c, alpha=0.15, linewidth=0)
        ax.plot(g["keep_frac"], g["mean"], color=c, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES.get(model, model))

    ax.set_xscale("log")
    xt = sorted(agg["keep_frac"].unique())
    ax.set_xticks(xt)
    def tick(x):
        if x >= 1:
            return "100%\n(array)"
        s = f"{x:.1%}".replace(".0%", "%")
        return f"{s}\n~{x * n_probes:,.0f}" if n_probes else s
    ax.set_xticklabels([tick(x) for x in xt])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("CpGs observed per sample"
                  + (f" (fraction and number of {n_probes:,} selected CpGs)" if n_probes else ""),
                  color=INK)
    ax.set_ylabel(metric.replace("_", " ").capitalize(), color=INK)
    ax.set_title("Classifier robustness to nanopore-like sparsity", loc="left", color=INK,
                 fontsize=11, fontweight="bold", pad=20)
    ax.text(0, 1.015, "Below 100%: random CpG subset, one simulated read per CpG (0/1). "
            "Mean ± SD over folds and repeats.", transform=ax.transAxes, fontsize=7.5, color=MUTED,
            va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(0.74, 0.20), labelcolor=INK)
    leg.set_zorder(5)
    fig.tight_layout()
    out = Path(out) if out else Path(results).with_name(f"{metric}_vs_sparsity_{split}.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] {out}")
    return out
