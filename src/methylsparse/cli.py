"""Command line interface: `methylsparse <command> ...` or `python -m methylsparse ...`."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def cmd_download(a):
    from . import geo
    out = Path(a.out or f"data/raw/{a.gse}")
    geo.download_series_matrix(a.gse, out)
    sdir = geo.series_dir(a.gse) + "/suppl"
    try:
        names = geo.list_dir(sdir)
    except Exception as e:  # noqa: BLE001
        names = []
        print(f"[warn] could not list {sdir}: {e}")
    if names:
        print(f"[info] supplementary files of {a.gse}:")
        for n in names:
            print(f"         {n}")
    picks = []
    if a.suppl == ["auto"]:
        picks = [n for n in names if re.search(r"beta|processed|normali", n, re.I)
                 and not n.lower().endswith(".tar")]
        if not picks:
            print("[warn] no processed beta file found automatically; pick one with --suppl NAME "
                  "or use the IDAT route (scripts/preprocess_idat.R)")
    elif a.suppl:
        picks = a.suppl
    for n in picks:
        geo.download(f"{sdir}/{n}", out / n)
    if a.filters:
        geo.download_mnp_filters(Path("data/filters"))


def cmd_inspect(a):
    import pandas as pd

    from .beta_io import map_samples, open_beta_file
    from .geo import parse_series_matrix
    anno = pd.concat([parse_series_matrix(p) for p in a.series])
    print(f"\n== annotation: {len(anno)} samples, columns ==")
    for c in anno.columns:
        vals = anno[c].dropna().astype(str)
        print(f"  {c[:40]:40s} unique={vals.nunique():5d}  e.g. {list(vals.head(2))}")
    if a.label_col in anno.columns:
        vc = anno[a.label_col].value_counts()
        print(f"\n== '{a.label_col}': {len(vc)} classes, top 25 ==")
        print(vc.head(25).to_string())
    if a.beta:
        names = []
        for p in a.beta:
            bf = open_beta_file(p)
            print(f"\n== {p} ==\n  {len(bf.header)} columns, {len(bf.beta_idx)} beta columns, "
                  f"sep={'TAB' if bf.sep == chr(9) else repr(bf.sep)}")
            print(f"  header starts: {bf.header[:6]}")
            names += bf.sample_names
        try:
            ids = map_samples(names, anno)
            print(f"  -> mapping OK, first: {list(zip(names[:3], ids[:3]))}")
        except ValueError as e:
            print(f"  -> {e}")


def cmd_prepare(a):
    from .prepare import prepare
    prepare(a.beta, a.series, a.out, n_top=a.n_top, label_col=a.label_col,
            exclude_lists=a.exclude, probes_from=a.probes_from, max_nan_frac=a.max_nan,
            assume_order=a.assume_order, chunksize=a.chunksize)


def cmd_run(a):
    from .evaluate import run
    from .plots import plot
    res = run(a.data, a.out, models=a.models.split(","), levels=a.levels, folds=a.folds,
              repeats=a.repeats, classes_regex=a.classes, min_class_size=a.min_class_size,
              external=a.external, quick=a.quick, seed=a.seed)
    plot(res)
    if a.external:
        plot(res, split="external")


def cmd_plot(a):
    from .plots import plot
    plot(a.results, out=a.out, metric=a.metric, split=a.split)


def cmd_summary(a):
    from .report import summary
    text = summary(a.results, split=a.split)
    if a.out:
        Path(a.out).write_text(text)
        print(f"[summ] {a.out}")
    else:
        print(text)


def cmd_synth(a):
    from .synthetic import make_synthetic
    make_synthetic(a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="methylsparse", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download series matrix (+ optional processed data) from GEO")
    s.add_argument("--gse", required=True)
    s.add_argument("--out", help="default data/raw/<GSE>")
    s.add_argument("--suppl", nargs="*", help="supplementary file names to fetch, or 'auto'")
    s.add_argument("--filters", action="store_true", help="also fetch Capper et al. probe filter lists")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("inspect", help="show annotation fields, classes and beta-file header")
    s.add_argument("--series", nargs="+", required=True)
    s.add_argument("--beta", nargs="*")
    s.add_argument("--label-col", default="methylation class")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("prepare", help="stream raw beta matrix -> compact .npz")
    s.add_argument("--beta", nargs="+", required=True, help="one or more beta matrix files (same probe order)")
    s.add_argument("--series", nargs="+", required=True, help="series matrix file(s)")
    s.add_argument("--out", required=True, help="output .npz")
    s.add_argument("--n-top", type=int, default=20000, help="keep the N most variable CpGs")
    s.add_argument("--label-col", default="methylation class")
    s.add_argument("--exclude", nargs="*", help="probe lists to drop (e.g. data/filters/*.txt)")
    s.add_argument("--probes-from", help="reuse the probe set of an existing .npz (external validation)")
    s.add_argument("--max-nan", type=float, default=0.05, help="max missing fraction per probe")
    s.add_argument("--assume-order", action="store_true", help="map matrix columns to samples by order")
    s.add_argument("--chunksize", type=int, default=20000)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("run", help="cross-validated benchmark across sparsity levels + plot")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--models", default="rf,logreg,mlp")
    s.add_argument("--levels", default="1,0.2b,0.05b,0.01b,0.005b,0.002b",
                   help="keep fractions; suffix b = binarised single reads")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--repeats", type=int, default=3, help="random masks per sparsity level")
    s.add_argument("--classes", help="regex to restrict classes, e.g. '^(MB|EPN|ATRT|ETMR)'")
    s.add_argument("--min-class-size", type=int, default=10)
    s.add_argument("--external", help=".npz built with --probes-from, evaluated after CV")
    s.add_argument("--quick", action="store_true", help="2 folds, small models: smoke test")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("plot", help="re-draw the figure from results.csv")
    s.add_argument("--results", required=True)
    s.add_argument("--out")
    s.add_argument("--metric", default="balanced_accuracy")
    s.add_argument("--split", default="cv")
    s.set_defaults(fn=cmd_plot)

    s = sub.add_parser("summary", help="markdown tables (mean ± SD) from results.csv")
    s.add_argument("--results", required=True)
    s.add_argument("--split", default="cv")
    s.add_argument("--out", help="write to file instead of printing")
    s.set_defaults(fn=cmd_summary)

    s = sub.add_parser("synth", help="write synthetic GEO-like test files")
    s.add_argument("--out", default="data/synth")
    s.set_defaults(fn=cmd_synth)

    a = p.parse_args(argv)
    if a.cmd == "run" and a.quick:
        a.folds = min(a.folds, 2)
        a.repeats = min(a.repeats, 1)
    a.fn(a)


if __name__ == "__main__":
    main()
