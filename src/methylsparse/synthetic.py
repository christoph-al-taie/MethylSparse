"""Synthetic GEO-like files for a smoke test of the whole pipeline.

Writes a probe x sample beta matrix with interleaved 'Detection Pval' columns
and a matching series matrix whose characteristics carry 'methylation class'
and 'material', mimicking the layout of GSE90496.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

FAMILIES = {
    "MB": ["MB, WNT", "MB, SHH CHL AD", "MB, G3", "MB, G4"],
    "EPN": ["EPN, PF A", "EPN, PF B", "EPN, RELA"],
    "ATRT": ["ATRT, MYC", "ATRT, SHH", "ATRT, TYR"],
    "CONTR": ["CONTR, CEBM", "CONTR, HEMI"],
}


def _profiles(rng, p):
    base = rng.beta(0.6, 0.6, p)
    out = {}
    for fam, classes in FAMILIES.items():
        fam_prof = base.copy()
        idx = rng.choice(p, p // 8, replace=False)
        fam_prof[idx] = rng.beta(0.4, 0.4, idx.size)
        for c in classes:
            prof = fam_prof.copy()
            idx = rng.choice(p, p // 30, replace=False)
            prof[idx] = 1 - prof[idx]
            out[c] = prof
    return base, out


def _write(outdir, tag, gse, rng, base, profiles, n_per_class, probes, start_id):
    classes = [c for c in profiles for _ in range(n_per_class)]
    rng.shuffle(classes)
    n, p = len(classes), len(probes)
    X = np.empty((n, p), np.float32)
    material = []
    for i, c in enumerate(classes):
        purity = rng.uniform(0.5, 1.0)
        x = purity * profiles[c] + (1 - purity) * base + rng.normal(0, 0.06, p)
        ffpe = rng.random() < 0.4
        if ffpe:
            x += rng.normal(0, 0.03, p)
        material.append("FFPE" if ffpe else "Frozen")
        X[i] = np.clip(x, 0.005, 0.995)
    X[rng.random(X.shape) < 0.001] = np.nan

    gsm = [f"GSM{start_id + i}" for i in range(n)]
    titles = [f"SAMPLE {start_id + i}" for i in range(n)]

    beta = Path(outdir) / f"{gse}_beta.txt.gz"
    with gzip.open(beta, "wt") as fh:
        hdr = ["ID_REF"] + [h for t in titles for h in (t, "Detection Pval")]
        fh.write("\t".join(hdr) + "\n")
        for j, pid in enumerate(probes):
            vals = []
            for i in range(n):
                v = X[i, j]
                vals += ["" if np.isnan(v) else f"{v:.4f}", "0.00"]
            fh.write(pid + "\t" + "\t".join(vals) + "\n")

    sm = Path(outdir) / f"{gse}_series_matrix.txt.gz"
    q = lambda vals: "\t".join(f'"{v}"' for v in vals)  # noqa: E731
    with gzip.open(sm, "wt") as fh:
        fh.write(f'!Series_title\t"synthetic {tag} set"\n')
        fh.write(f'!Series_geo_accession\t"{gse}"\n')
        fh.write(f"!Sample_title\t{q(titles)}\n")
        fh.write(f"!Sample_geo_accession\t{q(gsm)}\n")
        fh.write(f"!Sample_source_name_ch1\t{q(['tumor'] * n)}\n")
        fh.write(f"!Sample_characteristics_ch1\t{q(['methylation class: ' + c for c in classes])}\n")
        fh.write(f"!Sample_characteristics_ch1\t{q(['material: ' + m for m in material])}\n")
        fh.write("!series_matrix_table_begin\n")
        fh.write('"ID_REF"\t' + q(gsm) + "\n")
        fh.write("!series_matrix_table_end\n")
    return beta, sm


def make_synthetic(outdir, p=3000, n_per_class=25, n_val_per_class=8, seed=1):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    probes = [f"cg{i:08d}" for i in range(p)] + [f"ch.1.{i}F" for i in range(30)] + \
             [f"rs{i}" for i in range(20)]
    base, prof = _profiles(rng, len(probes))
    a = _write(outdir, "reference", "GSE000001", rng, base, prof, n_per_class, probes, 1000)
    b = _write(outdir, "validation", "GSE000002", rng, base, prof, n_val_per_class, probes, 9000)
    excl = outdir / "exclude_demo.txt"
    excl.write_text("\n".join(probes[:50]) + "\n")
    print(f"[synth] wrote {a[0].name}, {a[1].name}, {b[0].name}, {b[1].name} to {outdir}")
    return a, b, excl
