"""Helpers for GEO: URLs, downloads and series-matrix parsing.

Only the Python standard library and pandas are used, so this works without
GEOquery / GEOparse.
"""
from __future__ import annotations

import gzip
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

# Probe exclusion lists published with the Capper et al. (2018) training code
# (https://github.com/mwsill/mnp_training, folder "filter").
MNP_FILTER_BASE = "https://raw.githubusercontent.com/mwsill/mnp_training/master/filter"
MNP_FILTER_FILES = [
    "amb_3965probes.vh20151030.txt",       # ambiguous mapping
    "epicV1B2_32260probes.vh20160325.txt",  # not on EPIC
    "snp_7998probes.vh20151030.txt",        # SNP-affected
    "xy_11551probes.vh20151030.txt",        # sex chromosomes
]


def series_dir(gse: str) -> str:
    """GSE90496 -> https://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90496"""
    gse = gse.upper()
    num = gse[3:]
    stub = "GSE" + (num[:-3] if len(num) > 3 else "") + "nnn"
    return f"{GEO_FTP}/{stub}/{gse}"


def _open_url(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "methylsparse/0.1"})
    return urllib.request.urlopen(req, timeout=120)


def list_dir(url: str) -> list[str]:
    """Return file names linked from a GEO FTP-over-HTTPS directory listing."""
    with _open_url(url.rstrip("/") + "/") as r:
        html = r.read().decode("utf-8", "replace")
    names = re.findall(r'href="([^"?/][^"]*)"', html)
    return [n for n in names if not n.startswith("http") and n not in ("../",)]


def download(url: str, dest: Path, overwrite: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"[skip] {dest} exists")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    with _open_url(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                unit, div = ("GB", 1e9) if total >= 1e9 else ("MB", 1e6)
                sys.stderr.write(f"\r       {done / div:7.2f} / {total / div:7.2f} {unit}")
        if total:
            sys.stderr.write("\n")
    shutil.move(tmp, dest)
    return dest


def download_series_matrix(gse: str, outdir: Path) -> list[Path]:
    """Download all series-matrix files of a GSE (usually exactly one)."""
    mdir = series_dir(gse) + "/matrix"
    names = [n for n in list_dir(mdir) if n.endswith("_series_matrix.txt.gz")]
    if not names:
        raise RuntimeError(f"No series matrix found under {mdir}")
    return [download(f"{mdir}/{n}", Path(outdir) / n) for n in names]


def download_mnp_filters(outdir: Path) -> list[Path]:
    return [download(f"{MNP_FILTER_BASE}/{n}", Path(outdir) / n) for n in MNP_FILTER_FILES]


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _split(line: str) -> list[str]:
    return [c.strip().strip('"') for c in line.rstrip("\n").split("\t")]


def parse_series_matrix(path: Path | str) -> pd.DataFrame:
    """Parse the '!Sample_*' header of a GEO series matrix into a DataFrame.

    Index: GSM accession. Columns: title, source_name_ch1, description, ...
    plus one column per characteristics key, e.g. 'methylation class',
    'material'. (In GEOquery these appear as 'methylation class:ch1'.)
    """
    rows: dict[str, list[list[str]]] = {}
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = _split(line)
            key = parts[0][len("!Sample_"):]
            rows.setdefault(key, []).append(parts[1:])

    if "geo_accession" not in rows:
        raise ValueError(f"{path}: no !Sample_geo_accession line, is this a series matrix?")
    gsm = rows["geo_accession"][0]
    df = pd.DataFrame(index=pd.Index(gsm, name="gsm"))

    for key, vals in rows.items():
        if key == "geo_accession":
            continue
        if key.startswith("characteristics"):
            for v in vals:
                for i, cell in enumerate(v):
                    if ":" not in cell:
                        continue
                    k, val = cell.split(":", 1)
                    k = k.strip()
                    if k not in df.columns:
                        df[k] = pd.NA
                    df.iloc[i, df.columns.get_loc(k)] = val.strip()
        else:
            for j, v in enumerate(vals):
                col = key if j == 0 else f"{key}_{j + 1}"
                if len(v) == len(gsm):
                    df[col] = v
    return df


def read_probe_list(paths: list[Path | str]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        with _open_text(Path(p)) as fh:
            for line in fh:
                tok = line.strip().split()
                if tok:
                    out.add(tok[0].strip('"'))
    return out
