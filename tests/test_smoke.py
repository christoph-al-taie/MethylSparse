"""End-to-end smoke test on synthetic GEO-like data (about 30 s on CPU)."""
import gzip

import numpy as np
import pandas as pd

from methylsparse.cli import main
from methylsparse.sparsity import simulate


def test_simulate_shapes_and_values():
    rng = np.random.default_rng(0)
    X = rng.random((50, 400)).astype(np.float32)
    S = simulate(X, 0.1, True, rng)
    obs = ~np.isnan(S)
    assert S.shape == X.shape
    assert 0.05 < obs.mean() < 0.15
    assert set(np.unique(S[obs])) <= {0.0, 1.0}
    assert obs.any(axis=1).all()


def test_pipeline(tmp_path):
    raw = tmp_path / "raw"
    main(["synth", "--out", str(raw)])
    ref = tmp_path / "ref.npz"
    main(["prepare", "--beta", str(raw / "GSE000001_beta.txt.gz"),
          "--series", str(raw / "GSE000001_series_matrix.txt.gz"),
          "--out", str(ref), "--n-top", "800", "--exclude", str(raw / "exclude_demo.txt")])

    # same data split into two GSM-labelled batch files (IDAT-route layout)
    df = pd.read_csv(raw / "GSE000001_beta.txt.gz", sep="\t", index_col=0)
    df = df[[c for c in df.columns if not c.startswith("Detection")]]
    df.columns = ["GSM" + c.split()[-1] for c in df.columns]
    parts = []
    for k, cols in enumerate(np.array_split(df.columns, 2)):
        p = tmp_path / f"beta_batch_{k:03d}.csv.gz"
        with gzip.open(p, "wt") as fh:
            df[cols].to_csv(fh, sep="\t", index_label="ID_REF")
        parts.append(str(p))
    ref2 = tmp_path / "ref2.npz"
    main(["prepare", "--beta", *parts, "--series", str(raw / "GSE000001_series_matrix.txt.gz"),
          "--out", str(ref2), "--n-top", "800", "--exclude", str(raw / "exclude_demo.txt")])
    a, b = np.load(ref, allow_pickle=True), np.load(ref2, allow_pickle=True)
    assert np.array_equal(a["probes"], b["probes"])
    assert np.allclose(a["X"], b["X"])
    assert list(a["y"]) == list(b["y"])

    val = tmp_path / "val.npz"
    main(["prepare", "--beta", str(raw / "GSE000002_beta.txt.gz"),
          "--series", str(raw / "GSE000002_series_matrix.txt.gz"),
          "--out", str(val), "--probes-from", str(ref)])

    out = tmp_path / "run"
    main(["run", "--data", str(ref), "--out", str(out), "--quick",
          "--levels", "1,0.1b", "--external", str(val), "--classes", "^(MB|EPN|ATRT)"])
    res = pd.read_csv(out / "results.csv")
    assert set(res["model"]) == {"rf", "logreg", "mlp"}
    assert set(res["split"]) == {"cv", "external"}
    full = res[(res.keep_frac == 1) & (res.split == "cv")]
    assert (full.balanced_accuracy > 0.8).all()
    assert (out / "balanced_accuracy_vs_sparsity_cv.png").exists()
    main(["summary", "--results", str(out / "results.csv"), "--out", str(out / "summary.md")])
    assert "| 100% (array) |" in (out / "summary.md").read_text()


def test_map_by_sample_number():
    """GSE90496 layout: matrix 'SAMPLE n', series-matrix title '<class>, sample n [reference set]'."""
    from methylsparse.beta_io import map_samples
    anno = pd.DataFrame(
        {"title": ["GBM, G34, sample 1 [reference set]", "DMG, K27, sample 2 [reference set]",
                   "MB, G4, sample 10 [reference set]"],
         "material": ["Frozen", "FFPE", "Frozen"]},
        index=pd.Index(["GSM2402854", "GSM2402855", "GSM2402863"], name="gsm"))
    ids = map_samples(["SAMPLE 10", "SAMPLE 1", "SAMPLE 2"], anno)  # different order on purpose
    assert ids == ["GSM2402863", "GSM2402854", "GSM2402855"]
