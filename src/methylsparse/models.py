"""Classifiers. All expose fit(X, y) and predict_proba(X) where X may contain
NaN for unobserved CpGs (at prediction time)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- baselines --
# Trained on complete array data. At test time unobserved CpGs are filled with
# the training mean: the naive "train on arrays, apply to sparse data" setup.

def make_rf(n_estimators: int = 500, seed: int = 0, n_jobs: int = -1):
    return make_pipeline(
        SimpleImputer(strategy="mean", keep_empty_features=True),
        RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt",
                               class_weight="balanced", n_jobs=n_jobs, random_state=seed))


def make_logreg(C: float = 0.05, max_iter: int = 500):
    return make_pipeline(
        SimpleImputer(strategy="mean", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"))


# ------------------------------------------------------------- masked MLP --

class MaskedMLP:
    """MLP that sees (value, observed-mask) per CpG and is trained with random
    CpG dropout + read binarisation, so it learns to classify sparse samples.

    Input per CpG: [x - 0.5 if observed else 0,  1 if observed else 0].
    """

    def __init__(self, hidden=(512, 256), dropout=0.3, epochs=40, batch_size=64,
                 lr=1e-3, weight_decay=1e-4, min_keep=0.002, p_full=0.2,
                 p_binarize=0.7, label_smoothing=0.05, min_steps=400, seed=0, device=None,
                 verbose=True):
        self.hidden, self.dropout, self.epochs = hidden, dropout, epochs
        self.batch_size, self.lr, self.weight_decay = batch_size, lr, weight_decay
        self.min_keep, self.p_full, self.p_binarize = min_keep, p_full, p_binarize
        self.label_smoothing, self.seed, self.verbose = label_smoothing, seed, verbose
        self.min_steps = min_steps
        self.device = device

    # lazily import torch so the baselines work without it
    def _torch(self):
        import torch
        return torch

    def _build(self, p, n_classes):
        torch = self._torch()
        nn = torch.nn
        layers, d = [], 2 * p
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        return nn.Sequential(*layers)

    def _augment(self, xb):
        """Random sparsity per sample, applied on the device."""
        torch = self._torch()
        n = xb.shape[0]
        lo = math.log(self.min_keep)
        f = torch.exp(torch.empty(n, 1, device=xb.device).uniform_(lo, 0.0))
        full = torch.rand(n, 1, device=xb.device) < self.p_full
        f = torch.where(full, torch.ones_like(f), f)
        obs = torch.rand_like(xb) < f
        binz = (torch.rand(n, 1, device=xb.device) < self.p_binarize) & ~full
        xr = torch.where(binz, (torch.rand_like(xb) < xb).float(), xb)
        return torch.cat([torch.where(obs, xr - 0.5, torch.zeros_like(xr)), obs.float()], dim=1)

    @staticmethod
    def _encode(x):
        obs = ~np.isnan(x)
        v = np.where(obs, x - 0.5, 0.0).astype(np.float32)
        return np.concatenate([v, obs.astype(np.float32)], axis=1)

    def fit(self, X, y):
        torch = self._torch()
        torch.manual_seed(self.seed)
        dev = self.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device_ = dev
        self.classes_, yi = np.unique(y, return_inverse=True)
        X = np.nan_to_num(np.asarray(X, np.float32), nan=0.5)
        Xt = torch.from_numpy(X).to(dev)
        yt = torch.from_numpy(yi).long().to(dev)

        counts = np.bincount(yi, minlength=len(self.classes_))
        w = torch.tensor(len(yi) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
        self.net_ = self._build(X.shape[1], len(self.classes_)).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        per_epoch = math.ceil(len(yi) / self.batch_size)
        # small datasets: train at least `min_steps` optimizer steps
        self.epochs_ = max(self.epochs, math.ceil(self.min_steps / per_epoch))
        steps = self.epochs_ * per_epoch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=self.label_smoothing)

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(self.epochs_):
            self.net_.train()
            perm = torch.randperm(len(yi), generator=g).to(dev)
            tot = 0.0
            for s in range(0, len(yi), self.batch_size):
                idx = perm[s:s + self.batch_size]
                loss = lossf(self.net_(self._augment(Xt[idx])), yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(idx)
            if self.verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == self.epochs_):
                print(f"        [mlp] epoch {ep + 1:3d}/{self.epochs_}  loss {tot / len(yi):.4f}  ({dev})")
        return self

    def predict_proba(self, X, batch_size=256):
        torch = self._torch()
        self.net_.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                xb = torch.from_numpy(self._encode(np.asarray(X[s:s + batch_size], np.float32)))
                out.append(torch.softmax(self.net_(xb.to(self.device_)), dim=1).cpu().numpy())
        return np.concatenate(out)


def make_model(name: str, quick: bool = False, seed: int = 0, **kw):
    if name == "rf":
        return make_rf(n_estimators=100 if quick else 500, seed=seed)
    if name == "logreg":
        return make_logreg(max_iter=200 if quick else 500)
    if name == "mlp":
        return MaskedMLP(epochs=8 if quick else 40, seed=seed, **kw)
    raise ValueError(f"unknown model '{name}' (choose rf, logreg, mlp)")
