"""Pure metric functions for step 08: patient-level bootstrap, calibration, F1 threshold, strata, paired deltas."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EPS = 1e-6


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def auprc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None


def _patient_resamples(person_id: np.ndarray, n: int, seed: int):
    """Yield index arrays: each resample draws unique patients with replacement and takes all their rows."""
    uniq, inv = np.unique(person_id, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        yield np.concatenate([rows_of[k] for k in draw])


def bootstrap_ci(y: np.ndarray, p: np.ndarray, person_id: np.ndarray, stat, n: int, seed: int) -> tuple[float, float, float]:
    y, p, person_id = np.asarray(y), np.asarray(p), np.asarray(person_id)
    point = stat(y, p)
    vals = []
    for idx in _patient_resamples(person_id, n, seed):
        v = stat(y[idx], p[idx])
        if v is not None:
            vals.append(v)
    if not vals or point is None:
        return (float("nan") if point is None else float(point), float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    """brier; slope and intercept_joint from y ~ a + b*logit(p); citl = calibration-in-the-large, the intercept
    of y ~ a + offset(logit(p)) with the slope fixed at 1 (Van Calster / Steyerberg)."""
    y, p = np.asarray(y, dtype=int), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    out = {"brier": float(brier_score_loss(y, p)), "slope": float("nan"), "intercept_joint": float("nan"), "citl": float("nan")}
    if len(np.unique(y)) == 2:
        logit = np.log(p / (1 - p))
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y)
        out["slope"], out["intercept_joint"] = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        out["citl"] = float(brentq(lambda a: float((y - expit(a + logit)).sum()), -20, 20))
    return out


def reliability(y: np.ndarray, p: np.ndarray, bins: int) -> pl.DataFrame:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return pl.DataFrame({"bin": list(range(1, bins + 1)),
                         "p_mean": [float(p[c].mean()) if len(c) else float("nan") for c in chunks],
                         "y_rate": [float(y[c].mean()) if len(c) else float("nan") for c in chunks],
                         "n": [len(c) for c in chunks]})


def f1_threshold(y_dev: np.ndarray, p_dev: np.ndarray) -> float:
    y_dev, p_dev = np.asarray(y_dev, dtype=int), np.asarray(p_dev, dtype=float)
    if len(y_dev) == 0 or len(np.unique(y_dev)) < 2:
        raise ValueError("F1 threshold needs development out-of-fold rows with both classes")
    cands = np.unique(p_dev)
    best_t, best_f1 = cands[0], -1.0
    for t in cands:
        pred = p_dev >= t
        tp = int((pred & (y_dev == 1)).sum()); fp = int((pred & (y_dev == 0)).sum()); fn = int((~pred & (y_dev == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y, pred = np.asarray(y, dtype=int), np.asarray(p, dtype=float) >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "threshold": float(thr)}


def stratified(df: pl.DataFrame, strata: list[str], min_pos: int) -> pl.DataFrame:
    rows = []
    for s in strata:
        for level, sub in df.group_by(s, maintain_order=True):
            lv = level[0] if isinstance(level, tuple) else level
            y, p = sub["y"].to_numpy().astype(int), sub["p"].to_numpy()
            pos = int(y.sum())
            ok = pos >= min_pos and (len(y) - pos) >= 1
            rows.append({"stratum": s, "level": str(lv), "n": len(y), "pos": pos,
                         "auroc": auroc(y, p) if ok else None, "auprc": auprc(y, p) if ok else None})
    return pl.DataFrame(rows, schema={"stratum": pl.Utf8, "level": pl.Utf8, "n": pl.Int64, "pos": pl.Int64,
                                      "auroc": pl.Float64, "auprc": pl.Float64})


def paired_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, person_id: np.ndarray, n: int, seed: int) -> dict:
    """Bootstrap (same patient resamples for both runs) of AUROC(a) - AUROC(b) and AUPRC(a) - AUPRC(b)."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    out = {}
    for name, fn in [("auroc", auroc), ("auprc", auprc)]:
        pa, pb = fn(y, p_a), fn(y, p_b)
        point = (pa - pb) if pa is not None and pb is not None else float("nan")
        ds = []
        for idx in _patient_resamples(person_id, n, seed):
            a, b = fn(y[idx], p_a[idx]), fn(y[idx], p_b[idx])
            if a is not None and b is not None:
                ds.append(a - b)
        lo, hi = (np.percentile(ds, [2.5, 97.5]) if ds else (float("nan"), float("nan")))
        out.update({f"d_{name}": float(point), f"d_{name}_lo": float(lo), f"d_{name}_hi": float(hi)})
    return out
