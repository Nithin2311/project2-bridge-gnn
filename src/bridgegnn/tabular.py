"""Patient-grouped model selection, refit and held-out test prediction for step 05."""
from __future__ import annotations

import json
import logging
import warnings
from itertools import product

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import contract
from .eval.metrics import auroc

LOGGER = logging.getLogger(__name__)
GROUPS = {
    "prevalence": (),
    "lr": ("CODE", "DEMO", "UTIL"),
    "hgb": ("CODE", "DEMO", "UTIL", "BRIDGE"),
    "hgb_all": ("ALL", "DEMO", "UTIL"),
    "mlp": ("CODE", "DEMO", "UTIL", "BRIDGE"),
    "mlp_pool": ("CODE", "DEMO", "UTIL", "TEMP", "BRIDGE", "POOL"),
    "bridge_lr": ("BRIDGE", "DEMO", "UTIL"),
}
MODELS = list(GROUPS)
MLPS = {"mlp", "mlp_pool"}
BOOSTERS = {"hgb", "hgb_all"}
PREFIX = {"CODE": "c_", "ALL": "all_", "DEMO": "demo_", "UTIL": "util_", "TEMP": "temp_",
          "BRIDGE": "bridge_", "POOL": "pool_"}


def seed_invariant(model: str, n_dev: int) -> bool:
    # With all features and no early stopping, HistGB only uses randomness for binning large samples.
    return model in {"prevalence", "lr", "bridge_lr"} or (model in BOOSTERS and n_dev <= 200_000)


def feature_columns(model: str, columns: list[str]) -> list[str]:
    prefixes = tuple(PREFIX[group] for group in GROUPS[model])
    return [c for c in columns if c.startswith(prefixes)]


def grid(model: str) -> list[dict]:
    if model == "prevalence":
        return [{}]
    if model in {"lr", "bridge_lr"}:
        return [{"C": c, "class_weight": "balanced", "max_iter": 2000} for c in (0.01, 0.1, 1.0)]
    if model in BOOSTERS:
        # the number of boosting rounds is tuned (design section 7: baselines tuned as carefully as the GNNs)
        return [{"learning_rate": rate, "max_leaf_nodes": leaves, "max_iter": rounds,
                 "early_stopping": False, "class_weight": "balanced"}
                for rate, leaves, rounds in product((0.03, 0.1), (15, 31), (50, 100, 300))]
    if model in MLPS:
        return [{"hidden_layer_sizes": layers, "alpha": alpha, "max_iter": 300, "early_stopping": True}
                for layers, alpha in product(((64,), (128, 64)), (1e-4, 1e-3))]
    raise ValueError(f"unknown baseline {model!r}")


def make(model: str, params: dict, seed: int):
    if model == "prevalence":
        return DummyClassifier(strategy="prior")
    if model in {"lr", "bridge_lr"}:
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **params))
    if model in BOOSTERS:
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if model in MLPS:
        # The MLP's internal random row split can separate landmarks of the same patient.
        return make_pipeline(StandardScaler(),
                             MLPClassifier(random_state=seed, **(params | {"early_stopping": False})))
    raise ValueError(f"unknown baseline {model!r}")


def _prior_shift(y: np.ndarray, params: dict) -> float:
    """log(n_neg/n_pos) induced by class_weight='balanced'; removed at prediction so probabilities sit on the
    prevalence scale (same correction as step 06's logit_shift). Ranking metrics and selection are unaffected."""
    if params.get("class_weight") != "balanced":
        return 0.0
    n_pos, n_neg = int(np.sum(y)), int(len(y) - np.sum(y))
    return float(np.log(n_neg / n_pos)) if n_pos and n_neg else 0.0


def _probability(estimator, X: np.ndarray) -> np.ndarray:
    p = estimator.predict_proba(X)
    labels = estimator.classes_
    p = p[:, list(labels).index(True)] if True in labels else np.zeros(len(X))
    shift = float(getattr(estimator, "_logit_shift", 0.0))
    if shift:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        p = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) - shift)))
    return p


def _fit(model, params, seed, X, y, groups, epochs=None):
    if len(np.unique(y)) < 2:
        return make("prevalence", {}, seed).fit(X, y), None, []
    est = make(model, params, seed)
    if model not in MLPS:
        est.fit(X, y)
        est._logit_shift = _prior_shift(y, params)
        return est, None, []

    scaler, mlp = est.steps[0][1], est.steps[1][1]
    max_epochs = params["max_iter"]
    if epochs is not None:
        # Fit the winning configuration on every development row for the selected epoch count.
        scaled = scaler.fit_transform(X)
        for _ in range(epochs):
            mlp.partial_fit(scaled, y, classes=[False, True])
        return est, epochs, []

    # Inner validation is disjoint by patient from both fitting and outer scoring rows.
    if np.unique(groups).size < 2:
        return _fit(model, params, seed, X, y, groups, epochs=min(20, max_epochs))
    candidates = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    inner = next(((a, b) for a, b in candidates.split(X, y, groups)
                  if np.unique(y[a]).size == np.unique(y[b]).size == 2), None)
    if inner is None:
        return _fit(model, params, seed, X, y, groups, epochs=min(20, max_epochs))
    fit, val = inner
    X_fit, X_val = scaler.fit_transform(X[fit]), scaler.transform(X[val])
    best, best_epoch, curves = -np.inf, 1, []
    for epoch in range(1, max_epochs + 1):
        mlp.partial_fit(X_fit, y[fit], classes=[False, True])
        p_val = _probability(mlp, X_val)
        ap = float(average_precision_score(y[val], p_val))
        curves.append({"train_loss": float(mlp.loss_), "val_loss": float(log_loss(y[val], p_val)),
                       "val_auprc": ap})
        if ap > best + 1e-6:
            best, best_epoch = ap, epoch
        if epoch - best_epoch >= 20:
            break
    est, _, _ = _fit(model, params, seed, X, y, groups, epochs=best_epoch)
    return est, best_epoch, curves


def _select(model, X_dev, y_dev, groups_dev, seed, n_folds):
    unique_groups = len(np.unique(groups_dev))
    if unique_groups < 2:
        raise ValueError("grouped CV needs at least two development patients")
    folds = list(GroupKFold(n_splits=min(n_folds, unique_groups)).split(X_dev, y_dev, groups_dev))
    scoreable = [np.unique(y_dev[valid]).size == 2 for _, valid in folds]
    if not all(scoreable):
        warnings.warn("held-out fold with a single class skipped when scoring AUPRC", stacklevel=2)
    best_params, best_score, best_oof = None, None, None
    details = []
    for params in grid(model):
        oof = np.empty(len(y_dev), dtype=float)
        scores, epochs, curves = [], [], []
        for (fit, valid), score in zip(folds, scoreable):
            est, n_epochs, curve = _fit(model, params, seed, X_dev[fit], y_dev[fit], groups_dev[fit])
            oof[valid] = _probability(est, X_dev[valid])
            scores.append(float(average_precision_score(y_dev[valid], oof[valid])) if score else None)
            if n_epochs is not None:
                epochs.append(n_epochs)
            curves.append(curve)
        valid_scores = [s for s in scores if s is not None]
        mean_score = float(np.mean(valid_scores)) if valid_scores else None
        details.append({"params": params, "fold_scores": scores, "fold_epochs": epochs,
                        "curves": curves, "mean_score": mean_score})
        if best_params is None or (mean_score is not None and (best_score is None or mean_score > best_score)):
            best_params, best_score, best_oof = params.copy(), mean_score, oof
            if epochs:
                best_params["refit_epochs"] = max(1, round(float(np.median(epochs))))
    return best_params, best_score, best_oof, details


def select(model, X_dev, y_dev, groups_dev, seed, n_folds=5):
    """Mean fold AP for selection and one OOF probability per development row."""
    with threadpool_limits(limits=2):
        params, score, oof, _ = _select(model, X_dev, y_dev, groups_dev, seed, n_folds)
    return params, score, pl.DataFrame({"p": oof})


def run(model, target, tables: dict[str, pl.DataFrame], cfg: dict, seed: int, *,
        run_id: str | None = None, family: str = "baseline", subset: set[int] | None = None,
        feature_groups: dict | None = None) -> tuple[pl.DataFrame, dict]:
    run_id = run_id or f"B-{model}"
    dev_splits, test_splits = cfg["split"]["dev"], cfg["split"]["test"]
    for split, table in tables.items():
        contract.validate(table, "tabular")
        if not table["split"].eq(split).all() or not table["target"].eq(target).all():
            raise contract.ContractError(f"unexpected split or target in {split} feature table")
    dev = pl.concat([tables[s] for s in dev_splits])
    test = pl.concat([tables[s] for s in test_splits])
    if set(dev["person_id"]).intersection(test["person_id"]):
        raise contract.ContractError("a patient appears in both the development and test tables")
    if subset is not None:
        dev = dev.filter(pl.col("person_id").is_in(sorted(subset)))
    if not dev.height or not test.height:
        raise ValueError("empty development or test set")
    cols = feature_columns(model, dev.columns)
    if feature_groups is not None:
        grouped = {**feature_groups,
                   "ALL": feature_groups["all_by_target"][target],
                   "BRIDGE": feature_groups["ROW_BRIDGE"]}
        expected = {c for group in GROUPS[model] for c in grouped[group]}
        if set(cols) != expected:
            raise contract.ContractError(f"feature manifest and {target} {model} columns disagree")

    def matrix(df):
        return df.select(cols).to_numpy() if cols else np.zeros((df.height, 1))

    X, y, groups = matrix(dev), dev["positive"].to_numpy(), dev["person_id"].to_numpy()
    test_y = test["positive"].to_numpy()
    with threadpool_limits(limits=2):
        params, score, oof, details = _select(model, X, y, groups, seed, cfg["split"]["cv_folds"])
        LOGGER.info("%s", json.dumps({"run_id": run_id, "target": target, "seed": seed,
                                     "best_params": params, "grid": details}))
        refit_params = {k: v for k, v in params.items() if k != "refit_epochs"}
        epochs = params.get("refit_epochs", 20) if model in MLPS else None
        est, _, _ = _fit(model, refit_params, seed, X, y, groups, epochs=epochs)
        probabilities = _probability(est, matrix(test))
    preds = pl.concat([dev, test]).select(
        pl.lit(run_id).alias("run_id"), pl.lit(model).alias("model"), "target",
        pl.lit(seed, dtype=pl.Int32).alias("seed"), "person_id", "index_date", "split",
        pl.col("positive").alias("y"), pl.Series("p", np.concatenate([oof, probabilities])),
    )
    result = {
        "run_id": run_id, "family": family, "model": model, "target": target, "seed": seed,
        "layer": "none", "features": ",".join(GROUPS[model]), "params_json": json.dumps(params, sort_keys=True),
        "dev_oof_auprc": score, "test_auroc": auroc(test_y, probabilities),
        "test_auprc": float(average_precision_score(test_y, probabilities)) if test_y.any() else None,
        "test_brier": float(brier_score_loss(test_y, probabilities)),
        "n_test_rows": test.height, "n_test_pos": int(test_y.sum()),
    }
    return contract.validate(preds, "predictions"), result
