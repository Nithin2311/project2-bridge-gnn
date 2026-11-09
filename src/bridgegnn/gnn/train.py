"""Training protocol for step 06.

select():  hyperparameters chosen once on the selection seed by GroupKFold (by person_id) over development rows,
           early stopping on fold-validation AUPRC, score = mean out-of-fold AUPRC over folds with both classes.
run():     per seed, refit on all development rows for the mean best epoch count, predict test once, write the
           predictions (dev out-of-fold + test), the curves and the results row (replacing an earlier row with the
           same run_id, target, seed).
Covariate scalers are fit on the training rows of each fold, or on all development rows for the refit.
"""
from __future__ import annotations

import copy
import json
import random
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch_geometric.loader import DataLoader

from .. import contract
from .data import Example, apply_cov_scaler, fit_cov_scaler
from .models import SubgraphGNN, to_pyg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(examples: list[Example], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(to_pyg(examples), batch_size=batch_size, shuffle=shuffle, generator=gen)


def _auprc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None


@torch.no_grad()
def predict(model: SubgraphGNN, examples: list[Example], batch_size: int = 256) -> np.ndarray:
    if not examples:
        return np.zeros(0, np.float64)
    model.eval()
    out = np.zeros(len(examples), np.float64)
    for batch in _loader(examples, batch_size, shuffle=False, seed=0):
        logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
        out[batch.row_pos.numpy()] = torch.sigmoid(logits).numpy()
    return out


def fit(train_ex: list[Example], val_ex: list[Example], params: dict, seed: int, cfg_gnn: dict,
        model_kwargs: dict) -> tuple[SubgraphGNN, int, dict]:
    """Early stopping on val AUPRC when val_ex is given; otherwise exactly params['epochs'] epochs."""
    set_seed(seed)
    model = SubgraphGNN(**model_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params.get("weight_decay", 0.0)))
    y = np.array([e.y for e in train_ex], dtype=np.float64)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = _loader(train_ex, int(cfg_gnn["batch_size"]), shuffle=True, seed=seed)
    max_epochs = int(params["epochs"]) if not val_ex else int(cfg_gnn["max_epochs"])
    patience = int(cfg_gnn["patience"])
    yv = np.array([e.y for e in val_ex], dtype=np.int64) if val_ex else None
    curves = {"train_loss": [], "val_auprc": []}
    best_score, best_epoch, best_state, stale = -np.inf, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        curves["train_loss"].append(total / max(n, 1))
        if val_ex:
            score = _auprc(yv, predict(model, val_ex))
            curves["val_auprc"].append(score)
            s = -1.0 if score is None else score
            if s > best_score:
                best_score, best_epoch, best_state, stale = s, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        else:
            best_epoch = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, curves


def group_folds(examples: list[Example], n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pids = np.array([e.person_id for e in examples])
    uniq = np.unique(pids)
    rng = np.random.default_rng(seed)
    relabel = dict(zip(uniq, rng.permutation(len(uniq))))
    groups = np.array([relabel[p] for p in pids])
    return [(tr, va) for tr, va in GroupKFold(n_splits=n_folds).split(np.zeros(len(examples)), groups=groups)]


def select(dev_ex: list[Example], grid: list[dict], cfg: dict, model_kwargs: dict, seed: int,
           cov_names: list[str]) -> tuple[dict, float | None, np.ndarray, int, list]:
    folds = group_folds(dev_ex, int(cfg["split"]["cv_folds"]), seed)
    best = None
    for params in grid:
        oof = np.full(len(dev_ex), np.nan)
        scores, epochs, curves = [], [], []
        for k, (tr, va) in enumerate(folds):
            tr_ex = [dev_ex[i] for i in tr]
            va_ex = [dev_ex[i] for i in va]
            yv = np.array([e.y for e in va_ex])
            if len(np.unique(yv)) < 2:
                warnings.warn(f"fold {k}: validation rows have a single class; fold skipped for selection", UserWarning, stacklevel=2)
                mean, sd = fit_cov_scaler(tr_ex, cov_names)
                m, be, cv = fit(apply_cov_scaler(tr_ex, mean, sd, cov_names), [], params | {"epochs": 1}, seed, cfg["gnn"], model_kwargs)
                oof[va] = predict(m, apply_cov_scaler(va_ex, mean, sd, cov_names))
                curves.append(cv)
                continue
            mean, sd = fit_cov_scaler(tr_ex, cov_names)
            m, be, cv = fit(apply_cov_scaler(tr_ex, mean, sd, cov_names), apply_cov_scaler(va_ex, mean, sd, cov_names),
                            params, seed, cfg["gnn"], model_kwargs)
            p = predict(m, apply_cov_scaler(va_ex, mean, sd, cov_names))
            oof[va] = p
            scores.append(_auprc(yv, p))
            epochs.append(be)
            curves.append(cv)
        score = float(np.mean([s for s in scores if s is not None])) if scores else None
        mean_epoch = int(round(float(np.mean(epochs)))) if epochs else 1
        cand = (params, score, oof, max(mean_epoch, 1), curves)
        if best is None or (score is not None and (best[1] is None or score > best[1])):
            best = cand
    return best


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    two = len(np.unique(y)) == 2
    return {"test_auroc": float(roc_auc_score(y, p)) if two else float("nan"),
            "test_auprc": float(average_precision_score(y, p)) if two else float("nan"),
            "test_brier": float(brier_score_loss(y, p)), "n_test_rows": int(len(y)), "n_test_pos": int(y.sum())}


def _upsert_results(path: Path, row: dict) -> None:
    new = pl.DataFrame([row]).select(list(contract.SCHEMAS["results"])).cast(contract.SCHEMAS["results"])
    if path.exists():
        old = contract.read(path, "results").filter(~((pl.col("run_id") == row["run_id"]) & (pl.col("target") == row["target"])
                                                      & (pl.col("seed") == row["seed"])))
        new = pl.concat([old, new])
    contract.write(new, path, "results")


def run(run_id: str, target: str, dev_ex: list[Example], test_ex: list[Example], grid: list[dict], seeds: list[int],
        cfg: dict, model_kwargs: dict, flags: dict, out_dir: Path, cov_names: list[str], family: str = "gnn") -> list[dict]:
    out_dir = Path(out_dir)
    for sub in ["predictions", "results", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    sel_seed = int(cfg["gnn"].get("selection_seed", 0))
    best_params, best_score, oof, epochs, fold_curves = select(dev_ex, grid, cfg, model_kwargs, sel_seed, cov_names)
    y_test = np.array([e.y for e in test_ex], dtype=np.int64)
    results = []
    for seed in seeds:
        mean, sd = fit_cov_scaler(dev_ex, cov_names)
        model, _, refit_curves = fit(apply_cov_scaler(dev_ex, mean, sd, cov_names), [], best_params | {"epochs": epochs},
                                     seed, cfg["gnn"], model_kwargs)
        p_test = predict(model, apply_cov_scaler(test_ex, mean, sd, cov_names)) if test_ex else np.zeros(0)
        preds = pl.DataFrame({
            "run_id": [run_id] * (len(dev_ex) + len(test_ex)), "model": [flags.get("arch", model_kwargs.get("arch", ""))] * (len(dev_ex) + len(test_ex)),
            "target": [target] * (len(dev_ex) + len(test_ex)), "seed": pl.Series([seed] * (len(dev_ex) + len(test_ex)), dtype=pl.Int32),
            "person_id": pl.Series([e.person_id for e in dev_ex] + [e.person_id for e in test_ex], dtype=pl.Int64),
            "index_date": pl.Series([e.index_date for e in dev_ex] + [e.index_date for e in test_ex], dtype=pl.Date),
            "split": [e.split for e in dev_ex] + [e.split for e in test_ex],
            "y": [bool(e.y) for e in dev_ex] + [bool(e.y) for e in test_ex],
            "p": pl.Series(np.concatenate([oof, p_test]).astype(np.float64), dtype=pl.Float64)})
        contract.write(preds, out_dir / "predictions" / f"{run_id}_{target}_seed{seed}.parquet", "predictions")
        (out_dir / "logs" / f"06_{run_id}_{target}_seed{seed}.json").write_text(json.dumps(
            {"selection": fold_curves, "refit": refit_curves, "best_params": best_params, "epochs": epochs,
             "dev_oof_auprc": best_score}, default=float))
        m = _metrics(y_test, p_test) if len(test_ex) else {"test_auroc": float("nan"), "test_auprc": float("nan"),
                                                           "test_brier": float("nan"), "n_test_rows": 0, "n_test_pos": 0}
        row = {"run_id": run_id, "family": family, "model": flags.get("arch", model_kwargs.get("arch", "")), "target": target,
               "seed": int(seed), "layer": flags.get("layer", "U"), "features": flags.get("features", ""),
               "params_json": json.dumps({**flags, **best_params, "epochs": epochs}, default=str),
               "dev_oof_auprc": best_score if best_score is not None else float("nan"), **m}
        _upsert_results(out_dir / "results" / "gnn.csv", row)
        results.append(row)
    return results
