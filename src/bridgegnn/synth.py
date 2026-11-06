"""Synthetic fixtures shaped exactly like the Project 1 inputs Project 2 consumes.

Deterministic for a seed. Labels are produced by Project 1's own landmark-label code so their structure
matches the real p2_labels.parquet. A precedence signal A -> B is planted for the directed-layer tests.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import networkx as nx
import numpy as np
import polars as pl

from bridgedx import labels as p1_labels

from .contract import TARGETS

ERA_START = dt.date(2015, 10, 1)
DATA_END = dt.date(2023, 6, 30)
LABEL_CFG = {"project2": {"grid_start": "2017-10-01", "grid_step_days": 182, "min_history_days": 730,
                          "require_encounter_days": 365, "horizons_months": [6, 12]}}


@dataclass
class SynthData:
    labels: pl.DataFrame
    codes_long: pl.DataFrame
    visits: pl.DataFrame
    demo: pl.DataFrame
    split: pl.DataFrame
    nodes: pl.DataFrame
    target_defs: pl.DataFrame
    bridges: pl.DataFrame
    graph: nx.Graph
    precedence_pair: tuple[str, str]
    targets: dict[str, dict[str, set[str]]]


def _categories(n: int) -> list[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [f"{letters[i % 26]}{(i * 7) % 100:02d}" for i in range(n)]


def make(seed: int = 0, n_patients: int = 60, n_categories: int = 30) -> SynthData:
    rng = np.random.default_rng(seed)
    cats = _categories(n_categories)
    n_nodes = max(4, int(n_categories * 2 / 3))
    node_cats, extra_cats = cats[:n_nodes], cats[n_nodes:]
    a, b = node_cats[0], node_cats[1]

    # targets: cores of 2 node codes each, community = core + 1, chapter block = core + 1 other
    targets: dict[str, dict[str, set[str]]] = {"core": {}, "community": {}, "chapter": {}}
    pool = node_cats[2:]
    for i, t in enumerate(TARGETS):
        core = {pool[3 * i], pool[3 * i + 1]}
        targets["core"][t] = core
        targets["community"][t] = core | {pool[3 * i + 2]}
        targets["chapter"][t] = core | {pool[(3 * i + 5) % len(pool)]}

    person_ids = [115_000_000 + i for i in range(n_patients)]
    split_labels = rng.choice(["train", "val", "test"], size=n_patients, p=[0.34, 0.33, 0.33])
    code_rows, visit_rows, demo_rows, deaths = [], [], [], {}
    for j, pid in enumerate(person_ids):
        first = ERA_START + dt.timedelta(days=int(rng.integers(0, 1200)))
        end = first + dt.timedelta(days=int(rng.integers(400, 2600)))
        end = min(end, DATA_END)
        if rng.random() < 0.10:
            deaths[pid] = end
        k = int(rng.integers(5, min(26, n_categories + 1)))
        chosen = list(rng.choice(cats, size=k, replace=False))
        if j < n_categories:  # every category appears at least once
            chosen = list(dict.fromkeys(chosen + [cats[j]]))
        span = max(1, (end - first).days)
        firsts: dict[str, dt.date] = {}
        for c in chosen:
            firsts[c] = first + dt.timedelta(days=int(rng.integers(0, span)))
        if a in firsts and b in firsts and rng.random() < 0.70:  # planted A -> B
            firsts[b] = min(end, firsts[a] + dt.timedelta(days=int(rng.integers(60, 200))))
        for c, d0 in firsts.items():
            n_rec = int(rng.integers(1, 5))
            dates = {d0} | {d0 + dt.timedelta(days=int(rng.integers(0, max(1, (end - d0).days + 1)))) for _ in range(n_rec - 1)}
            for d in dates:
                if d <= end:
                    code_rows.append((pid, c, d))
        vdates = {d for (p, _c, d) in code_rows if p == pid} | {first + dt.timedelta(days=int(rng.integers(0, span))) for _ in range(5)}
        visit_rows += [(pid, d) for d in sorted(vdates) if d <= end]
        demo_rows.append((pid, "female" if rng.random() < 0.55 else "male", dt.date(int(rng.integers(1935, 1995)), int(rng.integers(1, 13)), 1)))

    codes_long = pl.DataFrame(code_rows, schema={"person_id": pl.Int64, "category": pl.Utf8, "condition_date": pl.Date}, orient="row").unique().sort("person_id", "category", "condition_date")
    visits = pl.DataFrame(visit_rows, schema={"person_id": pl.Int64, "visit_date": pl.Date}, orient="row").unique().sort("person_id", "visit_date")
    demo = pl.DataFrame(demo_rows, schema={"person_id": pl.Int64, "sex": pl.Utf8, "birth_date": pl.Date}, orient="row")
    split = pl.DataFrame({"person_id": pl.Series(person_ids, dtype=pl.Int64), "split": pl.Series(split_labels.tolist(), dtype=pl.Utf8)})

    # Project 1 label inputs
    pc = codes_long.group_by("person_id", "category").agg(pl.col("condition_date").min().alias("first_date"))
    pre = pl.DataFrame({"person_id": pl.Series([], dtype=pl.Int64), "category": pl.Series([], dtype=pl.Utf8)})
    pat = codes_long.group_by("person_id").agg(pl.col("condition_date").min().alias("first_era"), pl.col("condition_date").max().alias("last_dx"))
    pat = pat.with_columns(pl.col("person_id").map_elements(lambda p: deaths.get(p), return_dtype=pl.Date).alias("death")).sort("person_id")
    encounters = {k[0] if isinstance(k, tuple) else k: np.array(sorted(v)) for k, v in codes_long.group_by("person_id").agg(pl.col("condition_date")).rows()}
    labels = p1_labels.build_landmark_labels(LABEL_CFG, pc, pre, pat, encounters, targets, dict(zip(split["person_id"], split["split"])))
    labels = labels.with_columns(pl.col("days_to_event").cast(pl.Int64), pl.col("horizon_months").cast(pl.Int64),
                                 pl.col("history_days").cast(pl.Int64), pl.col("followup_days").cast(pl.Int64))

    chapters = {"A": "infectious", "B": "infectious", "C": "neoplasms", "D": "blood", "E": "endocrine/metabolic", "F": "mental"}
    nodes = pl.DataFrame({"category": node_cats, "name": [f"Synthetic condition {c}" for c in node_cats],
                          "chapter": [chapters.get(c[0], "other") for c in node_cats]})
    td_rows = []
    for t in TARGETS:
        for c in node_cats:
            td_rows.append((t, c, f"Synthetic condition {c}", c in targets["core"][t], c in targets["core"][t],
                            c in targets["community"][t], c in targets["chapter"][t]))
    target_defs = pl.DataFrame(td_rows, schema=["target", "category", "name", "in_core", "in_core_0.8", "in_community", "in_chapter_block"], orient="row")

    g = nx.Graph()
    g.add_nodes_from(node_cats)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.3:
                w = float(rng.uniform(0.05, 0.6))
                g.add_edge(node_cats[i], node_cats[j], weight=w, distance=1.0 / w)
    pairs = [f"{TARGETS[i]}-{TARGETS[j]}" for i in range(4) for j in range(i + 1, 4)]
    br_rows = []
    for p in pairs:
        for c in node_cats:
            score = float(rng.uniform(0.0, 0.01)) if rng.random() < 0.3 else 0.0
            br_rows.append((p, c, score, score > 0.006))
    bridges = pl.DataFrame(br_rows, schema=["pair", "category", "bridge_betweenness", "bridge_pass"], orient="row")

    return SynthData(labels, codes_long, visits, demo, split, nodes, target_defs, bridges, g, (a, b), targets)
