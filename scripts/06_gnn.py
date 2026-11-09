"""Step 6: GCN / GraphSAGE / GAT on per-row patient subgraphs. CLI contract: docs/INTERFACES.md.

Example:
  uv run python scripts/06_gnn.py --run-id G-sage --target mental_health --arch sage
  uv run python scripts/06_gnn.py --run-id A1a --target cardiometabolic --arch sage --features CODE,DEMO,UTIL,TEMP,STRUCT
  uv run python scripts/06_gnn.py --run-id D1-rewire0 --target mental_health --arch sage --layer N --null-graph rewire0
Node features: --node-features-dir, else outputs/features/ (Dylan's step 02) if present, else built into
outputs/features_xcheck/ from Project 1 files (nodefeat.build). Selection once per (run, target) on the
selection seed; refit + test per seed. --dry-run prints the resolved configuration and exits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import polars as pl
import yaml

from bridgegnn import config, contract, nodefeat
from bridgegnn.gnn import data

ALL_FEATURES = ["CODE", "DEMO", "UTIL", "TEMP", "STRUCT", "BRIDGE"]


def parse_args(argv=None):
    cfg = config.load()
    g = cfg["gnn"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--target", required=True, choices=contract.TARGETS)
    ap.add_argument("--arch", required=True, choices=["gcn", "sage", "gat"])
    ap.add_argument("--layer", default="U", choices=["U", "U-bb", "U-bh", "D", "N"])
    ap.add_argument("--features", default=",".join(ALL_FEATURES))
    ap.add_argument("--lookback", type=int, default=int(cfg["windows"]["primary_lookback"]), choices=[365, 1095, 0])
    ap.add_argument("--expand", type=int, default=0, choices=[0, 1])
    ap.add_argument("--weighted", type=int, default=1, choices=[0, 1])
    ap.add_argument("--depth", type=int, default=int(g["depth"]))
    ap.add_argument("--agg", default=g["agg"], choices=["mean", "max"])
    ap.add_argument("--heads", type=int, default=int(g["heads"]))
    ap.add_argument("--hidden", type=int, default=int(g["hidden"]))
    ap.add_argument("--seeds", type=int, nargs="+", default=[int(s) for s in cfg["seeds"]])
    ap.add_argument("--null-graph", default="none", choices=["none", "rewire0", "rewire1", "rewire2", "rewire3", "rewire4", "shuffle"])
    ap.add_argument("--drop-codes", default="none", choices=["none", "bridge"])
    ap.add_argument("--subsample-file", default=None, help="path template with {seed}; one development person_id per line")
    ap.add_argument("--node-features-dir", default=None)
    ap.add_argument("--quick", action="store_true", help="smoke run: cfg.gnn.quick epochs/patience")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv), cfg


def load_layer(args, cfg) -> tuple[nx.Graph | nx.DiGraph, bool]:
    p1 = config.p1_outputs(cfg)
    out = config.outputs_dir(cfg)
    if args.layer == "U":
        return nx.read_graphml(p1 / "graph_main.graphml"), False
    if args.layer == "U-bb":
        return nx.read_graphml(p1 / "p2" / "graph_backbone_topk.graphml"), False
    if args.layer == "U-bh":
        return nx.read_graphml(out / "layers" / "graph_bh.graphml"), False
    if args.layer == "D":
        return nx.read_graphml(out / "layers" / "graph_directed.graphml"), True
    if args.null_graph == "none":
        sys.exit("--layer N needs --null-graph")
    return nx.read_graphml(out / "layers" / "null" / f"{args.null_graph}.graphml"), False


def resolve_node_features(args, cfg) -> tuple[Path, pl.DataFrame, object, dict]:
    out = config.outputs_dir(cfg)
    if args.node_features_dir:
        d = Path(args.node_features_dir)
    elif (out / "features" / "node_x.npy").exists():
        d = out / "features"
    else:
        d = out / "features_xcheck"
        if not (d / "node_x.npy").exists():
            p1 = config.p1_outputs(cfg)
            idx, X, groups = nodefeat.build(pl.read_csv(p1 / "p2" / "node_features.csv"), pl.read_csv(p1 / "bridges.csv"), contract.TARGETS)
            nodefeat.write(idx, X, groups, d)
    idx, X, groups = nodefeat.read(d)
    return d, idx, X, groups


def family_for(run_id: str) -> str:
    p = config.REPO_ROOT / "config" / "experiments.yaml"
    if p.exists():
        for e in yaml.safe_load(p.read_text()).get("experiments", []):
            if e["id"] == run_id:
                return str(e["family"])
    return "gnn"


def main(argv=None):
    t0 = time.time()
    args, cfg = parse_args(argv)
    if args.quick:
        cfg["gnn"] = {**cfg["gnn"], **cfg["gnn"]["quick"]}
    features = [f for f in args.features.split(",") if f]
    bad = set(features) - set(ALL_FEATURES)
    if bad:
        sys.exit(f"unknown feature groups {sorted(bad)}")
    directed = args.layer == "D"
    symmetrise = directed and args.arch == "gcn"
    out = config.outputs_dir(cfg)
    p1 = config.p1_outputs(cfg)
    nf_dir, node_index, X, groups = resolve_node_features(args, cfg)
    drop = set()
    if args.drop_codes == "bridge":
        drop = nodefeat.bflag_codes(pl.read_csv(p1 / "p2" / "node_features.csv"), pl.read_csv(p1 / "bridges.csv"), args.target)
    opts = data.Options(layer=args.layer, features=features, lookback=args.lookback, expand=bool(args.expand),
                        weighted=bool(args.weighted), drop_codes=drop, directed=directed, symmetrise=symmetrise)
    fc = data.feature_columns(groups, features)
    cov_names = data.covariate_names(features)
    in_dim = fc["n_static"] + fc["n_temp"] + len(fc["flags"])
    model_kwargs = dict(arch=args.arch, in_dim=in_dim, cov_dim=len(cov_names), hidden=args.hidden, depth=args.depth,
                        dropout=float(cfg["gnn"]["dropout"]), heads=args.heads, agg=args.agg, n_nodes=node_index.height,
                        embedding_dim=int(cfg["gnn"]["embedding_dim"]), use_embedding="CODE" in features)
    grid = [{"lr": float(lr), "weight_decay": float(wd)} for lr in cfg["gnn"]["lr_grid"] for wd in cfg["gnn"]["weight_decay_grid"]]
    flags = {k: v for k, v in vars(args).items() if k not in {"dry_run", "seeds"}}
    flags.update({"features": ",".join(features), "symmetrise": symmetrise, "node_features_dir": str(nf_dir),
                  "drop_codes_n": len(drop), "quick": bool(args.quick)})
    resolved = {"flags": flags, "model_kwargs": model_kwargs, "grid": grid, "cov_names": cov_names, "seeds": args.seeds,
                "family": family_for(args.run_id), "gnn_cfg": {k: v for k, v in cfg["gnn"].items() if k != "quick"}}
    if args.dry_run:
        print(json.dumps(resolved, indent=1, default=str))
        return 0

    from bridgegnn.gnn import train  # torch import deferred so --dry-run and --help stay instant

    rows_all = contract.read(out / "rows.parquet", "rows")
    rc = contract.read(out / "row_codes.parquet", "row_codes")
    rows_t = rows_all.filter(pl.col("target") == args.target)
    g_layer, _ = load_layer(args, cfg)
    exs = data.build_examples(rows_t, rc, args.target, node_index, X, groups, g_layer, opts, cfg)
    dev_all = [e for e in exs if e.split in cfg["split"]["dev"]]
    test = [e for e in exs if e.split in cfg["split"]["test"]]
    log = out / "logs" / f"06_{args.run_id}_{args.target}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), **resolved}, default=str) + "\n")
    print(f"{args.run_id} {args.target} {args.arch} layer {args.layer}: {len(dev_all):,} dev rows, {len(test):,} test rows, "
          f"node features {nf_dir.name} ({X.shape[1]} cols), in_dim {in_dim}, cov {len(cov_names)}, grid {len(grid)}  ({time.time()-t0:.0f}s)", flush=True)

    def run_seeds(dev, seeds):
        return train.run(args.run_id, args.target, dev, test, grid, seeds, cfg, model_kwargs, flags, out, cov_names, family=resolved["family"])

    results = []
    if args.subsample_file:
        for s in args.seeds:
            ids = {int(x) for x in Path(args.subsample_file.format(seed=s)).read_text().split()}
            dev = [e for e in dev_all if e.person_id in ids]
            results += run_seeds(dev, [s])
    else:
        results = run_seeds(dev_all, args.seeds)
    for r in results:
        line = {k: r[k] for k in ["run_id", "target", "seed", "dev_oof_auprc", "test_auroc", "test_auprc", "test_brier", "n_test_pos"]}
        print(json.dumps(line, default=float), flush=True)
        with open(log, "a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
