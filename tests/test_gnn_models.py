import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader

from bridgegnn.gnn import models
from bridgegnn.gnn.data import Example


def make_examples(n=8, n_nodes_total=20, f=5, cov=3, seed=0, virtual_only=False):
    rng = np.random.default_rng(seed)
    exs = []
    for i in range(n):
        if virtual_only:
            x = np.zeros((1, f), np.float32); x[0, -2] = 1
            ids = np.array([-1]); ei = np.zeros((2, 0), np.int64); ew = np.zeros(0, np.float32)
        else:
            k = int(rng.integers(2, 6))
            ids = rng.choice(n_nodes_total, size=k, replace=False).astype(np.int64)
            x = rng.normal(size=(k, f)).astype(np.float32)
            src = np.arange(k - 1); dst = np.arange(1, k)
            ei = np.array([np.r_[src, dst], np.r_[dst, src]], dtype=np.int64)
            ew = rng.uniform(0.1, 1, size=ei.shape[1]).astype(np.float32)
        exs.append(Example(x=x, edge_index=ei, edge_weight=ew, node_ids=ids, cov=rng.normal(size=cov).astype(np.float32),
                           y=int(i % 2), person_id=i, index_date=None, split="train"))
    return exs


@pytest.mark.parametrize("arch", ["gcn", "sage", "gat", "hetero"])
def test_forward_shapes_and_finite(arch):
    exs = make_examples()
    m = models.build_model(arch=arch, in_dim=5, cov_dim=3, hidden=16, depth=2, dropout=0.1, heads=4, agg="mean",
                           n_nodes=20, embedding_dim=8, use_embedding=True)
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=4)))
    out = m(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
    assert out.shape == (4,) and torch.isfinite(out).all()


def test_no_embedding_when_code_off():
    m = models.SubgraphGNN("sage", 5, 3, 16, 2, 0.1, 4, "mean", 20, 8, use_embedding=False)
    assert m.embedding is None
    exs = make_examples()
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=8)))
    assert m(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov).shape == (8,)


def test_sage_max_and_gat_heads_configurable():
    exs = make_examples()
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=8)))
    for m in [models.SubgraphGNN("sage", 5, 3, 16, 3, 0.1, 4, "max", 20, 8, True),
              models.SubgraphGNN("gat", 5, 3, 16, 2, 0.1, 2, "mean", 20, 8, True),
              models.SubgraphGNN("gat", 5, 3, 16, 2, 0.1, 8, "mean", 20, 8, True)]:
        assert torch.isfinite(m(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)).all()


def test_batch_of_virtual_only_rows_runs():
    exs = make_examples(virtual_only=True)
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=8)))
    for arch in ["gcn", "sage", "gat"]:
        m = models.SubgraphGNN(arch, 5, 3, 16, 2, 0.1, 4, "mean", 20, 8, True)
        assert torch.isfinite(m(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)).all()


def test_no_covariates_and_zero_cov_dim():
    exs = [Example(e.x, e.edge_index, e.edge_weight, e.node_ids, np.zeros(0, np.float32), e.y, e.person_id, None, "train") for e in make_examples()]
    m = models.SubgraphGNN("gcn", 5, 0, 16, 2, 0.1, 4, "mean", 20, 8, True)
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=8)))
    assert m(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov).shape == (8,)


def test_to_pyg_keeps_row_positions_and_labels():
    exs = make_examples()
    ds = models.to_pyg(exs)
    assert [int(d.row_pos) for d in ds] == list(range(len(exs))) and [float(d.y) for d in ds] == [float(e.y) for e in exs]
    assert ds[0].cov.shape == (1, 3)


def _kw(arch="hetero", f=5, cov=3, **over):
    kw = {"arch": arch, "in_dim": f, "cov_dim": cov, "hidden": 16, "depth": 2, "dropout": 0.1, "heads": 4, "agg": "mean",
          "n_nodes": 20, "embedding_dim": 4, "use_embedding": True}
    kw.update(over)
    return kw


def test_build_model_dispatch():
    assert isinstance(models.build_model(**_kw("sage")), models.SubgraphGNN)
    assert isinstance(models.build_model(**_kw("hetero")), models.HeteroSubgraphGNN)
    with pytest.raises(AssertionError):
        models.build_model(**_kw("rgcn"))


def test_hetero_virtual_only_rows():
    exs = make_examples(n=6, virtual_only=True)
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=6)))
    model = models.build_model(**_kw()).eval()
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
    assert out.shape == (6,) and torch.isfinite(out).all()


def test_hetero_without_covariates():
    exs = make_examples(n=5, cov=0)
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=5)))
    model = models.build_model(**_kw(cov=0)).eval()
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
    assert out.shape == (5,) and torch.isfinite(out).all()


def test_hetero_every_parameter_receives_gradient():
    """Codes update before the patient aggregates them, so the last layer's code update feeds the readout (no dead weights)."""
    exs = make_examples(n=6)
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=6)))
    model = models.build_model(**_kw()).train()
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
    out.sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert dead == [], dead


def test_hetero_batch_isolation_and_sensitivity():
    exs = make_examples(n=5)
    model = models.build_model(**_kw()).eval()
    batch = next(iter(DataLoader(models.to_pyg(exs), batch_size=5)))
    together = model(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)
    alone = torch.cat([model(b.x, b.edge_index, b.edge_weight, b.node_ids, b.batch, b.cov)
                       for b in DataLoader(models.to_pyg(exs), batch_size=1)])
    assert torch.allclose(together, alone, atol=1e-6)
    cov2 = batch.cov.clone(); cov2[0] += 1.0
    changed = model(batch.x, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, cov2)
    assert not torch.isclose(changed[0], together[0]) and torch.allclose(changed[1:], together[1:], atol=1e-6)
    x2 = batch.x.clone(); x2[0] += 1.0
    assert not torch.isclose(model(x2, batch.edge_index, batch.edge_weight, batch.node_ids, batch.batch, batch.cov)[0], together[0])
