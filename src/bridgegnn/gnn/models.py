"""SubgraphGNN: GCN / GraphSAGE / GAT over a patient subgraph, mean+max readout, covariate head.
HeteroSubgraphGNN: the bipartite patient--code variant (experiment B2), readout from the patient node."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool, global_mean_pool

from .data import Example


def to_pyg(examples: list[Example]) -> list[Data]:
    out = []
    for i, e in enumerate(examples):
        out.append(Data(x=torch.from_numpy(e.x), edge_index=torch.from_numpy(e.edge_index),
                        edge_weight=torch.from_numpy(e.edge_weight), node_ids=torch.from_numpy(e.node_ids),
                        cov=torch.from_numpy(e.cov).reshape(1, -1), y=torch.tensor(float(e.y)), row_pos=torch.tensor(i)))
    return out


class SubgraphGNN(nn.Module):
    def __init__(self, arch: str, in_dim: int, cov_dim: int, hidden: int, depth: int, dropout: float, heads: int,
                 agg: str, n_nodes: int, embedding_dim: int, use_embedding: bool):
        super().__init__()
        assert arch in {"gcn", "sage", "gat"}
        self.arch, self.dropout, self.cov_dim = arch, dropout, cov_dim
        self.embedding = nn.Embedding(n_nodes + 1, embedding_dim) if use_embedding else None   # index 0 = virtual
        d_in = in_dim + (embedding_dim if use_embedding else 0)
        self.convs = nn.ModuleList()
        for layer in range(depth):
            d = d_in if layer == 0 else hidden
            if arch == "gcn":
                self.convs.append(GCNConv(d, hidden, add_self_loops=True, normalize=True))
            elif arch == "sage":
                self.convs.append(SAGEConv(d, hidden, aggr=agg))
            else:
                assert hidden % heads == 0, "hidden must be divisible by heads"
                self.convs.append(GATConv(d, hidden // heads, heads=heads, concat=True, dropout=dropout, add_self_loops=True))
        self.head = nn.Sequential(nn.Linear(2 * hidden + cov_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_weight, node_ids, batch, cov):
        h = x
        if self.embedding is not None:
            h = torch.cat([h, self.embedding(node_ids + 1)], dim=1)
        for conv in self.convs:
            if self.arch == "gcn":
                h = conv(h, edge_index, edge_weight)
            else:
                h = conv(h, edge_index)
            h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        if self.cov_dim:
            g = torch.cat([g, cov.reshape(g.shape[0], -1)], dim=1)
        return self.head(g).squeeze(-1)


class HeteroSubgraphGNN(nn.Module):
    """Two node types per row: the patient (initial state from the covariates, or a learned vector without them) and
    its codes (node features + CODE embedding). Each layer applies relation-specific SAGE-style messages, computed
    directly on the batched homogeneous graph: first the codes update, code<-code over the Layer U edges (SAGEConv,
    mean) plus code<-patient (the patient state broadcast to its codes, summed into the same update); then the patient
    updates from its own state plus patient<-code (the mean of the *updated* code states). The within-layer order means
    a depth-d patient readout sees d rounds of code-code passing, matching the homogeneous models' receptive field,
    and no parameter is dead. The prediction is read from the patient node alone, so covariates enter only through it.
    Patients without eligible codes link to the single virtual code node, as in the homogeneous models; `--agg` changes
    only the code<-code aggregation (patient<-code is always the mean)."""

    def __init__(self, arch: str, in_dim: int, cov_dim: int, hidden: int, depth: int, dropout: float, heads: int,
                 agg: str, n_nodes: int, embedding_dim: int, use_embedding: bool):
        super().__init__()
        assert arch == "hetero"
        self.arch, self.dropout, self.cov_dim = arch, dropout, cov_dim
        self.embedding = nn.Embedding(n_nodes + 1, embedding_dim) if use_embedding else None
        d_in = in_dim + (embedding_dim if use_embedding else 0)
        self.patient_in = nn.Linear(cov_dim, hidden) if cov_dim else None
        self.patient_init = nn.Parameter(torch.zeros(hidden)) if not cov_dim else None
        self.code_code = nn.ModuleList()
        self.code_from_patient = nn.ModuleList()
        self.patient_self = nn.ModuleList()
        self.patient_from_code = nn.ModuleList()
        for layer in range(depth):
            d = d_in if layer == 0 else hidden
            self.code_code.append(SAGEConv(d, hidden, aggr=agg))
            self.code_from_patient.append(nn.Linear(hidden, hidden, bias=False))
            self.patient_self.append(nn.Linear(hidden, hidden))
            self.patient_from_code.append(nn.Linear(hidden, hidden, bias=False))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_weight, node_ids, batch, cov):
        n_graphs = cov.shape[0]   # the batched covariate tensor is always [n_graphs, cov_dim], even when cov_dim == 0
        h = x
        if self.embedding is not None:
            h = torch.cat([h, self.embedding(node_ids + 1)], dim=1)
        if self.patient_in is not None:
            p = self.patient_in(cov.reshape(n_graphs, -1))
        else:
            p = self.patient_init.unsqueeze(0).expand(n_graphs, -1)
        for cc, cp, ps, pc in zip(self.code_code, self.code_from_patient, self.patient_self, self.patient_from_code):
            h = F.dropout(F.relu(cc(h, edge_index) + cp(p)[batch]), p=self.dropout, training=self.training)
            p = F.dropout(F.relu(ps(p) + pc(global_mean_pool(h, batch, size=n_graphs))), p=self.dropout, training=self.training)
        return self.head(p).squeeze(-1)


def build_model(**kwargs) -> nn.Module:
    """Model for `arch`: gcn / sage / gat -> SubgraphGNN; hetero -> HeteroSubgraphGNN (experiment B2)."""
    arch = kwargs["arch"]
    assert arch in {"gcn", "sage", "gat", "hetero"}, f"unknown arch {arch!r}"
    return HeteroSubgraphGNN(**kwargs) if arch == "hetero" else SubgraphGNN(**kwargs)
