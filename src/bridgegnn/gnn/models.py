"""SubgraphGNN: GCN / GraphSAGE / GAT over a patient subgraph, mean+max readout, covariate head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
