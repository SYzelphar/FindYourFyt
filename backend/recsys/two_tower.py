"""Two-tower retrieval model, trained offline on real H&M purchases (ml/train_two_tower.py).

Item tower:  [FashionCLIP image (512) | FashionCLIP text (512)] + categorical embeddings
             (product type, colour, pattern, colour family, index group, garment group, section)
             -> MLP -> 128-d unit vector ("taste space")
User tower:  the customer's recent items (most recent first), embedded by the item tower
             -> attention pooling with recency-rank embeddings + mean pooling -> MLP -> 128-d

Score = cos(user, item) / temperature, trained with a full softmax over the catalog to
predict the next purchase. The item tower only sees content features (no item-ID
embeddings), so it also works for items that were never bought (inductive / cold start).

At serving time the user tower runs on the session's liked items (backend/recsys/recommender.py).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ItemTower(nn.Module):
    def __init__(self, dense_dim: int, cat_cardinalities: list[int], cat_dim=32, hidden=512,
                 out_dim=128, dropout=0.1):
        super().__init__()
        self.cat_embs = nn.ModuleList(nn.Embedding(n + 1, cat_dim) for n in cat_cardinalities)  # +1 = unknown
        self.mlp = nn.Sequential(
            nn.Linear(dense_dim + cat_dim * len(cat_cardinalities), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, dense: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        parts = [dense] + [emb(cats[:, j]) for j, emb in enumerate(self.cat_embs)]
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class UserTower(nn.Module):
    def __init__(self, dim=128, max_len=20, hidden=256, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.rank_emb = nn.Embedding(max_len, dim)      # 0 = most recent item
        self.query = nn.Parameter(torch.randn(dim) / dim ** 0.5)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, hist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hist [B, L, d] item vectors (most recent first), mask [B, L] bool."""
        L = hist.shape[1]
        keys = hist + self.rank_emb.weight[:L]
        att = (keys @ self.query) / keys.shape[-1] ** 0.5
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        attended = (att.unsqueeze(-1) * hist).sum(1)
        m = mask.unsqueeze(-1).float()
        mean = (hist * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(mean + self.mlp(torch.cat([attended, mean], dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, dense_dim: int, cat_cardinalities: list[int], dim=128, max_len=20):
        super().__init__()
        self.item_tower = ItemTower(dense_dim, cat_cardinalities, out_dim=dim)
        self.user_tower = UserTower(dim=dim, max_len=max_len)
        self.log_tau = nn.Parameter(torch.tensor(-3.0))   # temperature ~0.05

    def user_vectors(self, item_vecs: torch.Tensor, hist_idx: torch.Tensor) -> torch.Tensor:
        """hist_idx [B, L] item indices, -1 = padding."""
        mask = hist_idx >= 0
        return self.user_tower(item_vecs[hist_idx.clamp(min=0)], mask)


def user_vector(user_tower: UserTower, item_vecs, liked_rows: list[int]):
    """Serving helper: liked rows oldest -> newest; returns a 128-d numpy vector."""
    rows = list(reversed(liked_rows))[: user_tower.max_len]       # most recent first
    with torch.inference_mode():
        hist = torch.as_tensor(item_vecs[rows]).unsqueeze(0)
        mask = torch.ones(1, len(rows), dtype=torch.bool)
        return user_tower(hist, mask)[0].numpy()
