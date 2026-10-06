"""Two-tower retrieval model with in-batch negatives.

User tower and item tower each map an ID embedding through an MLP to a
shared embedding space. Training uses sampled softmax with in-batch
negatives: for a batch of (user, positive_item) pairs, every other item in
the batch serves as a negative — the standard efficient retrieval-training
setup from "Deep Neural Networks for YouTube Recommendations".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, n_ids: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(n_ids, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.embedding(ids))
        return F.normalize(x, p=2, dim=-1)  # cosine space


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, out_dim: int = 64):
        super().__init__()
        self.user_tower = Tower(n_users, out_dim=out_dim)
        self.item_tower = Tower(n_items, out_dim=out_dim)
        self.n_items = n_items

    def forward(self, user_ids, item_ids):
        return self.user_tower(user_ids), self.item_tower(item_ids)


def in_batch_softmax_loss(user_emb: torch.Tensor, item_emb: torch.Tensor,
                          temperature: float = 0.07) -> torch.Tensor:
    """Sampled softmax: diagonal (true pairs) vs all in-batch items."""
    logits = (user_emb @ item_emb.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)
