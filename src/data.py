"""Synthetic user-item interaction data with latent-factor ground truth.

Users and items live in a shared latent space; interaction probability is a
function of their dot product plus popularity bias. This gives the two-tower
model a learnable structure while keeping the demo fully self-contained.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray,
                 labels: np.ndarray):
        self.user_ids = torch.from_numpy(user_ids).long()
        self.item_ids = torch.from_numpy(item_ids).long()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.user_ids[i], self.item_ids[i], self.labels[i]


def generate_interactions(n_users: int = 600, n_items: int = 300,
                          latent_dim: int = 8, pos_per_user: int = 30,
                          neg_per_pos: int = 4, seed: int = 42):
    """Return (train_ds, test_ds, n_users, n_items) with a held-out item/user split."""
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(n_users, latent_dim))
    item_factors = rng.normal(size=(n_items, latent_dim))
    item_pop = rng.normal(size=n_items) * 0.5  # popularity bias

    def sample_for_user(u: int, k: int, exclude: set[int]) -> list[int]:
        scores = user_factors[u] @ item_factors.T + item_pop
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        chosen: list[int] = []
        pool = [i for i in range(n_items) if i not in exclude]
        p = probs[pool] / probs[pool].sum()
        while len(chosen) < k and pool:
            i = rng.choice(pool, p=p)
            chosen.append(int(i))
            pool.remove(int(i))
            p = probs[pool] / probs[pool].sum() if pool else p
        return chosen

    rows: list[tuple[int, int, int]] = []
    test_pairs: dict[int, list[int]] = {}
    for u in range(n_users):
        positives = sample_for_user(u, pos_per_user, set())
        # hold out 2 positives per user for recall@k evaluation
        held = set(rng.choice(positives, size=2, replace=False).tolist())
        test_pairs[u] = sorted(held)
        train_pos = [p for p in positives if p not in held]
        for p in train_pos:
            rows.append((u, p, 1))
            negs = sample_for_user(u, neg_per_pos * 2, set(positives))
            for n in negs[:neg_per_pos]:
                rows.append((u, n, 0))

    arr = np.array(rows)
    rng.shuffle(arr)
    users, items, labels = arr[:, 0], arr[:, 1], arr[:, 2]
    train_ds = InteractionDataset(users, items, labels)
    return train_ds, test_pairs, n_users, n_items
