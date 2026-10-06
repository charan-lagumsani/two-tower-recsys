"""Training loop and recall@k evaluation for the two-tower model."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .model import TwoTower, in_batch_softmax_loss


def train(model: TwoTower, pos_loader: DataLoader,
          epochs: int = 10, lr: float = 3e-3, temperature: float = 0.2,
          weight_decay: float = 1e-4, device: str = "cpu") -> list[float]:
    """Train on positive (user, item) pairs with in-batch negatives.

    Defaults are tuned for the synthetic demo (latent_dim=8): a soft
    temperature and small weight decay keep the towers from collapsing to a
    memorization solution and let them recover the latent factors, which is
    what makes held-out recall@k beat random.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    losses = []
    for epoch in range(epochs):
        total, nb = 0.0, 0
        for user_ids, item_ids, labels in pos_loader:
            mask = labels == 1
            if mask.sum() == 0:
                continue
            user_ids, item_ids = user_ids[mask].to(device), item_ids[mask].to(device)
            u_emb, i_emb = model(user_ids, item_ids)
            loss = in_batch_softmax_loss(u_emb, i_emb, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        avg = total / max(nb, 1)
        losses.append(avg)
        print(f"  epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def recall_at_k(model: TwoTower, test_pairs: dict[int, list[int]],
                k: int = 10, device: str = "cpu",
                eval_users: int = 200) -> float:
    """Recall@k over held-out (user -> items) pairs via exact top-k."""
    model.to(device).eval()
    item_ids = torch.arange(model.n_items, device=device)
    item_emb = model.item_tower(item_ids)  # (n_items, dim)
    users = sorted(test_pairs)[:eval_users]
    hits, total = 0, 0
    for u in users:
        held = test_pairs[u]
        u_emb = model.user_tower(torch.tensor([u], device=device))
        scores = (u_emb @ item_emb.T).squeeze(0)
        topk = scores.topk(k).indices.tolist()
        hits += len(set(topk) & set(held))
        total += len(held)
    return hits / total if total else 0.0
