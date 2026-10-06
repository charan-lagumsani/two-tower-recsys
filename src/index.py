"""FAISS ANN index over item-tower embeddings for low-latency serving.

Builds an IndexFlatIP (exact cosine, since embeddings are L2-normalized)
over all item embeddings, then serves top-k retrieval for user embeddings.
Swap in IndexIVFFlat for approximate search at larger catalog sizes.
"""

from __future__ import annotations

import faiss
import numpy as np
import torch

from .model import TwoTower


class ANNIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def build(self, model: TwoTower, device: str = "cpu") -> None:
        model.eval()
        with torch.no_grad():
            item_ids = torch.arange(model.n_items, device=device)
            emb = model.item_tower(item_ids).cpu().numpy().astype(np.float32)
        self.index.add(emb)
        print(f"  FAISS index built: {self.index.ntotal} items, dim={emb.shape[1]}")

    def query(self, user_emb: np.ndarray, k: int = 10):
        """Return (item_ids, scores) for each user embedding row."""
        q = np.ascontiguousarray(user_emb, dtype=np.float32)
        scores, ids = self.index.search(q, k)
        return ids, scores

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, dim: int) -> "ANNIndex":
        idx = cls(dim)
        idx.index = faiss.read_index(path)
        return idx
