"""End-to-end demo: generate synthetic interactions, train the two-tower
model with in-batch negatives, evaluate recall@k, and serve retrieval
through a FAISS ANN index.

Usage:
    python demo.py [--users 2000] [--items 500] [--epochs 5]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import generate_interactions
from src.model import TwoTower
from src.train import train, recall_at_k
from src.index import ANNIndex

INDEX_PATH = "data/items.faiss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=600)
    ap.add_argument("--items", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(42)
    print("two-tower-recsys — retrieval model demo")

    print(f"\n[1/4] Generating synthetic interactions "
          f"({args.users} users x {args.items} items)...")
    train_ds, test_pairs, n_users, n_items = generate_interactions(
        n_users=args.users, n_items=args.items)
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    print(f"      {len(train_ds)} training pairs, "
          f"{len(test_pairs)} users with held-out items")

    print("\n[2/4] Training two-tower model (in-batch negatives)...")
    model = TwoTower(n_users, n_items, out_dim=16)
    t0 = time.time()
    train(model, loader, epochs=args.epochs)
    print(f"      training time: {time.time() - t0:.1f}s")

    print("\n[3/4] Evaluating recall@k on held-out items...")
    for k in (5, 10, 20):
        r = recall_at_k(model, test_pairs, k=k)
        print(f"      recall@{k:2d} = {r:.3f}")

    print("\n[4/4] Building FAISS ANN index and serving retrieval...")
    ann = ANNIndex(dim=16)
    ann.build(model)
    ann.save(INDEX_PATH)
    model.eval()
    with torch.no_grad():
        sample_users = torch.arange(5)
        u_emb = model.user_tower(sample_users).numpy()
    t0 = time.perf_counter()
    ids, scores = ann.query(u_emb, k=5)
    ms = (time.perf_counter() - t0) * 1000
    print(f"      ANN query for 5 users: {ms:.2f} ms "
          f"({ms / 5:.2f} ms/user)")
    print(f"      top-5 for user 0: items {ids[0].tolist()} "
          f"(scores {[f'{s:.2f}' for s in scores[0]]})")
    print(f"\nIndex saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
