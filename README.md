# two-tower-recsys — Two-Tower Retrieval Model

A **two-tower neural retrieval model** for recommendation (ads, content,
products): separate user and item towers map IDs into a shared embedding
space, trained with sampled softmax over **in-batch negatives**, evaluated
with recall@k on held-out interactions, and served through a **FAISS ANN
index** — the same architecture pattern as YouTube's candidate generation.

## Architecture

```
synthetic interactions (latent-factor ground truth, popularity bias)
        │
        ▼
┌──────────────┐     ┌──────────────┐
│  User tower  │     │  Item tower  │
│ Embedding→MLP│     │ Embedding→MLP│  → L2-normalized (cosine space)
└──────┬───────┘     └──────┬───────┘
       │  in-batch sampled softmax (τ=0.2, AdamW-style wd=1e-4)
       ▼
train.py ──► loss per epoch
        │
eval: recall@k on HELD-OUT (user → items) pairs
        │
index.py ──► FAISS IndexFlatIP over item embeddings → top-k serving
```

## Quickstart

```bash
pip install -r requirements.txt
python demo.py   # train + recall@k + FAISS serving demo
```

## Sample output

```
[2/4] Training two-tower model (in-batch negatives)...
  epoch 1/10  loss=4.7934
  ...
  epoch 10/10  loss=3.6339
      training time: 23.9s

[3/4] Evaluating recall@k on held-out items...
      recall@ 5 = 0.060
      recall@10 = 0.140
      recall@20 = 0.307

[4/4] Building FAISS ANN index and serving retrieval...
  FAISS index built: 300 items, dim=16
      ANN query for 5 users: 0.10 ms (0.02 ms/user)
```

Random baselines for reference: recall@10 ≈ 0.067, recall@20 ≈ 0.133 — the
model beats random by ~2.3× by recovering the latent user/item factors.

## How it works

- **Data** (`src/data.py`) — users/items live in a shared latent space;
  interaction probability ∝ dot product + popularity bias. Two positives per
  user are held out for the recall@k eval; the model never sees them.
- **Model** (`src/model.py`) — ID embedding → MLP → L2 normalization per
  tower. In-batch softmax: each (user, positive) pair competes against every
  other item in the batch as a negative.
- **Training** (`src/train.py`) — the tuned regime (soft temperature,
  small weight decay, embedding dim matched to the latent structure) is what
  lets the towers recover generalizable factors instead of memorizing pairs.
- **Serving** (`src/index.py`) — item embeddings indexed once in FAISS;
  per-user retrieval is a single ANN query (~0.02 ms/user).

## Tech stack

Python, PyTorch (two-tower, in-batch softmax), FAISS (ANN serving), NumPy.
