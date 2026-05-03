"""
MLP tabular baseline for ψ prediction. Per baselines/MLP_DEFERRED.md.

Loads a corpus shard, builds (user_id, item_id) embeddings + MLP head,
trains on the items/users/time train split, evaluates on test.

Requires PyTorch. CPU-runnable on small shards but designed for A100 in
Plan.md Phase 5.

Run:
  python -m baselines.mlp_tabular --shard corpus/spine/kuairand_pure.parquet \
                                   --split items --epochs 20

Hyperparameters per mp_07 §11; defaults are mid-range search-space picks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def main():
    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Install via `pip install torch` and re-run.")
        print("This baseline is GPU-bound; see baselines/MLP_DEFERRED.md for the rationale.")
        return

    from corpus_build.splits import items_split, time_split, users_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--split", choices=["items", "users", "time"], default="items")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.shard)
    split_fn = {"items": items_split, "users": users_split, "time": time_split}[args.split]
    mask = split_fn(df)
    train = df[mask == "train"].reset_index(drop=True)
    val = df[mask == "val"].reset_index(drop=True)
    test = df[mask == "test"].reset_index(drop=True)

    # Build vocabularies on train
    user_vocab = {u: i for i, u in enumerate(train["user_id"].unique())}
    item_vocab = {i: j for j, i in enumerate(train["item_id"].unique())}
    n_users = len(user_vocab) + 1  # +1 for OOV
    n_items = len(item_vocab) + 1

    def encode(df_):
        u = df_["user_id"].map(user_vocab).fillna(n_users - 1).astype(np.int64).values
        i = df_["item_id"].map(item_vocab).fillna(n_items - 1).astype(np.int64).values
        y = df_["psi"].astype(np.float32).values
        return u, i, y

    u_train, i_train, y_train = encode(train)
    u_test, i_test, y_test = encode(test)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}; users={n_users:,}; items={n_items:,}; train={len(train):,}; test={len(test):,}")

    class TabularMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.u_embed = nn.Embedding(n_users, args.embed_dim)
            self.i_embed = nn.Embedding(n_items, args.embed_dim)
            self.head = nn.Sequential(
                nn.Linear(args.embed_dim * 2, args.hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, 1),
                nn.Sigmoid(),
            )

        def forward(self, u, i):
            x = torch.cat([self.u_embed(u), self.i_embed(i)], dim=-1)
            return self.head(x).squeeze(-1) * 2.0  # ψ ∈ [0, 2]

    model = TabularMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_n = len(train)
    for ep in range(args.epochs):
        idx = np.random.permutation(train_n)
        total = 0.0
        for s in range(0, train_n, args.batch_size):
            b = idx[s:s + args.batch_size]
            u_b = torch.tensor(u_train[b], device=device)
            i_b = torch.tensor(i_train[b], device=device)
            y_b = torch.tensor(y_train[b], device=device)
            pred = model(u_b, i_b)
            loss = ((pred - y_b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        print(f"  ep {ep+1}/{args.epochs}: train MSE = {total/train_n:.6f}")

    # Eval
    model.eval()
    with torch.no_grad():
        u_t = torch.tensor(u_test, device=device)
        i_t = torch.tensor(i_test, device=device)
        pred = model(u_t, i_t).cpu().numpy()

    err = y_test - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sp = pd.Series(y_test).corr(pd.Series(pred), method="spearman")

    result = {
        "shard": args.shard, "split": args.split, "seed": args.seed,
        "epochs": args.epochs, "embed_dim": args.embed_dim,
        "hidden": args.hidden, "lr": args.lr, "batch_size": args.batch_size,
        "n_test": int(len(y_test)),
        "mae": mae, "rmse": rmse, "spearman": float(sp),
    }
    print(f"\n## MLP tabular result\n{json.dumps(result, indent=2)}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
