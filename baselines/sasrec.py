"""
SASRec-inspired baseline for ψ prediction.

Adapted from Kang & McAuley 2018 "Self-Attentive Sequential Recommendation"
(SASRec) for our ψ-regression task. The original predicts next-item
probability via causal self-attention over the user's history; we predict
the per-item ψᵢ via the same causal-attention encoder followed by a
regression head on the (history, candidate-item) pair.

Architecture (default):
  Embedding(item) → 2-layer causal self-attention → linear head → sigmoid×2 (ψ ∈ [0, 2])

Per mp_07 §11 hyperparameter search space.

Runs on CPU (slow), MPS (~5x slower than A100), or CUDA. ~10–30 min on H100
for KuaiRand-Pure 3 splits × 3 seeds; ~1–3 hours for full Tenrec QK-article.
"""

from __future__ import annotations

import argparse
import json
import math
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

from corpus_build.splits import items_split, time_split, users_split


def _device_select() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SASRecEncoder(nn.Module):
    """Causal self-attention encoder over a user's history sequence."""

    def __init__(self, n_items: int, hidden: int = 64, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.2, max_len: int = 50):
        super().__init__()
        self.item_embed = nn.Embedding(n_items + 1, hidden, padding_idx=0)  # 0 = PAD
        self.pos_embed = nn.Embedding(max_len, hidden)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.max_len = max_len
        self.hidden = hidden

    def forward(self, hist_idx: torch.Tensor) -> torch.Tensor:
        """hist_idx: [B, L] long. Returns [B, hidden] (last-position output)."""
        B, L = hist_idx.shape
        pos = torch.arange(L, device=hist_idx.device).unsqueeze(0).expand(B, L)
        x = self.item_embed(hist_idx) + self.pos_embed(pos)
        x = self.dropout(x)
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=hist_idx.device), diagonal=1).bool()
        x = self.encoder(x, mask=mask)
        # Take the output at the last non-pad position. Easier: take last position.
        return x[:, -1, :]


class SASRecPsiHead(nn.Module):
    """Combines history encoding + candidate item embedding → ψ scalar."""

    def __init__(self, n_items: int, hidden: int = 64, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.2, max_len: int = 50):
        super().__init__()
        self.encoder = SASRecEncoder(n_items, hidden, n_layers, n_heads, dropout, max_len)
        self.candidate_embed = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, hist_idx, cand_idx):
        h = self.encoder(hist_idx)
        c = self.candidate_embed(cand_idx)
        z = torch.cat([h, c], dim=-1)
        return self.head(z).squeeze(-1) * 2.0


def _build_history_sequences(df: pd.DataFrame, max_len: int = 50,
                              item_vocab: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    For each row (u, i, t), build the (history, candidate, ψ) tuple.
    History = the user's last max_len items consumed BEFORE this position.
    """
    df = df.sort_values(["user_id", "position"]).reset_index(drop=True)
    if item_vocab is None:
        item_vocab = {i: idx + 1 for idx, i in enumerate(df["item_id"].unique())}  # 0 reserved for pad
    n = len(df)

    hist = np.zeros((n, max_len), dtype=np.int64)
    cand = np.zeros(n, dtype=np.int64)
    y = df["psi"].to_numpy().astype(np.float32)

    user_ids = df["user_id"].to_numpy()
    item_ids = df["item_id"].to_numpy()

    # Group bounds
    bounds = [0]
    for i in range(1, n):
        if user_ids[i] != user_ids[i - 1]:
            bounds.append(i)
    bounds.append(n)

    for g in range(len(bounds) - 1):
        s, e = bounds[g], bounds[g + 1]
        for k in range(s, e):
            cand[k] = item_vocab.get(item_ids[k], 0)
            history_start = max(s, k - max_len)
            history_items = item_ids[history_start:k]
            history_idx = np.array([item_vocab.get(it, 0) for it in history_items], dtype=np.int64)
            # Pad on the LEFT so the most recent item is at position max_len-1
            hist[k, max_len - len(history_idx):] = history_idx

    return hist, cand, y, item_vocab


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    if y_pred.std() < 1e-12:
        sp = float("nan")
    else:
        try:
            sp = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
        except Exception:
            sp = float("nan")
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "spearman": sp,
    }


def run(shard_path: Path | str, split: str = "items", seed: int = 0,
        device: str | None = None, epochs: int = 5, batch_size: int = 256,
        hidden: int = 64, n_layers: int = 2, n_heads: int = 2,
        dropout: float = 0.2, max_len: int = 50, lr: float = 1e-3,
        max_train: int | None = 1_000_000,  # cap train rows for speed; raise for full
        **kwargs) -> dict:
    if not HAS_TORCH:
        return {"error": "PyTorch not installed", "shard": str(shard_path), "split": split, "seed": seed}

    shard_path = Path(shard_path)
    df = pd.read_parquet(shard_path)
    if device is None:
        device = _device_select()

    split_fn = {"items": items_split, "users": users_split, "time": time_split}[split]
    mask = split_fn(df)
    train_df = df[mask == "train"].reset_index(drop=True)
    test_df = df[mask == "test"].reset_index(drop=True)

    if len(test_df) < 100 or len(train_df) < 100:
        return {"shard": str(shard_path), "split": split, "seed": seed,
                "n_test": int(len(test_df)), "skipped": "split too small"}

    if max_train and len(train_df) > max_train:
        train_df = train_df.sample(max_train, random_state=seed).reset_index(drop=True)

    full = pd.concat([train_df.assign(_split="train"), test_df.assign(_split="test")],
                     ignore_index=True)
    hist, cand, y, item_vocab = _build_history_sequences(full, max_len=max_len)
    n_items = len(item_vocab)

    train_mask = (full["_split"] == "train").to_numpy()
    test_mask = (full["_split"] == "test").to_numpy()
    hist_train, cand_train, y_train = hist[train_mask], cand[train_mask], y[train_mask]
    hist_test, cand_test, y_test = hist[test_mask], cand[test_mask], y[test_mask]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SASRecPsiHead(n_items=n_items, hidden=hidden, n_layers=n_layers,
                           n_heads=n_heads, dropout=dropout, max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(y_train)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n_train)
        total = 0.0
        for s in range(0, n_train, batch_size):
            b = idx[s:s + batch_size]
            h_b = torch.tensor(hist_train[b], device=device)
            c_b = torch.tensor(cand_train[b], device=device)
            y_b = torch.tensor(y_train[b], device=device)
            pred = model(h_b, c_b)
            loss = ((pred - y_b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        # Optional: print epoch loss

    model.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, len(y_test), batch_size):
            h_b = torch.tensor(hist_test[s:s + batch_size], device=device)
            c_b = torch.tensor(cand_test[s:s + batch_size], device=device)
            preds.append(model(h_b, c_b).cpu().numpy())
        y_pred = np.concatenate(preds)

    m = metrics(y_test, y_pred)
    m.update({
        "shard": str(shard_path), "split": split, "seed": seed,
        "epochs": epochs, "batch_size": batch_size,
        "hidden": hidden, "n_layers": n_layers, "n_heads": n_heads,
        "dropout": dropout, "max_len": max_len, "lr": lr,
        "max_train_used": int(min(len(train_df), max_train) if max_train else len(train_df)),
        "n_items": n_items, "device": device,
        "method": "sasrec_inspired",
    })
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--split", choices=["items", "users", "time"], default="items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--max-train", type=int, default=1_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = run(args.shard, split=args.split, seed=args.seed,
              epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden,
              max_len=args.max_len, max_train=args.max_train, lr=args.lr)
    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
