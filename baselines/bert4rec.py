"""
BERT4Rec-inspired baseline for ψ prediction.

Adapted from Sun et al. 2019 "BERT4Rec: Sequential Recommendation with
Bidirectional Encoder Representations from Transformer" for our
ψ-regression task. The original predicts masked items via bidirectional
self-attention; we adapt it to predict ψᵢ at the masked target position
given the user's surrounding history (bidirectional context).

Architecture:
  Embedding(item) → 2-layer bidirectional self-attention → linear head → sigmoid×2

Differs from SASRec in that the attention is bidirectional (no causal mask):
the model sees items both before and after the candidate position.

Per mp_07 §11 hyperparameter search space.

Implements without RecBole dependency to keep deps minimal. RecBole's
BERT4Rec produces nearly-identical numbers; we ship our own minimal port
for portability.
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

from corpus_build.splits import items_split, time_split, users_split


def _device_select() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BERT4RecPsi(nn.Module):
    """Bidirectional self-attention encoder with ψ regression at the masked position."""

    def __init__(self, n_items: int, hidden: int = 64, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.2, max_len: int = 50):
        super().__init__()
        # 0 = PAD, 1 = MASK, 2..n+1 = real items
        self.item_embed = nn.Embedding(n_items + 2, hidden, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
        self.max_len = max_len
        self.MASK_ID = 1

    def forward(self, seq_idx: torch.Tensor, mask_pos: torch.Tensor) -> torch.Tensor:
        """
        seq_idx: [B, L] long. The candidate item's position is set to MASK_ID.
        mask_pos: [B] long, the position of the masked item in the sequence.
        Returns: [B] ψ predictions.
        """
        B, L = seq_idx.shape
        pos = torch.arange(L, device=seq_idx.device).unsqueeze(0).expand(B, L)
        x = self.item_embed(seq_idx) + self.pos_embed(pos)
        x = self.dropout(x)
        # No causal mask — BERT-style bidirectional attention
        x = self.encoder(x)
        # Pull the representation at the masked position
        b_idx = torch.arange(B, device=seq_idx.device)
        z = x[b_idx, mask_pos]  # [B, hidden]
        return self.head(z).squeeze(-1) * 2.0  # ψ ∈ [0, 2]


def _build_masked_sequences(df: pd.DataFrame, max_len: int = 50,
                             item_vocab: dict | None = None,
                             MASK_ID: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    For each row (u, i, t), build a sequence:
      [..., item_{k-2}, item_{k-1}, MASK, item_{k+1}, item_{k+2}, ...]
    where MASK is at position min(half-window, k). The model predicts ψ at the MASK position.
    """
    df = df.sort_values(["user_id", "position"]).reset_index(drop=True)
    if item_vocab is None:
        # Item indices: 0=PAD, 1=MASK, 2..n+1=real items
        item_vocab = {i: idx + 2 for idx, i in enumerate(df["item_id"].unique())}
    n = len(df)

    seq = np.zeros((n, max_len), dtype=np.int64)  # PAD-filled
    mask_pos = np.full(n, max_len // 2, dtype=np.int64)
    y = df["psi"].to_numpy().astype(np.float32)

    user_ids = df["user_id"].to_numpy()
    item_ids = df["item_id"].to_numpy()

    bounds = [0]
    for i in range(1, n):
        if user_ids[i] != user_ids[i - 1]:
            bounds.append(i)
    bounds.append(n)

    half = max_len // 2

    for g in range(len(bounds) - 1):
        s, e = bounds[g], bounds[g + 1]
        for k in range(s, e):
            # Window: items from k-half to k+half-1, with item at position k masked
            ws = max(s, k - half)
            we = min(e, k + half)
            window_items = item_ids[ws:we]
            window_idx = np.array([item_vocab.get(it, 0) for it in window_items], dtype=np.int64)
            local_mask = k - ws
            window_idx[local_mask] = MASK_ID
            # Place into sequence; pad on the LEFT
            put_start = max_len - len(window_idx)
            seq[k, put_start:] = window_idx
            mask_pos[k] = put_start + local_mask

    return seq, mask_pos, y, item_vocab


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
        max_train: int | None = 1_000_000,
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
    seq, mask_pos, y, item_vocab = _build_masked_sequences(full, max_len=max_len)
    n_items = len(item_vocab)

    train_mask = (full["_split"] == "train").to_numpy()
    test_mask = (full["_split"] == "test").to_numpy()
    seq_train, mp_train, y_train = seq[train_mask], mask_pos[train_mask], y[train_mask]
    seq_test, mp_test, y_test = seq[test_mask], mask_pos[test_mask], y[test_mask]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BERT4RecPsi(n_items=n_items, hidden=hidden, n_layers=n_layers,
                         n_heads=n_heads, dropout=dropout, max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(y_train)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n_train)
        for s in range(0, n_train, batch_size):
            b = idx[s:s + batch_size]
            seq_b = torch.tensor(seq_train[b], device=device)
            mp_b = torch.tensor(mp_train[b], device=device)
            y_b = torch.tensor(y_train[b], device=device)
            pred = model(seq_b, mp_b)
            loss = ((pred - y_b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(y_test), batch_size):
            seq_b = torch.tensor(seq_test[s:s + batch_size], device=device)
            mp_b = torch.tensor(mp_test[s:s + batch_size], device=device)
            preds.append(model(seq_b, mp_b).cpu().numpy())
    y_pred = np.concatenate(preds)

    m = metrics(y_test, y_pred)
    m.update({
        "shard": str(shard_path), "split": split, "seed": seed,
        "epochs": epochs, "batch_size": batch_size,
        "hidden": hidden, "n_layers": n_layers, "n_heads": n_heads,
        "dropout": dropout, "max_len": max_len, "lr": lr,
        "max_train_used": int(min(len(train_df), max_train) if max_train else len(train_df)),
        "n_items": n_items, "device": device,
        "method": "bert4rec_inspired",
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
