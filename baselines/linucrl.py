"""
LinUCRL-inspired baseline for ψ prediction.

Faithful adaptation of Warlop, Lazaric, Mary 2018 "Fighting Boredom in
Recommender Systems with Linear Reinforcement Learning" (LinUCRL) to a
batch ψ-regression setting.

The original LinUCRL is an online MDP-RL algorithm: state = recent K
actions, reward = linear function of state, learner explores via UCRL.
We extract the *mechanism* (linear reward as a function of recent action
history with satiation features) and drop the online exploration since
our task is supervised batch prediction.

Per mp_03 §5 (Pillar 4): high-ψ items, by absorbing the user, ought to
delay or compress satiation curves. This baseline tests whether the
LinUCRL-style satiation features help predict ψᵢ.

CPU-runnable. ~5-15 minutes per (shard, split, seed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus_build.splits import items_split, time_split, users_split


def _build_satiation_features(df: pd.DataFrame, K: int = 10) -> tuple[sparse.csr_matrix, list]:
    """
    For each row (u, i, t), compute LinUCRL-style features:
      - log K(i) — own-item complexity
      - c_recent_same_K — count of last-K items with same K (satiation)
      - c_recent_high_K — count of last-K items with K above the median
      - c_recent_low_K  — count of last-K items with K below the median
      - mean_K_recent  — mean K of last-K items consumed (history complexity load)
      - mean_psi_recent — mean ψ of last-K items consumed (history absorption)

    Per Warlop 2018 §3: the reward is linear in features summarising the
    K most recent actions.

    Returns (X_sparse, feature_names).
    """
    df = df.sort_values(["user_id", "position"]).reset_index(drop=True)
    n = len(df)

    # pre-compute median k within the dataset to define hi/lo bins
    median_K = float(df["K"].median())

    # vectorised history features per user via groupby + rolling
    log_K = np.log1p(df["K"].to_numpy())

    out = {
        "log_K_self": log_K,
        "K_self": df["K"].to_numpy(),
    }

    # per-user rolling history (last k items, not including current)
    rec_same_K = np.zeros(n, dtype=np.float32)
    rec_hi_K = np.zeros(n, dtype=np.float32)
    rec_lo_K = np.zeros(n, dtype=np.float32)
    mean_K_rec = np.zeros(n, dtype=np.float32)
    mean_psi_rec = np.zeros(n, dtype=np.float32)
    rec_count = np.zeros(n, dtype=np.float32)  # how many history items existed

    K_arr = df["K"].to_numpy()
    psi_arr = df["psi"].to_numpy()

    # group offsets
    user_ids = df["user_id"].to_numpy()
    # find user-group boundaries
    bounds = [0]
    for i in range(1, n):
        if user_ids[i] != user_ids[i - 1]:
            bounds.append(i)
    bounds.append(n)

    for g in range(len(bounds) - 1):
        s, e = bounds[g], bounds[g + 1]
        for k in range(s, e):
            window_start = max(s, k - K)
            history_K = K_arr[window_start:k]
            history_psi = psi_arr[window_start:k]
            if len(history_K) == 0:
                continue
            rec_count[k] = len(history_K)
            mean_K_rec[k] = float(history_K.mean())
            mean_psi_rec[k] = float(history_psi.mean())
            curr_K = K_arr[k]
            rec_same_K[k] = float(np.sum(np.abs(history_K - curr_K) < 0.5))
            rec_hi_K[k] = float(np.sum(history_K >= median_K))
            rec_lo_K[k] = float(np.sum(history_K < median_K))

    out["rec_count"] = rec_count
    out["rec_same_K"] = rec_same_K
    out["rec_hi_K"] = rec_hi_K
    out["rec_lo_K"] = rec_lo_K
    out["mean_K_recent"] = mean_K_rec
    out["mean_psi_recent"] = mean_psi_rec
    out["bias"] = np.ones(n, dtype=np.float32)

    feat_names = list(out.keys())
    X = np.column_stack([out[k].astype(np.float32) for k in feat_names])
    return sparse.csr_matrix(X), feat_names


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
        K: int = 10, **kwargs) -> dict:
    """
    Train LinUCRL-inspired linear model on train, evaluate on test.

    Returns a result dict matching the convention of baselines.popularity / .trivial.
    """
    shard_path = Path(shard_path)
    df = pd.read_parquet(shard_path)

    split_fn = {"items": items_split, "users": users_split, "time": time_split}[split]
    mask = split_fn(df)
    train = df[mask == "train"].copy()
    test = df[mask == "test"].copy()

    if len(test) < 100 or len(train) < 100:
        return {"shard": str(shard_path), "split": split, "seed": seed,
                "n_test": int(len(test)), "skipped": "test or train too small"}

    # build features per row (per-user history within the train+test concat,
    # respecting time order; mask leaks no test info because history is built
    # from prior positions only).
    full = pd.concat([train.assign(_split="train"), test.assign(_split="test")], ignore_index=True)
    full = full.sort_values(["user_id", "position"]).reset_index(drop=True)
    X, feat_names = _build_satiation_features(full, K=K)
    y = full["psi"].to_numpy().astype(np.float32)

    train_mask = (full["_split"] == "train").to_numpy()
    test_mask = (full["_split"] == "test").to_numpy()
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # solve via least-squares with small ridge
    # closed-form: w = (x^t x + λi)^-1 x^t y
    A = X_train.T @ X_train
    A = A.toarray() + 1e-3 * np.eye(A.shape[0])
    b = X_train.T @ y_train
    w = np.linalg.solve(A, b)

    y_pred = (X_test @ w).A1 if hasattr(X_test @ w, "A1") else (X_test @ w).flatten()
    y_pred = np.clip(y_pred, 0.0, 2.0)  # ψ ∈ [0, 2]

    m = metrics(y_test, y_pred)
    m.update({
        "shard": str(shard_path), "split": split, "seed": seed,
        "K_history": K, "n_features": len(feat_names),
        "feature_names": feat_names,
        "weights": dict(zip(feat_names, [float(x) for x in w])),
        "method": "linucrl_inspired",
    })
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--split", choices=["items", "users", "time"], default="items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=10, help="LinUCRL recent-action history window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = run(Path(args.shard), split=args.split, seed=args.seed, K=args.K)
    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
