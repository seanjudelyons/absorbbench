"""
F2 — Content recoverability above popularity baseline. Per mp_08 §3.

A model that predicts ψᵢ from CONTENT FEATURES ALONE (no per-item engagement
statistics) should beat the global mean by ≥10% relative MAE reduction on a
cold-start items split.

Operationalisation:
  - Content features per item: K(i), and a 32-dim PCA projection of the
    one-hot tag/category encoding (or, for text, a small count-based feature).
  - Predict per-item ψᵢ (the mean over its encounters in train) using only
    content features.
  - Evaluate on cold-start items (items in test never seen in train).
  - Compare MAE to the train-set global-mean baseline.

Pre-registered threshold (mp_08 §3 F2): MAE-reduction ≥ 10% relative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus_build.splits import items_split


def run(shard_path: Path | str, seed: int = 0, **kwargs) -> dict:
    shard_path = Path(shard_path)
    df = pd.read_parquet(shard_path)

    mask = items_split(df, seed=seed)
    train_df = df[mask == "train"]
    test_df = df[mask == "test"]
    if len(test_df) < 100 or len(train_df) < 100:
        return {"shard": str(shard_path), "test": "F2_content_only",
                "skipped": "split too small"}

    # per-item ψᵢ on train and test
    psi_train = train_df.groupby("item_id")["psi"].mean()
    psi_test = test_df.groupby("item_id")["psi"].mean()

    # content features per item
    # use k and (if available) modality and dataset_source as content cues.
    item_K_train = train_df.groupby("item_id")["K"].mean()
    item_K_test = test_df.groupby("item_id")["K"].mean()

    train_items = pd.DataFrame({
        "K": item_K_train, "psi": psi_train,
    }).dropna()
    test_items = pd.DataFrame({
        "K": item_K_test, "psi": psi_test,
    }).dropna()

    if len(train_items) < 100 or len(test_items) < 100:
        return {"shard": str(shard_path), "test": "F2_content_only",
                "skipped": "after item aggregation, too few items"}

    # simple content-only model: linear regression of ψᵢ on k (and k-bin one-hot)
    # k is in bits ∈ [1, ~12]; use raw k + log(k) + k^2 as features
    X_train = np.column_stack([
        train_items["K"].to_numpy(),
        np.log1p(train_items["K"].to_numpy()),
        train_items["K"].to_numpy() ** 2,
        np.ones(len(train_items)),
    ])
    y_train = train_items["psi"].to_numpy()
    X_test = np.column_stack([
        test_items["K"].to_numpy(),
        np.log1p(test_items["K"].to_numpy()),
        test_items["K"].to_numpy() ** 2,
        np.ones(len(test_items)),
    ])
    y_test = test_items["psi"].to_numpy()

    # closed-form ridge
    A = X_train.T @ X_train + 1e-3 * np.eye(X_train.shape[1])
    b = X_train.T @ y_train
    w = np.linalg.solve(A, b)
    y_pred = X_test @ w
    y_pred = np.clip(y_pred, 0.0, 2.0)

    global_mean = float(y_train.mean())
    mae_content = float(np.mean(np.abs(y_test - y_pred)))
    mae_globalmean = float(np.mean(np.abs(y_test - global_mean)))
    rel_reduction = (mae_globalmean - mae_content) / mae_globalmean if mae_globalmean > 0 else 0.0

    rho, _ = stats.spearmanr(y_test, y_pred)

    verdict = "PASS" if rel_reduction >= 0.10 else "FAIL"
    return {
        "shard": str(shard_path),
        "test": "F2_content_only",
        "n_items_train": int(len(train_items)),
        "n_items_test_cold": int(len(test_items)),
        "mae_content_only": mae_content,
        "mae_global_mean": mae_globalmean,
        "relative_mae_reduction": float(rel_reduction),
        "spearman_rho_content_vs_actual": float(rho) if not np.isnan(rho) else None,
        "threshold": 0.10,
        "verdict": verdict,
        "feature_set": "K, log(1+K), K^2",
        "weights": dict(zip(["K", "log_K", "K_sq", "bias"], [float(x) for x in w])),
        "limitation": "Content features used: K only. Tag-distribution + body-text features deferred (would require modality-specific embedding pipelines on H100 in Phase 5).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(Path(args.shard), seed=args.seed)
    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
