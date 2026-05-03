"""
F2 with richer object-side features on KuaiRand-Pure.

Original F2 used only S(i), log(1+S(i)), S(i)^2 as content features
and got 8.7% relative MAE-reduction over the global mean — under
the 10% threshold. Here we add richer features available in the
KuaiRand-Pure schema:
  - tag (one-hot of single tag, ~55 levels)
  - log10(video_duration_ms)
  - upload_type (one-hot, ~5 levels)
  - server_width, server_height (resolution)
  - music_type (categorical)

Selection rule: pre-registered F2 threshold is relative MAE-reduction
≥ 10% over global mean on cold-start items.

Cold-start split: items with ≥30 encounters → train; items with
≤5 encounters → test (cold-start).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
KR_DIR = ROOT / "datasets" / "KuaiRand-Pure" / "data"
PARQUET = ROOT / "corpus" / "spine" / "kuairand_pure_S3.parquet"
OUT = ROOT / "baselines" / "F2_richer_features_kuairand_pure.json"

WARM_MIN = 30
COLD_MAX = 5
RNG_SEED = 0


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("Loading KR-Pure parquet (with ψ_S3)…")
    df = pd.read_parquet(PARQUET, columns=["item_id", "psi_S3"])
    psi_per_item = df.groupby("item_id")["psi_S3"].mean()
    n_per_item = df.groupby("item_id").size()
    print(f"items: {len(psi_per_item):,}")

    print("Loading video features…")
    vf = pd.read_csv(KR_DIR / "video_features_basic_pure.csv")
    vf["item_id"] = "kr_" + vf["video_id"].astype(str)
    vf = vf.set_index("item_id")
    vf["log_duration"] = np.log10(vf["video_duration"].clip(lower=1))
    vf["log_width"] = np.log10(vf["server_width"].clip(lower=1))
    vf["log_height"] = np.log10(vf["server_height"].clip(lower=1))

    # Align
    items_in_log = psi_per_item.index
    items_with_meta = items_in_log.intersection(vf.index)
    psi_per_item = psi_per_item.loc[items_with_meta]
    n_per_item = n_per_item.loc[items_with_meta]
    vf_sub = vf.loc[items_with_meta]
    print(f"items with metadata: {len(vf_sub):,}")

    # Cold/warm split
    warm_mask = n_per_item >= WARM_MIN
    cold_mask = n_per_item <= COLD_MAX
    warm_items = psi_per_item[warm_mask].index
    cold_items = psi_per_item[cold_mask].index
    print(f"warm (≥{WARM_MIN} enc): {len(warm_items):,}; cold (≤{COLD_MAX} enc): {len(cold_items):,}")

    if len(warm_items) < 100 or len(cold_items) < 100:
        out = {"test": "F2_richer_features", "skipped": "insufficient items"}
        OUT.write_text(json.dumps(out, indent=2))
        return

    # Build feature matrix on full warm+cold population
    cat_cols = ["tag", "upload_type", "video_type", "music_type"]
    num_cols = ["log_duration", "log_width", "log_height"]
    combined = vf_sub.loc[list(warm_items) + list(cold_items)]
    # one-hot encode categoricals (handle NaN as separate category)
    combined[cat_cols] = combined[cat_cols].fillna("__missing__").astype(str)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(combined[cat_cols])
    X_num = combined[num_cols].fillna(0.0).to_numpy(dtype=float)
    X = np.hstack([X_num, X_cat])
    y = pd.concat([psi_per_item.loc[warm_items], psi_per_item.loc[cold_items]]).to_numpy(dtype=float)
    train_mask = np.array([True]*len(warm_items) + [False]*len(cold_items))
    test_mask = ~train_mask

    print(f"Feature matrix: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  numeric features: {num_cols}")
    print(f"  categorical features (one-hot): {cat_cols} → {X_cat.shape[1]} dims")

    # Fit Ridge on warm; evaluate MAE on cold
    model = Ridge(alpha=1.0)
    model.fit(X[train_mask], y[train_mask])
    pred_cold = model.predict(X[test_mask])
    actual_cold = y[test_mask]

    mae_pred = float(np.mean(np.abs(pred_cold - actual_cold)))
    global_mean_train = float(y[train_mask].mean())
    mae_baseline = float(np.mean(np.abs(global_mean_train - actual_cold)))
    relative_reduction = (mae_baseline - mae_pred) / mae_baseline

    # Spearman ρ between prediction and actual on cold
    from scipy.stats import spearmanr
    rho, p = spearmanr(pred_cold, actual_cold)

    out = {
        "test": "F2_richer_features (KR-Pure)",
        "shard": str(PARQUET),
        "warm_min_encounters": WARM_MIN,
        "cold_max_encounters": COLD_MAX,
        "n_warm_train_items": int(len(warm_items)),
        "n_cold_test_items": int(len(cold_items)),
        "feature_set": {
            "numeric": num_cols,
            "categorical_onehot": cat_cols,
            "total_dims": int(X.shape[1]),
        },
        "global_mean_train": global_mean_train,
        "mae_baseline_global_mean": mae_baseline,
        "mae_predictor": mae_pred,
        "relative_mae_reduction": float(relative_reduction),
        "passes_10_percent_band": bool(relative_reduction >= 0.10),
        "spearman_rho_pred_vs_actual_cold": float(rho),
        "p_value_spearman": float(p),
        "note_vs_original_F2": (
            "Original F2 (S-polynomials only) reported 8.7% on KR-Pure. "
            "This run uses tag, upload_type, video_type, music_type, "
            "log_duration, log_width, log_height as additional features."
        ),
    }
    print(json.dumps(out, indent=2))
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
