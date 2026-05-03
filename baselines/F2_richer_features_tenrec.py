"""
F2 with richer object-side features on Tenrec QK-article.

Original F2 used only S(i), log(1+S(i)), S(i)^2 and got 4.5%
relative MAE-reduction — well under the 10% threshold. This uses
all available object-side features:
  - category_first, category_second (one-hot)
  - item_score1, item_score2, item_score3 (numeric, low-cardinality)

Cold-start split: items with ≥10 encounters → warm (train);
items with ≤5 encounters → cold (test).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "tenrec" / "QK-article.csv"
OUT = ROOT / "baselines" / "F2_richer_features_tenrec_qk_article.json"

WARM_MIN = 10
COLD_MAX = 5
RNG_SEED = 0


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("Loading Tenrec QK-article…")
    cols = ["user_id", "item_id", "category_first", "category_second",
            "item_score1", "item_score2", "item_score3", "read",
            "read_percentage", "share", "like", "follow", "favorite"]
    df = pd.read_csv(SRC, usecols=cols)
    print(f"rows: {len(df):,}")

    # Compute released ψ
    print("Computing per-encounter ψ (released form)…")
    df["c"] = np.where(df["read"].astype(bool),
                       df["read_percentage"].fillna(0).astype(float) / 100.0, 0.0).clip(0, 1)
    refl = ["share", "like", "follow", "favorite"]
    df[refl] = df[refl].astype(bool).astype(int)
    df["R_count"] = df[refl].sum(axis=1)
    pos = df["R_count"][df["R_count"] > 0]
    p90 = float(np.percentile(pos, 90)) if len(pos) > 0 else 1.0
    df["R_norm"] = (df["R_count"].clip(upper=p90) / max(p90, 1)).astype(float)
    # cat_pair S
    df["cat_pair"] = df["category_first"].astype(str) + "_" + df["category_second"].astype(str)
    p_cat = df["cat_pair"].value_counts(normalize=True).to_dict()
    s_map = {k: -np.log2(v) for k, v in p_cat.items()}
    df["S_raw"] = df["cat_pair"].map(s_map)
    df["S"] = np.maximum(df["S_raw"].fillna(0.0), 1.0)
    df["psi"] = df["c"] * (1.0 + df["R_norm"]) / df["S"]

    print("Aggregating per-item…")
    item_grp = df.groupby("item_id")
    psi_per_item = item_grp["psi"].mean()
    n_per_item = item_grp.size()
    # per-item metadata (first row's values)
    meta = df.drop_duplicates("item_id").set_index("item_id")[
        ["category_first", "category_second", "item_score1", "item_score2", "item_score3"]
    ]

    items_aligned = psi_per_item.index.intersection(meta.index)
    psi_per_item = psi_per_item.loc[items_aligned]
    n_per_item = n_per_item.loc[items_aligned]
    meta = meta.loc[items_aligned]
    print(f"items with metadata: {len(meta):,}")

    warm_mask = n_per_item >= WARM_MIN
    cold_mask = n_per_item <= COLD_MAX
    warm_items = psi_per_item[warm_mask].index
    cold_items = psi_per_item[cold_mask].index
    print(f"warm (≥{WARM_MIN} enc): {len(warm_items):,}; cold (≤{COLD_MAX} enc): {len(cold_items):,}")

    if len(warm_items) < 100 or len(cold_items) < 100:
        out = {"test": "F2_richer_features_tenrec", "skipped": "insufficient items"}
        OUT.write_text(json.dumps(out, indent=2))
        return

    # Build feature matrix
    cat_cols = ["category_first", "category_second"]
    num_cols = ["item_score1", "item_score2", "item_score3"]
    combined = pd.concat([meta.loc[warm_items], meta.loc[cold_items]], axis=0)
    combined[cat_cols] = combined[cat_cols].fillna(-1).astype(int).astype(str)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(combined[cat_cols])
    X_num = combined[num_cols].fillna(0.0).to_numpy(dtype=float)
    X = np.hstack([X_num, X_cat])
    y = pd.concat([psi_per_item.loc[warm_items], psi_per_item.loc[cold_items]]).to_numpy(dtype=float)
    train_mask = np.array([True]*len(warm_items) + [False]*len(cold_items))
    test_mask = ~train_mask

    print(f"X.shape={X.shape}; numeric={num_cols}; cat one-hot dims={X_cat.shape[1]}")

    model = Ridge(alpha=1.0)
    model.fit(X[train_mask], y[train_mask])
    pred_cold = model.predict(X[test_mask])
    actual_cold = y[test_mask]
    mae_pred = float(np.mean(np.abs(pred_cold - actual_cold)))
    global_mean_train = float(y[train_mask].mean())
    mae_baseline = float(np.mean(np.abs(global_mean_train - actual_cold)))
    relative_reduction = (mae_baseline - mae_pred) / mae_baseline
    rho, p = spearmanr(pred_cold, actual_cold)

    out = {
        "test": "F2_richer_features (Tenrec QK-article)",
        "shard": str(SRC),
        "warm_min_encounters": WARM_MIN,
        "cold_max_encounters": COLD_MAX,
        "n_warm_train_items": int(len(warm_items)),
        "n_cold_test_items": int(len(cold_items)),
        "feature_set": {"numeric": num_cols, "categorical_onehot": cat_cols,
                        "total_dims": int(X.shape[1])},
        "mae_baseline_global_mean": mae_baseline,
        "mae_predictor": mae_pred,
        "relative_mae_reduction": float(relative_reduction),
        "passes_10_percent_band": bool(relative_reduction >= 0.10),
        "spearman_rho_pred_vs_actual_cold": float(rho),
        "p_value_spearman": float(p),
        "note_vs_original_F2": (
            "Original F2 on Tenrec (S-polynomials only): 4.5% MAE-reduction. "
            "This uses category_first, category_second (one-hot) and item_score1/2/3 (numeric)."
        ),
    }
    print(json.dumps(out, indent=2))
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
