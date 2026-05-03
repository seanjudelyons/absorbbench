"""
Canonical F4 — Cross-slice persistence on KuaiRand-Pure using the
real is_rand indicator.

Re-merges is_rand from the source CSVs (log_random_*.csv → 1,
log_standard_*.csv → 0) into the released parquet via (user_id, video_id,
time_ms) join, splits per-item ψᵢ across slices, and reports Pearson r and
Spearman ρ on items with ≥10 encounters per slice.

Pre-registered threshold (mp_08 §3 F4): Pearson r ≥ 0.6.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KR_DIR = ROOT / "datasets" / "KuaiRand-Pure" / "data"
PARQUET = ROOT / "corpus" / "spine" / "kuairand_pure.parquet"
OUT = ROOT / "baselines" / "falsifiability_F4_canonical_kuairand_pure.json"

MIN_ENC = 10


def load_with_isrand() -> pd.DataFrame:
    """Re-merge is_rand from source CSVs onto the released ψ values."""
    standard_files = [
        KR_DIR / "log_standard_4_08_to_4_21_pure.csv",
        KR_DIR / "log_standard_4_22_to_5_08_pure.csv",
    ]
    random_files = [KR_DIR / "log_random_4_22_to_5_08_pure.csv"]
    keep = ["user_id", "video_id", "time_ms", "is_rand"]
    chunks = []
    for p in standard_files + random_files:
        df = pd.read_csv(p, usecols=keep)
        chunks.append(df)
    src = pd.concat(chunks, ignore_index=True)

    # Source schema: user_id (int), video_id (int), time_ms (int).
    # Released parquet schema: user_id="kr_<int>", item_id="kr_<int>",
    # timestamp=datetime64[ns]. Reconstruct join keys.
    par = pd.read_parquet(PARQUET, columns=["user_id", "item_id", "timestamp", "psi"])
    par["uid_int"] = par["user_id"].str.removeprefix("kr_").astype("int64")
    par["vid_int"] = par["item_id"].str.removeprefix("kr_").astype("int64")
    # Released parquet stores timestamp at datetime64[ms] resolution, so
    # astype("int64") already yields milliseconds since epoch.
    par["time_ms"] = par["timestamp"].astype("int64")

    # Sanity: source time_ms is unix-ms; verify scale.
    src["time_ms"] = src["time_ms"].astype("int64")

    merged = par.merge(
        src.rename(columns={"user_id": "uid_int", "video_id": "vid_int"}),
        on=["uid_int", "vid_int", "time_ms"],
        how="left",
        validate="many_to_many",
    )
    return merged


def f4_canonical(merged: pd.DataFrame) -> dict:
    matched = merged["is_rand"].notna().sum()
    total = len(merged)
    if matched / total < 0.50:
        return {
            "test": "F4_canonical",
            "skipped": f"only {matched}/{total} ({100*matched/total:.1f}%) released rows matched is_rand on (uid, vid, time_ms); join failed",
        }

    # is_rand: 1 = random-exposure slice, 0 = ranker-served slice.
    rand = merged[merged["is_rand"] == 1]
    serv = merged[merged["is_rand"] == 0]

    psi_rand = rand.groupby("item_id")["psi"].mean()
    psi_serv = serv.groupby("item_id")["psi"].mean()
    n_rand = rand.groupby("item_id").size()
    n_serv = serv.groupby("item_id").size()

    keep = (n_rand[n_rand >= MIN_ENC].index
            .intersection(n_serv[n_serv >= MIN_ENC].index))
    if len(keep) < 30:
        return {
            "test": "F4_canonical",
            "matched_rows": int(matched),
            "n_items_random_ge_10": int((n_rand >= MIN_ENC).sum()),
            "n_items_served_ge_10": int((n_serv >= MIN_ENC).sum()),
            "n_items_intersection": int(len(keep)),
            "skipped": f"<30 items meet per-slice threshold (got {len(keep)})",
        }

    a = psi_rand.loc[keep].to_numpy(dtype=float)
    b = psi_serv.loc[keep].to_numpy(dtype=float)
    pearson_r, p_p = stats.pearsonr(a, b)
    rho, p_r = stats.spearmanr(a, b)

    verdict = "PASS" if pearson_r >= 0.6 else "FAIL"
    return {
        "test": "F4_canonical (is_rand merged from source CSVs)",
        "shard": str(PARQUET),
        "matched_rows": int(matched),
        "matched_fraction": float(matched / total),
        "rows_random_slice": int(len(rand)),
        "rows_served_slice": int(len(serv)),
        "min_encounters_per_slice": MIN_ENC,
        "n_items_random_ge_10": int((n_rand >= MIN_ENC).sum()),
        "n_items_served_ge_10": int((n_serv >= MIN_ENC).sum()),
        "n_items_intersection": int(len(keep)),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(rho),
        "p_value_pearson": float(p_p),
        "threshold": 0.6,
        "verdict": verdict,
        "psi_rand_mean": float(np.mean(a)),
        "psi_rand_std": float(np.std(a)),
        "psi_serv_mean": float(np.mean(b)),
        "psi_serv_std": float(np.std(b)),
    }


def main():
    print(f"Loading parquet + re-merging is_rand from {KR_DIR}…")
    merged = load_with_isrand()
    print(f"merged shape: {merged.shape}; is_rand non-null: {merged['is_rand'].notna().sum()}")
    res = f4_canonical(merged)
    print(json.dumps(res, indent=2, default=str))
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
