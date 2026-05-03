"""
Recompute ψ on the KuaiRand-Pure spine using S3 (joint surprisal of
tag-string × log10(video_duration_ms) bucket of width 0.5), and re-run
F1, F5, F6 (split-half ramp + ICC ramp), F4-canonical against ψ_S3.

Outputs:
  corpus/spine/kuairand_pure_S3.parquet            (with psi_S3 column)
  baselines/F_tests_S3_kuairand_pure.json
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KR_DIR = ROOT / "datasets" / "KuaiRand-Pure" / "data"
PARQUET_IN = ROOT / "corpus" / "spine" / "kuairand_pure.parquet"
PARQUET_OUT = ROOT / "corpus" / "spine" / "kuairand_pure_S3.parquet"
F_OUT = ROOT / "baselines" / "F_tests_S3_kuairand_pure.json"

K_FLOOR = 1.0
RNG_SEED = 0


def k_floor(s_raw: float) -> float:
    if not np.isfinite(s_raw):
        return K_FLOOR
    return max(s_raw, K_FLOOR)


def s3_tag_x_duration(vf: pd.DataFrame) -> pd.Series:
    """Joint marginal surprisal of (tag-string, log10(duration_ms)//0.5)."""
    dur = vf["video_duration"].astype(float)
    bucket = np.floor(np.log10(dur.clip(lower=1)) / 0.5).astype("Int64")
    keys = list(zip(vf["tag"], bucket))
    keys_clean = [k if (isinstance(k[0], str) and not pd.isna(k[1])) else None for k in keys]
    keys_s = pd.Series(keys_clean, index=vf.index)
    p = keys_s.value_counts(normalize=True).to_dict()
    surp = {v: -math.log2(prob) for v, prob in p.items() if prob > 0}
    return keys_s.map(surp)


def icc_1_k(per_item_groups: pd.core.groupby.SeriesGroupBy, k: int) -> tuple[float, int]:
    sizes = per_item_groups.size()
    eligible = sizes[sizes >= 2].index
    if len(eligible) < 100:
        return float("nan"), int(len(eligible))
    means = per_item_groups.mean().loc[eligible]
    var_within = per_item_groups.var(ddof=1).loc[eligible].fillna(0.0)
    sb2 = float(np.var(means, ddof=1))
    sw2 = float(np.mean(var_within))
    denom = sb2 + sw2 / k
    if denom <= 0:
        return float("nan"), int(len(eligible))
    return sb2 / denom, int(len(eligible))


def split_half_rho(df: pd.DataFrame, value_col: str,
                   n_low: int, n_high: int | None,
                   rng: np.random.Generator) -> tuple[float, int]:
    sizes = df.groupby("item_id", sort=False).size()
    if n_high is None:
        keep = sizes[sizes >= n_low].index
    else:
        keep = sizes[(sizes >= n_low) & (sizes <= n_high)].index
    if len(keep) < 30:
        return float("nan"), int(len(keep))
    sub = df[df["item_id"].isin(keep)].copy()
    sub["_half"] = rng.integers(0, 2, size=len(sub))
    g = sub.groupby(["item_id", "_half"])[value_col].mean().unstack("_half")
    g = g.dropna()
    if len(g) < 30:
        return float("nan"), int(len(g))
    rho, _ = stats.spearmanr(g[0], g[1])
    return float(rho), int(len(g))


def f1_within_item(df: pd.DataFrame, value_col: str,
                   min_enc: int = 30) -> tuple[float, int]:
    sizes = df.groupby("item_id", sort=False).size()
    keep = sizes[sizes >= min_enc].index
    if len(keep) < 30:
        return float("nan"), int(len(keep))
    sub = df[df["item_id"].isin(keep)].copy()
    rng = np.random.default_rng(0)
    sub["_user_half"] = rng.integers(0, 2, size=len(sub))
    g = sub.groupby(["item_id", "_user_half"])[value_col].mean().unstack("_user_half")
    g = g.dropna()
    rho, _ = stats.spearmanr(g[0], g[1])
    return float(rho), int(len(g))


def f5_popularity_overlap(df: pd.DataFrame, value_col: str,
                          min_enc: int) -> tuple[float, int]:
    sizes = df.groupby("item_id", sort=False).size()
    keep = sizes[sizes >= min_enc].index
    if len(keep) < 30:
        return float("nan"), int(len(keep))
    psi_i = df[df["item_id"].isin(keep)].groupby("item_id")[value_col].mean()
    counts = sizes.loc[keep]
    rho, _ = stats.spearmanr(psi_i, counts)
    return float(rho), int(len(keep))


def main():
    rng = np.random.default_rng(RNG_SEED)

    print(f"Loading {PARQUET_IN}…")
    df = pd.read_parquet(PARQUET_IN)
    print(f"rows: {len(df):,}  cols: {list(df.columns)}")

    print(f"Loading video features and computing S3…")
    vf = pd.read_csv(KR_DIR / "video_features_basic_pure.csv")
    vf["item_id"] = "kr_" + vf["video_id"].astype(str)
    s3_raw_per_video = s3_tag_x_duration(vf)
    s3_by_item = pd.Series(s3_raw_per_video.values, index=vf["item_id"].values)
    s3_by_item = s3_by_item.groupby(level=0).first()

    print(f"S3 unique values: {s3_by_item.nunique()}, mean={s3_by_item.mean():.3f}, std={s3_by_item.std():.3f}")

    df["S3_raw"] = df["item_id"].map(s3_by_item)
    df["S3"] = df["S3_raw"].fillna(0.0).map(k_floor)
    df["psi_S3"] = (df["c"] * (1.0 + df["R_norm"])) / df["S3"]
    df["user_num"] = df["c"] * (1.0 + df["R_norm"])

    n_missing_meta = df["S3_raw"].isna().sum()
    print(f"rows with no S3 metadata (cold-start meta): {n_missing_meta:,} ({100*n_missing_meta/len(df):.2f}%)")

    print(f"Writing {PARQUET_OUT}…")
    df.drop(columns=["user_num"]).to_parquet(PARQUET_OUT, index=False)

    print("\n=== F-tests on ψ_S3 vs ψ_S0 (released) ===\n")

    # F1: within-item user-half consistency on items with ≥30 encounters.
    f1_S3,  n_f1 = f1_within_item(df, "psi_S3")
    f1_S0,  _    = f1_within_item(df, "psi")
    f1_num, _    = f1_within_item(df, "user_num")

    # F5: popularity overlap at ≥10 encounters (matches paper threshold).
    f5_S3,  n_f5 = f5_popularity_overlap(df, "psi_S3", 10)
    f5_S0,  _    = f5_popularity_overlap(df, "psi", 10)
    f5_num, _    = f5_popularity_overlap(df, "user_num", 10)

    # F6: ICC(1,k) ramp on full ψ_S3, ψ_S0, numerator-only.
    icc_ramp = {}
    for value in ["psi_S3", "psi", "user_num"]:
        grp = df.groupby("item_id", sort=False)[value]
        ramp = {}
        for k in [1, 2, 5, 10, 30, 100]:
            ic, n_eligible = icc_1_k(grp, k)
            ramp[f"k={k}"] = {"icc": ic, "n_items": n_eligible}
        icc_ramp[value] = ramp

    # F6 split-half ramp at the buckets the paper reports for KR.
    rho_buckets = [(31, 100), (101, None)]
    rho_ramp = {}
    for value in ["psi_S3", "psi", "user_num"]:
        rho_ramp[value] = {}
        for lo, hi in rho_buckets:
            r, n = split_half_rho(df, value, lo, hi, rng)
            label = f"n>={lo}" if hi is None else f"n in [{lo},{hi}]"
            rho_ramp[value][label] = {"rho": r, "n_items": n}

    # F4-canonical with ψ_S3.
    print("Re-merging is_rand for F4-canonical with ψ_S3…")
    standard = [KR_DIR / "log_standard_4_08_to_4_21_pure.csv",
                KR_DIR / "log_standard_4_22_to_5_08_pure.csv"]
    rand_csv = KR_DIR / "log_random_4_22_to_5_08_pure.csv"
    keep = ["user_id", "video_id", "time_ms", "is_rand"]
    chunks = [pd.read_csv(p, usecols=keep) for p in standard + [rand_csv]]
    src = pd.concat(chunks, ignore_index=True)
    df_join = df.copy()
    df_join["uid_int"] = df_join["user_id"].str.removeprefix("kr_").astype("int64")
    df_join["vid_int"] = df_join["item_id"].str.removeprefix("kr_").astype("int64")
    df_join["time_ms"] = df_join["timestamp"].astype("int64")
    src.rename(columns={"user_id": "uid_int", "video_id": "vid_int"}, inplace=True)
    merged = df_join.merge(src, on=["uid_int", "vid_int", "time_ms"], how="left",
                           validate="many_to_many")

    f4_results = {}
    for value in ["psi_S3", "psi"]:
        rand_slice = merged[merged["is_rand"] == 1]
        serv_slice = merged[merged["is_rand"] == 0]
        psi_r = rand_slice.groupby("item_id")[value].mean()
        psi_s = serv_slice.groupby("item_id")[value].mean()
        n_r = rand_slice.groupby("item_id").size()
        n_s = serv_slice.groupby("item_id").size()
        keep_items = (n_r[n_r >= 10].index.intersection(n_s[n_s >= 10].index))
        if len(keep_items) < 30:
            f4_results[value] = {"skipped": f"<30 items ({len(keep_items)})"}
            continue
        a = psi_r.loc[keep_items].to_numpy(dtype=float)
        b = psi_s.loc[keep_items].to_numpy(dtype=float)
        pearson, _ = stats.pearsonr(a, b)
        rho, _ = stats.spearmanr(a, b)
        f4_results[value] = {
            "n_items": int(len(keep_items)),
            "pearson_r": float(pearson),
            "spearman_rho": float(rho),
            "psi_rand_mean": float(np.mean(a)),
            "psi_serv_mean": float(np.mean(b)),
        }

    out = {
        "shard": str(PARQUET_IN),
        "n_rows": int(len(df)),
        "S_definition": "S3 = -log2 P(tag-string, log10(video_duration_ms)//0.5 bucket)",
        "S3_unique_values": int(s3_by_item.nunique()),
        "S3_mean": float(s3_by_item.mean()),
        "S3_std": float(s3_by_item.std(ddof=1)),
        "rows_missing_S3_metadata": int(n_missing_meta),
        "F1_within_item_user_half_rho_min30enc": {
            "psi_S3": {"rho": f1_S3, "n_items": n_f1},
            "psi (released S0)": {"rho": f1_S0, "n_items": n_f1},
            "numerator_only c(1+R)": {"rho": f1_num, "n_items": n_f1},
        },
        "F5_popularity_overlap_min10enc": {
            "psi_S3": {"rho": f5_S3, "n_items": n_f5},
            "psi (released S0)": {"rho": f5_S0, "n_items": n_f5},
            "numerator_only c(1+R)": {"rho": f5_num, "n_items": n_f5},
        },
        "F6_icc_1_k_ramp": icc_ramp,
        "F6_split_half_rho_ramp": rho_ramp,
        "F4_canonical_with_isrand_merged": f4_results,
    }
    print(json.dumps(out, indent=2, default=str))
    F_OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {F_OUT}")


if __name__ == "__main__":
    main()
