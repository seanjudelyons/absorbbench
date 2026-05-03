"""
Tenrec QK-article analog of the S3 experiment.

Article-side metadata: (category_first, category_second) for the current S0,
plus item_score1 (5 discrete values) and item_score3 (10 discrete values)
as candidate object-side enrichment dimensions. There is no per-item
duration analog for articles in this schema.

Tries:
  S0_tenrec_released         (category_first, category_second) — released
  S3a_cat_x_score1           (cat_first, cat_second, item_score1)
  S3b_cat_x_score3           (cat_first, cat_second, item_score3)
  S3c_cat_x_score1_x_score3  (cat_first, cat_second, item_score1, item_score3)

For each, computes ICC(1,1), ICC(1,5), ICC(1,30); split-half ρ at
n∈[2], n∈[6,10], n≥31; numerator-only baselines.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "tenrec" / "QK-article.csv"
PARQUET_OUT = ROOT / "corpus" / "spine" / "tenrec_qk_article_S3.parquet"
F_OUT = ROOT / "baselines" / "F_tests_S3_tenrec_qk_article.json"

K_FLOOR = 1.0
RNG_SEED = 0


def k_floor(s_raw):
    if not np.isfinite(s_raw):
        return K_FLOOR
    return max(s_raw, K_FLOOR)


def joint_surprisal(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    keys = list(zip(*[df[c].values for c in cols]))
    keys_s = pd.Series(keys, index=df.index)
    p = keys_s.value_counts(normalize=True).to_dict()
    surp = {v: -math.log2(prob) for v, prob in p.items() if prob > 0}
    return keys_s.map(surp)


def normalise_R_p90(R_count: pd.Series) -> pd.Series:
    pos = R_count[R_count > 0]
    if len(pos) == 0:
        return R_count.astype(float) * 0.0
    p90 = float(np.percentile(pos, 90))
    if p90 <= 0:
        return R_count.astype(float) * 0.0
    return (R_count.clip(upper=p90) / p90).astype(float)


def icc_1_k(per_item_groups, k: int) -> tuple[float, int]:
    sizes = per_item_groups.size()
    eligible = sizes[sizes >= 2].index
    if len(eligible) < 100:
        return float("nan"), int(len(eligible))
    means = per_item_groups.mean().loc[eligible]
    var_within = per_item_groups.var(ddof=1).loc[eligible].fillna(0.0)
    sb2 = float(np.var(means, ddof=1))
    sw2 = float(np.mean(var_within))
    denom = sb2 + sw2 / k
    return (sb2 / denom if denom > 0 else float("nan")), int(len(eligible))


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


def main():
    print(f"Loading {SRC}…  (this is the 46M-row file)")
    cols = ["user_id", "item_id", "category_first", "category_second",
            "item_score1", "item_score3", "read", "read_percentage",
            "share", "like", "follow", "favorite"]
    df = pd.read_csv(SRC, usecols=cols)
    print(f"rows: {len(df):,}")

    # User-side: c and R̃
    df["read_bool"] = df["read"].astype(bool)
    df["c"] = np.where(df["read_bool"],
                       df["read_percentage"].fillna(0).astype(float) / 100.0,
                       0.0).clip(0, 1)
    refl = ["share", "like", "follow", "favorite"]
    df[refl] = df[refl].astype(bool).astype(int)
    df["R_count"] = df[refl].sum(axis=1)
    df["R_norm"] = normalise_R_p90(df["R_count"])
    df["user_num"] = df["c"] * (1.0 + df["R_norm"])
    df["item_id_str"] = "tr_qka_" + df["item_id"].astype(str)

    # Per-item metadata table (one row per item)
    item_meta = df.drop_duplicates("item_id_str")[
        ["item_id_str", "category_first", "category_second",
         "item_score1", "item_score3"]
    ].set_index("item_id_str")

    candidates = {
        "S0_cat_pair (released analog)":
            joint_surprisal(item_meta.reset_index(),
                            ["category_first", "category_second"]),
        "S3a_cat_x_score1":
            joint_surprisal(item_meta.reset_index(),
                            ["category_first", "category_second", "item_score1"]),
        "S3b_cat_x_score3":
            joint_surprisal(item_meta.reset_index(),
                            ["category_first", "category_second", "item_score3"]),
        "S3c_cat_x_score1_x_score3":
            joint_surprisal(item_meta.reset_index(),
                            ["category_first", "category_second",
                             "item_score1", "item_score3"]),
    }

    rng = np.random.default_rng(RNG_SEED)
    results = []
    for name, s_per_item in candidates.items():
        # s_per_item is indexed by reset_index range; rejoin to item_id_str
        s_lookup = pd.Series(s_per_item.values,
                             index=item_meta.reset_index()["item_id_str"].values)
        s_lookup = s_lookup.groupby(level=0).first()

        df["S_raw"] = df["item_id_str"].map(s_lookup)
        df["S"] = df["S_raw"].fillna(0.0).map(k_floor)
        df["psi_cand"] = df["user_num"] / df["S"]

        # Build a working frame with a single "item_id" string column
        # (the source has int item_id; our helpers expect "item_id" string).
        df_eval = df.drop(columns=["item_id"]).rename(
            columns={"item_id_str": "item_id"})

        grp_full = df_eval.groupby("item_id", sort=False)["psi_cand"]
        grp_num = df_eval.groupby("item_id", sort=False)["user_num"]

        ramp_full = {}
        ramp_num = {}
        for k in [1, 5, 10, 30]:
            f_icc, n_e = icc_1_k(grp_full, k)
            n_icc, _   = icc_1_k(grp_num,  k)
            ramp_full[f"k={k}"] = f_icc
            ramp_num[f"k={k}"]  = n_icc

        rho_full = {}
        rho_num = {}
        for lo, hi in [(2, 2), (3, 5), (6, 10), (31, 100), (101, None)]:
            rf, n_rf = split_half_rho(df_eval, "psi_cand", lo, hi, rng)
            rn, _    = split_half_rho(df_eval, "user_num", lo, hi, rng)
            label = f"n>={lo}" if hi is None else (
                f"n=={lo}" if lo == hi else f"n in [{lo},{hi}]")
            rho_full[label] = {"rho": rf, "n": n_rf}
            rho_num[label]  = {"rho": rn}

        s_per = item_meta.assign(S=s_lookup)["S"]
        results.append({
            "name": name,
            "S_unique_values": int(s_per.nunique()),
            "S_mean": float(s_per.mean()),
            "S_std": float(s_per.std(ddof=1)),
            "rows_missing_metadata": int(df["S_raw"].isna().sum()),
            "ICC_full_psi": ramp_full,
            "ICC_numerator_only": ramp_num,
            "ICC_gap_at_k1": ramp_full["k=1"] - ramp_num["k=1"],
            "split_half_rho_full_psi": rho_full,
            "split_half_rho_numerator": rho_num,
        })

        # Save the S3a ψ to a parquet for later F-tests / paper integration.
        if name == "S3a_cat_x_score1":
            print(f"Writing {PARQUET_OUT}…")
            df.rename(columns={"item_id_str": "item_id_full",
                               "psi_cand": "psi_S3a"})[
                ["user_id", "item_id_full", "c", "R_count", "R_norm",
                 "S_raw", "S", "psi_S3a"]
            ].to_parquet(PARQUET_OUT, index=False)

    out = {
        "shard": str(SRC),
        "n_rows": int(len(df)),
        "candidates": results,
    }
    print(json.dumps(out, indent=2, default=str))
    F_OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {F_OUT}")


if __name__ == "__main__":
    main()
