"""
S-richness experiment: try several richer S candidates and check whether
any yields a numerator-only-vs-full-ψ ICC gap large enough to make S
load-bearing.

Ranges of S(i) tested on KuaiRand-Pure spine:
  S0  current — tag-string categorical surprisal (released artefact)
  S1  multi-tag mean — for multi-tag videos, mean tag-marginal surprisal
                       across constituent tags (expanded to single-tag basis)
  S2  multi-tag joint — joint marginal of *sorted tag-set tuple*
                        treated as a single category
  S3  tag × duration-bucket — joint marginal of (tag-string,
                              log10(video_duration_ms)-bucket of width 0.5)
  S4  tag × upload_type — joint marginal of (tag-string, upload_type)
  S5  tag × video_type × upload_type — three-way joint marginal

For each candidate we recompute ψ = c·(1+R̃)/S and report:
  - ICC(1,1) numerator-only [c·(1+R̃)] : same across all candidates
  - ICC(1,1) full-ψ
  - Gap = ICC(full) − ICC(numerator)
  - Mean and std of S(i) across items
  - Split-half Spearman ρ on ψᵢ at n∈[6,10]
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
PARQUET = ROOT / "corpus" / "spine" / "kuairand_pure.parquet"
OUT = ROOT / "baselines" / "s_richness_experiment_kuairand_pure.json"

K_FLOOR = 1.0
RNG_SEED = 0


def k_floor(s_raw: float) -> float:
    if not np.isfinite(s_raw):
        return K_FLOOR
    return max(s_raw, K_FLOOR)


def surprisal_from_marginal(values: pd.Series) -> dict:
    p = values.value_counts(normalize=True).to_dict()
    return {v: -math.log2(prob) for v, prob in p.items() if prob > 0}


def s0_tagstring(vf: pd.DataFrame) -> pd.Series:
    """Current released S(i): surprisal of full tag string."""
    surp = surprisal_from_marginal(vf["tag"].dropna())
    return vf["tag"].map(surp)


def s1_multitag_mean(vf: pd.DataFrame) -> pd.Series:
    """Mean surprisal of constituent single tags."""
    tags_lists = vf["tag"].dropna().astype(str).str.split(",")
    flat = [t.strip() for ts in tags_lists for t in ts]
    p = Counter(flat); total = sum(p.values())
    surp = {t: -math.log2(c / total) for t, c in p.items() if c > 0}

    def per_video(s):
        if not isinstance(s, str):
            return np.nan
        ts = [t.strip() for t in s.split(",")]
        vals = [surp.get(t) for t in ts if t in surp]
        return float(np.mean(vals)) if vals else np.nan
    return vf["tag"].map(per_video)


def s2_tagset_joint(vf: pd.DataFrame) -> pd.Series:
    """Joint marginal of sorted-tag-set tuple."""
    def canonical(s):
        if not isinstance(s, str):
            return None
        return tuple(sorted(t.strip() for t in s.split(",")))
    canon = vf["tag"].map(canonical)
    surp = surprisal_from_marginal(canon.dropna())
    return canon.map(surp)


def s3_tag_x_duration(vf: pd.DataFrame) -> pd.Series:
    """Joint marginal of (tag-string, log10(duration_ms) // 0.5)."""
    dur = vf["video_duration"].astype(float)
    bucket = np.floor(np.log10(dur.clip(lower=1)) / 0.5).astype("Int64")
    key = list(zip(vf["tag"], bucket))
    key_s = pd.Series([k if (k[0] is not None and not pd.isna(k[1])) else None
                       for k in key], index=vf.index)
    surp = surprisal_from_marginal(key_s.dropna())
    return key_s.map(surp)


def s4_tag_x_uploadtype(vf: pd.DataFrame) -> pd.Series:
    key = list(zip(vf["tag"], vf["upload_type"]))
    key_s = pd.Series(
        [k if (k[0] is not None and isinstance(k[1], str)) else None for k in key],
        index=vf.index,
    )
    surp = surprisal_from_marginal(key_s.dropna())
    return key_s.map(surp)


def s5_tag_x_videotype_x_uploadtype(vf: pd.DataFrame) -> pd.Series:
    key = list(zip(vf["tag"], vf["video_type"], vf["upload_type"]))
    key_s = pd.Series(
        [k if (k[0] is not None and isinstance(k[1], str) and isinstance(k[2], str)) else None
         for k in key],
        index=vf.index,
    )
    surp = surprisal_from_marginal(key_s.dropna())
    return key_s.map(surp)


def icc_1_1(per_item_groups: pd.core.groupby.SeriesGroupBy) -> float:
    """ICC(1,1) computed via variance decomposition.
    sigma_between^2 = Var(item means)
    sigma_within^2 = E[Var within item]
    ICC(1,1) = sigma_between^2 / (sigma_between^2 + sigma_within^2)
    """
    sizes = per_item_groups.size()
    eligible = sizes[sizes >= 2].index
    if len(eligible) < 100:
        return float("nan")
    means = per_item_groups.mean().loc[eligible]
    var_within = per_item_groups.var(ddof=1).loc[eligible].fillna(0.0)
    sigma_between_sq = float(np.var(means, ddof=1))
    sigma_within_sq = float(np.mean(var_within))
    denom = sigma_between_sq + sigma_within_sq
    if denom <= 0:
        return float("nan")
    return sigma_between_sq / denom


def split_half_rho_in_bucket(df: pd.DataFrame, value_col: str,
                             n_low: int, n_high: int,
                             rng: np.random.Generator) -> tuple[float, int]:
    """Spearman ρ between two random halves of each item's encounters,
    on items whose total encounter count is in [n_low, n_high]."""
    sizes = df.groupby("item_id", sort=False).size()
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


def evaluate(name: str, df: pd.DataFrame, s_raw_per_video_id: pd.Series,
             rng: np.random.Generator) -> dict:
    s_floored = s_raw_per_video_id.fillna(0.0).map(k_floor)
    df = df.assign(_S=df["item_id"].map(s_floored))
    df = df.dropna(subset=["_S"])  # items missing metadata
    df["_psi_full"] = df["_user"] / df["_S"]
    # numerator-only is df["_user"] directly

    grp_full = df.groupby("item_id", sort=False)["_psi_full"]
    grp_num = df.groupby("item_id", sort=False)["_user"]

    icc_full = icc_1_1(grp_full)
    icc_num = icc_1_1(grp_num)

    # KR-Pure has very few items in n∈[6,10] (random-exposure spreads
    # encounters); F6 in the paper reports KR ramp at n∈[31,100].
    rho_full, n_full = split_half_rho_in_bucket(df, "_psi_full", 31, 100, rng)
    rho_num,  n_num  = split_half_rho_in_bucket(df, "_user",     31, 100, rng)

    s_per_item = df.groupby("item_id")["_S"].first()
    return {
        "name": name,
        "n_items_with_metadata": int(s_per_item.notna().sum()),
        "S_mean": float(s_per_item.mean()),
        "S_std": float(s_per_item.std(ddof=1)),
        "S_unique_values": int(s_per_item.nunique()),
        "icc_1_1_full_psi": icc_full,
        "icc_1_1_numerator_only": icc_num,
        "icc_gap_full_minus_num": icc_full - icc_num,
        "split_half_rho_full_psi_n31_100": rho_full,
        "split_half_rho_numerator_n31_100": rho_num,
        "rho_gap_full_minus_num_n31_100": rho_full - rho_num,
        "n_items_in_n31_100_bucket": n_full,
    }


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("Loading parquet…")
    df = pd.read_parquet(PARQUET, columns=["item_id", "c", "R_norm"])
    df["_user"] = df["c"] * (1.0 + df["R_norm"])
    print(f"rows: {len(df):,}")

    print("Loading video features…")
    vf = pd.read_csv(KR_DIR / "video_features_basic_pure.csv")
    vf["item_id"] = "kr_" + vf["video_id"].astype(str)

    # Controls for S3: is the gain coming from duration alone, or from
    # the joint of (tag, duration)?
    def s_duration_only(vf):
        dur = vf["video_duration"].astype(float)
        bucket = np.floor(np.log10(dur.clip(lower=1)) / 0.5).astype("Int64")
        bucket_key = bucket.astype("string")
        surp = surprisal_from_marginal(bucket_key.dropna())
        return bucket_key.map(surp)

    def s_random_categorical(vf, n_buckets=20):
        rng_local = np.random.default_rng(42)
        rb = pd.Series(rng_local.integers(0, n_buckets, size=len(vf)),
                       index=vf.index).astype("string")
        surp = surprisal_from_marginal(rb)
        return rb.map(surp)

    candidates = {
        "S0_tagstring (released)":         s0_tagstring(vf),
        "S1_multitag_mean":                s1_multitag_mean(vf),
        "S2_tagset_joint":                 s2_tagset_joint(vf),
        "S3_tag_x_duration":               s3_tag_x_duration(vf),
        "S4_tag_x_uploadtype":             s4_tag_x_uploadtype(vf),
        "S5_tag_x_videotype_x_uploadtype": s5_tag_x_videotype_x_uploadtype(vf),
        "Sctrl_duration_only":             s_duration_only(vf),
        "Sctrl_random_20bucket":           s_random_categorical(vf),
    }

    results = []
    for name, s_raw in candidates.items():
        # build per-item lookup
        s_by_item = pd.Series(s_raw.values, index=vf["item_id"].values)
        s_by_item = s_by_item.groupby(level=0).first()  # dedupe
        print(f"\n=== {name} ===")
        try:
            r = evaluate(name, df, s_by_item, rng)
        except Exception as e:
            r = {"name": name, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(r, indent=2, default=str))
        results.append(r)

    OUT.write_text(json.dumps({
        "shard": str(PARQUET),
        "n_rows": int(len(df)),
        "candidates": results,
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
