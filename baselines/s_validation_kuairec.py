"""
Out-of-sample S3 validation on KuaiRec.

KuaiRec is a forced-exposure short-form video corpus (4.7M
user-item pairs in big_matrix; every user × every video). It shares
modality (video) with KuaiRand-Pure but uses a different exposure
regime (forced) and a different platform feed. If the same selection
rule that picked S3 on KR-Pure (joint of category + log-duration
beats random-bucket control AND duration-only) also picks the joint
on KuaiRec, the S choice generalises within video modality and is
not an artefact of KR's specific exposure dynamics.

KuaiRec has no per-encounter reflective acts (forced exposure with
play_duration only), so R̃ ≡ 0 and ψ collapses to c/S. The
S-richness ablation is therefore a clean test of whether 1/S
contributes ICC reliability above the user-side numerator.

Pre-registered selection rule (the same as on KR-Pure):
  S3 wins if its ICC(1,1) gap over numerator-only is greater than
  the random-bucket control gap AND greater than duration-only gap.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KR_DIR = ROOT / "datasets" / "KuaiRec" / "KuaiRec 2.0" / "data"
OUT = ROOT / "baselines" / "s_validation_kuairec.json"

K_FLOOR = 1.0
RNG_SEED = 0


def k_floor(s):
    if not np.isfinite(s):
        return K_FLOOR
    return max(s, K_FLOOR)


def surprisal_marginal(values: pd.Series) -> dict:
    p = values.value_counts(normalize=True).to_dict()
    return {v: -math.log2(prob) for v, prob in p.items() if prob > 0}


def icc_1_1(grp: pd.core.groupby.SeriesGroupBy) -> tuple[float, int]:
    sizes = grp.size()
    eligible = sizes[sizes >= 2].index
    if len(eligible) < 100:
        return float("nan"), int(len(eligible))
    means = grp.mean().loc[eligible]
    var_within = grp.var(ddof=1).loc[eligible].fillna(0.0)
    sb = float(np.var(means, ddof=1))
    sw = float(np.mean(var_within))
    return (sb / (sb + sw)) if (sb + sw) > 0 else float("nan"), int(len(eligible))


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("Loading KuaiRec big_matrix.csv (4.7M+ rows)…")
    log = pd.read_csv(KR_DIR / "big_matrix.csv",
                      usecols=["user_id", "video_id", "play_duration",
                               "video_duration", "watch_ratio"])
    print(f"rows: {len(log):,}")

    print("Loading category metadata…")
    cats = pd.read_csv(KR_DIR / "item_categories.csv")
    cats["feat_str"] = cats["feat"].astype(str)
    # Per-item duration: use median of observed durations from log (each video
    # has a fixed duration in the source; take the first non-null).
    dur_per_item = log.dropna(subset=["video_duration"]).groupby("video_id")["video_duration"].first()
    cats["duration_ms"] = cats["video_id"].map(dur_per_item)

    # ψ ingredients
    log["c"] = log["watch_ratio"].astype(float).clip(0, 1)
    # KuaiRec has no per-encounter reflective acts → R̃ ≡ 0, user-side = c.
    log["user_side"] = log["c"]

    # S candidates (per-item)
    print("Building S candidates…")
    # S0_kuairec: feat-string marginal (the released analog)
    s_map_S0 = surprisal_marginal(cats["feat_str"].dropna())
    cats["S0"] = cats["feat_str"].map(s_map_S0).fillna(0.0).map(k_floor)

    # S3_kuairec: joint (feat-string, log-duration-bucket)
    bucket = np.floor(np.log10(cats["duration_ms"].clip(lower=1)) / 0.5).astype("Int64")
    keys = list(zip(cats["feat_str"], bucket))
    keys_clean = [k if (isinstance(k[0], str) and not pd.isna(k[1])) else None for k in keys]
    keys_s = pd.Series(keys_clean, index=cats.index)
    s_map_S3 = surprisal_marginal(keys_s.dropna())
    cats["S3"] = keys_s.map(s_map_S3).fillna(0.0).map(k_floor)

    # Control: duration-only
    bucket_only = bucket.astype("string")
    s_map_dur = surprisal_marginal(bucket_only.dropna())
    cats["S_dur"] = bucket_only.map(s_map_dur).fillna(0.0).map(k_floor)

    # Control: random 20-bucket
    rb_rng = np.random.default_rng(42)
    rb = pd.Series(rb_rng.integers(0, 20, size=len(cats)), index=cats.index).astype("string")
    s_map_rand = surprisal_marginal(rb)
    cats["S_rand20"] = rb.map(s_map_rand).fillna(0.0).map(k_floor)

    s_lookups = {name: dict(zip(cats["video_id"], cats[name]))
                 for name in ["S0", "S3", "S_dur", "S_rand20"]}

    print("Computing per-encounter ψ for each candidate and evaluating ICC…")

    results = {
        "test": "S_validation_KuaiRec (out-of-sample for S3 selection rule)",
        "shard": "KuaiRec big_matrix",
        "n_rows": int(len(log)),
        "exposure_regime": "forced-exposure (every user × every video)",
        "user_side_form": "c only (no per-encounter reflective acts)",
        "candidates": [],
    }

    # Numerator-only ICC (same regardless of S)
    log_for_grp = log.assign(item_id=log["video_id"].astype(int))
    grp_num = log_for_grp.groupby("item_id", sort=False)["user_side"]
    icc_num, n_num = icc_1_1(grp_num)
    results["icc_numerator_only_c"] = float(icc_num)
    print(f"numerator-only ICC(1,1) = c-mean: {icc_num:.4f} on {n_num} items")

    for name in ["S0", "S3", "S_dur", "S_rand20"]:
        s_per_item = log["video_id"].map(s_lookups[name]).fillna(K_FLOOR)
        psi = log["user_side"] / s_per_item
        df_eval = log.assign(item_id=log["video_id"].astype(int), psi=psi)
        grp = df_eval.groupby("item_id", sort=False)["psi"]
        icc, n_e = icc_1_1(grp)
        gap = float(icc) - icc_num
        s_unique = int(cats[name].nunique())
        s_mean = float(cats[name].mean())
        s_std = float(cats[name].std(ddof=1))
        r = {
            "name": name,
            "S_unique_values": s_unique,
            "S_mean": s_mean,
            "S_std": s_std,
            "ICC_full_psi": float(icc),
            "ICC_gap_over_numerator": gap,
            "n_items": int(n_e),
        }
        results["candidates"].append(r)
        print(f"{name:10s}  S_unique={s_unique:4d}  ICC_full={icc:.4f}  gap={gap:+.4f}")

    # Pre-registered selection rule
    cand = {c["name"]: c for c in results["candidates"]}
    s3_gap = cand["S3"]["ICC_gap_over_numerator"]
    rand_gap = cand["S_rand20"]["ICC_gap_over_numerator"]
    dur_gap = cand["S_dur"]["ICC_gap_over_numerator"]
    s3_wins = (s3_gap > rand_gap) and (s3_gap > dur_gap)

    results["selection_rule"] = (
        "S3 wins iff ICC_gap(S3) > ICC_gap(random-control) AND > ICC_gap(duration-only)"
    )
    results["selection_rule_outcome"] = {
        "S3_gap": s3_gap,
        "random_control_gap": rand_gap,
        "duration_only_gap": dur_gap,
        "S3_wins_under_rule": bool(s3_wins),
    }
    print()
    print("=" * 60)
    print(f"Selection rule outcome: S3 {'WINS' if s3_wins else 'LOSES'}")
    print(f"  S3 gap: {s3_gap:+.4f}")
    print(f"  random-control gap: {rand_gap:+.4f}")
    print(f"  duration-only gap: {dur_gap:+.4f}")
    print("=" * 60)

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
