"""
F7 — External anchor: does ψᵢ predict held-out behavioural outcomes
that are NOT components of ψ?

Test design:
  Time-split KuaiRand-Pure into train (first 70% by time_ms) and test
  (last 30%). For each item with sufficient encounters in both
  windows, compute:
    - training ψᵢ (under both ψ_S0 and ψ_S3)
    - test-window per-encounter mean of three external behavioural
      signals not used in computing ψ:
        * profile_stay_time (seconds on author's profile)
        * comment_stay_time (seconds reading comments)
        * is_hate rate (negative-engagement indicator)

Hypothesis: high-ψᵢ items should predict longer profile/comment dwell
and lower is_hate rate. Reference threshold: |Spearman ρ| ≥ 0.20 on
the relevant direction.

This addresses the "circular benchmark" attack: ψ is built from
(c, R̃, S); profile_stay_time / comment_stay_time / is_hate are not
components of (c, R̃, S) and not in any S formulation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
KR_DIR = ROOT / "datasets" / "KuaiRand-Pure" / "data"
OUT = ROOT / "baselines" / "F7_external_anchor_kuairand_pure.json"

K_FLOOR = 1.0
RNG_SEED = 0
TRAIN_FRAC = 0.70
MIN_ENC_PER_WINDOW = 10


def k_floor(s):
    if not np.isfinite(s):
        return K_FLOOR
    return max(s, K_FLOOR)


def surprisal_marginal(values: pd.Series) -> dict:
    p = values.value_counts(normalize=True).to_dict()
    return {v: -math.log2(prob) for v, prob in p.items() if prob > 0}


def normalise_R(R_count: pd.Series) -> pd.Series:
    pos = R_count[R_count > 0]
    if len(pos) == 0:
        return R_count.astype(float) * 0.0
    p90 = float(np.percentile(pos, 90))
    if p90 <= 0:
        return R_count.astype(float) * 0.0
    return (R_count.clip(upper=p90) / p90).astype(float)


def main():
    print("Loading KuaiRand-Pure source CSVs…")
    cols = ["user_id", "video_id", "time_ms", "play_time_ms", "duration_ms",
            "is_like", "is_follow", "is_comment", "is_forward", "long_view",
            "is_profile_enter", "is_hate", "is_click",
            "profile_stay_time", "comment_stay_time"]
    paths = [
        KR_DIR / "log_standard_4_08_to_4_21_pure.csv",
        KR_DIR / "log_standard_4_22_to_5_08_pure.csv",
        KR_DIR / "log_random_4_22_to_5_08_pure.csv",
    ]
    chunks = [pd.read_csv(p, usecols=cols) for p in paths]
    log = pd.concat(chunks, ignore_index=True)
    print(f"rows: {len(log):,}")

    print("Loading video features for S0 and S3…")
    vf = pd.read_csv(KR_DIR / "video_features_basic_pure.csv")
    s0_map = surprisal_marginal(vf["tag"].dropna())
    vf["S0"] = vf["tag"].map(s0_map).fillna(0.0).map(k_floor)

    dur = vf["video_duration"].astype(float)
    bucket = np.floor(np.log10(dur.clip(lower=1)) / 0.5).astype("Int64")
    s3_keys = list(zip(vf["tag"], bucket))
    s3_keys_clean = [k if (isinstance(k[0], str) and not pd.isna(k[1])) else None for k in s3_keys]
    s3_keys_s = pd.Series(s3_keys_clean, index=vf.index)
    s3_map = surprisal_marginal(s3_keys_s.dropna())
    vf["S3"] = s3_keys_s.map(s3_map).fillna(0.0).map(k_floor)

    s0_by_id = dict(zip(vf["video_id"], vf["S0"]))
    s3_by_id = dict(zip(vf["video_id"], vf["S3"]))

    print("Computing per-encounter c, R̃, ψ_S0, ψ_S3…")
    log["c"] = (log["play_time_ms"].astype(float)
                / log["duration_ms"].clip(lower=1).astype(float)).clip(0, 1)
    refl = ["is_like", "is_follow", "is_comment", "is_forward",
            "long_view", "is_profile_enter"]
    log["R_count"] = log[refl].sum(axis=1).astype(int)
    log["R_norm"] = normalise_R(log["R_count"])
    log["S0"] = log["video_id"].map(s0_by_id).fillna(K_FLOOR)
    log["S3"] = log["video_id"].map(s3_by_id).fillna(K_FLOOR)
    log["psi_S0"] = log["c"] * (1.0 + log["R_norm"]) / log["S0"]
    log["psi_S3"] = log["c"] * (1.0 + log["R_norm"]) / log["S3"]

    # Time split
    print("Time-splitting…")
    cutoff = log["time_ms"].quantile(TRAIN_FRAC)
    train = log[log["time_ms"] <= cutoff]
    test = log[log["time_ms"] > cutoff]
    print(f"  train rows: {len(train):,}  ({100*len(train)/len(log):.1f}%)")
    print(f"  test  rows: {len(test):,}  ({100*len(test)/len(log):.1f}%)")

    # Per-item training ψᵢ
    g_train = train.groupby("video_id")
    n_train = g_train.size()
    psi_S0_train = g_train["psi_S0"].mean()
    psi_S3_train = g_train["psi_S3"].mean()

    # Per-item held-out behavioural outcomes
    g_test = test.groupby("video_id")
    n_test = g_test.size()
    profile_stay_test = g_test["profile_stay_time"].mean()
    comment_stay_test = g_test["comment_stay_time"].mean()
    hate_rate_test = g_test["is_hate"].mean()
    click_rate_test = g_test["is_click"].mean()
    long_view_test = g_test["long_view"].mean()

    # Items with sufficient encounters in BOTH windows
    keep = (n_train[n_train >= MIN_ENC_PER_WINDOW].index
            .intersection(n_test[n_test >= MIN_ENC_PER_WINDOW].index))
    print(f"Items with ≥{MIN_ENC_PER_WINDOW} enc in both windows: {len(keep)}")

    if len(keep) < 30:
        out = {"test": "F7_external_anchor", "skipped": f"<30 items ({len(keep)})"}
        OUT.write_text(json.dumps(out, indent=2))
        return

    def report(psi_label, psi_train_series, anchor_label, anchor_test_series, expected_sign):
        psi = psi_train_series.loc[keep].to_numpy(dtype=float)
        anchor = anchor_test_series.loc[keep].to_numpy(dtype=float)
        # Drop pairs with NaN
        mask = np.isfinite(psi) & np.isfinite(anchor)
        if mask.sum() < 30:
            return {"skipped": f"<30 valid pairs ({mask.sum()})"}
        rho, p = stats.spearmanr(psi[mask], anchor[mask])
        pearson, p_p = stats.pearsonr(psi[mask], anchor[mask])
        passes_band = (rho >= 0.20) if expected_sign == "+" else (rho <= -0.20)
        return {
            "psi": psi_label,
            "anchor": anchor_label,
            "expected_sign": expected_sign,
            "n_items": int(mask.sum()),
            "spearman_rho": float(rho),
            "pearson_r": float(pearson),
            "p_value_spearman": float(p),
            "passes_0.20_band": bool(passes_band),
        }

    results = {
        "test": "F7_external_anchor",
        "shard": "kuairand_pure",
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "train_test_split": f"first {int(TRAIN_FRAC*100)}% by time_ms",
        "n_items_in_intersection": int(len(keep)),
        "min_encounters_per_window": MIN_ENC_PER_WINDOW,
        "external_anchors_NOT_in_psi_formula": [
            "profile_stay_time (seconds on author profile)",
            "comment_stay_time (seconds reading comments)",
            "is_hate (negative-engagement indicator)",
            "is_click (binary click)",
            "long_view (≥10s view flag)",
        ],
        "anchors": [],
    }

    pairs = [
        ("psi_S0", psi_S0_train, "profile_stay_time (held-out)", profile_stay_test, "+"),
        ("psi_S3", psi_S3_train, "profile_stay_time (held-out)", profile_stay_test, "+"),
        ("psi_S0", psi_S0_train, "comment_stay_time (held-out)", comment_stay_test, "+"),
        ("psi_S3", psi_S3_train, "comment_stay_time (held-out)", comment_stay_test, "+"),
        ("psi_S0", psi_S0_train, "is_hate rate (held-out)",       hate_rate_test,    "-"),
        ("psi_S3", psi_S3_train, "is_hate rate (held-out)",       hate_rate_test,    "-"),
        ("psi_S0", psi_S0_train, "is_click rate (held-out)",      click_rate_test,   "+"),
        ("psi_S3", psi_S3_train, "is_click rate (held-out)",      click_rate_test,   "+"),
        ("psi_S0", psi_S0_train, "long_view rate (held-out)",     long_view_test,    "+"),
        ("psi_S3", psi_S3_train, "long_view rate (held-out)",     long_view_test,    "+"),
    ]
    for psi_label, psi_train_s, anchor_label, anchor_test_s, exp in pairs:
        r = report(psi_label, psi_train_s, anchor_label, anchor_test_s, exp)
        results["anchors"].append(r)
        print(json.dumps(r, indent=2))

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
