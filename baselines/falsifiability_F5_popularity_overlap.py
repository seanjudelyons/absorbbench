"""
F5 — Bias-of-popularity test. Per mp_08 §3 (the user's flag, written falsifiably).

Test: rank-correlation between popularity-derived per-item ψ̂ and locked-formula
per-item ψᵢ on the same items.

Pre-registered three-way verdict (mp_08 §3 F5):
- ρ > 0.95: popularity ≈ locked-formula; corpus contribution is procedural.
- 0.6 ≤ ρ ≤ 0.95: formula and popularity rank items similarly but differ on
  specific item populations; report where they diverge.
- ρ < 0.6: formula and popularity measure substantially different per-item
  statistics.

Operational definitions:
- popularity-ψ̂(i) = number of (u, i, t) tuples in the corpus, normalised
  to a per-item rank. (Pure click-count weighting.)
- locked-formula ψᵢ = mean ψ(u, i, t) per mp_00 §10. (Engagement-time and
  reflective-act and complexity-corrected.)

Note: the two estimators are computed from the *same* tuples in the *same*
corpus; the difference is the aggregation rule. F5 therefore tests whether
the locked formula adds something beyond a per-item count statistic.
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


def run(shard_path: Path | str, min_encounters: int = 5, **kwargs) -> dict:
    shard_path = Path(shard_path)
    df = pd.read_parquet(shard_path)

    item_g = df.groupby("item_id", sort=False)
    counts = item_g.size()
    psi_mean = item_g["psi"].mean()

    # Filter to items with at least min_encounters
    keep = counts[counts >= min_encounters].index
    if len(keep) < 30:
        return {"shard": str(shard_path), "skipped": "<30 items meet threshold",
                "n_items": int(len(keep))}

    counts_f = counts.loc[keep]
    psi_f = psi_mean.loc[keep]

    rho, p = stats.spearmanr(counts_f, psi_f)
    pearson, _ = stats.pearsonr(np.log1p(counts_f.to_numpy()), psi_f.to_numpy())

    if rho > 0.95:
        verdict = "POPULARITY_EQUALS_FORMULA"
    elif rho >= 0.6:
        verdict = "MODERATE_OVERLAP"
    else:
        verdict = "FORMULA_DIFFERS_FROM_POPULARITY"

    return {
        "shard": str(shard_path),
        "test": "F5_popularity_overlap",
        "min_encounters": min_encounters,
        "n_items": int(len(keep)),
        "spearman_rho_count_vs_psi": float(rho),
        "pearson_r_log_count_vs_psi": float(pearson),
        "p_value": float(p),
        "verdict": verdict,
        "interpretation": {
            "POPULARITY_EQUALS_FORMULA": "popularity is the locked-formula estimator; corpus contribution is procedural.",
            "MODERATE_OVERLAP": "formula and popularity rank items similarly but differ on specific populations.",
            "FORMULA_DIFFERS_FROM_POPULARITY": "formula and popularity measure substantially different per-item statistics.",
        }[verdict],
        "popularity_summary": {
            "mean_count": float(counts_f.mean()),
            "median_count": float(counts_f.median()),
            "max_count": int(counts_f.max()),
        },
        "psi_summary": {
            "mean": float(psi_f.mean()),
            "std": float(psi_f.std()),
            "min": float(psi_f.min()),
            "max": float(psi_f.max()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--min-encounters", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(Path(args.shard), min_encounters=args.min_encounters)
    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
