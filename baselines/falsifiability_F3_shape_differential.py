"""
F3 — Engagement-shape differential vs popularity. Per mp_08 §3.

Tests whether ψᵢ correlates with the *upper-tail* of P(c | item, click=1)
distinct from per-item click-rate (popularity).

Operationally:
  - For each item, compute high-completion-rate = P(c ≥ 0.8 | (u, i, t) ∈ corpus).
  - Compute per-item click-rate = mean(c > 0) (proxy: positive engagement).
  - Compute per-item ψᵢ = mean ψ(u, i, t).
  - Test: partial correlation between ψᵢ and high-completion-rate
    controlling for click-rate.

Pre-registered: rank-correlation ≥ 0.4 *after* partialling out click-rate.
Operationalised as Spearman ρ between residuals of (ψᵢ ~ click-rate)
and (high-completion-rate ~ click-rate).
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


def _residuals(y: pd.Series, x: pd.Series) -> pd.Series:
    """OLS residuals of y ~ x."""
    X = np.vstack([np.ones(len(x)), x.to_numpy()]).T
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    return y - (X @ beta)


def run(shard_path: Path | str, min_encounters: int = 30, **kwargs) -> dict:
    shard_path = Path(shard_path)
    df = pd.read_parquet(shard_path)

    item_g = df.groupby("item_id", sort=False)
    n = item_g.size()
    psi_mean = item_g["psi"].mean()
    high_completion = item_g["c"].apply(lambda s: float((s >= 0.8).mean()))
    click_rate = item_g["c"].apply(lambda s: float((s > 0).mean()))

    keep = n[n >= min_encounters].index
    if len(keep) < 30:
        return {"shard": str(shard_path), "skipped": "<30 items meet threshold",
                "n_items": int(len(keep))}

    psi_f = psi_mean.loc[keep]
    hc_f = high_completion.loc[keep]
    cr_f = click_rate.loc[keep]

    # partial correlation: residualise both psi_f and hc_f against cr_f, then correlate residuals.
    psi_res = _residuals(psi_f, cr_f)
    hc_res = _residuals(hc_f, cr_f)
    rho, p = stats.spearmanr(psi_res, hc_res)
    raw_rho, _ = stats.spearmanr(psi_f, hc_f)

    verdict = "PASS" if rho >= 0.4 else "FAIL"
    return {
        "shard": str(shard_path),
        "test": "F3_shape_differential",
        "min_encounters": min_encounters,
        "n_items": int(len(keep)),
        "raw_spearman_psi_vs_high_completion": float(raw_rho),
        "partial_spearman_psi_vs_high_completion_given_click_rate": float(rho),
        "p_value": float(p),
        "threshold": 0.4,
        "verdict": verdict,
        "click_rate_summary": {
            "mean": float(cr_f.mean()),
            "min": float(cr_f.min()),
            "max": float(cr_f.max()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--min-encounters", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(Path(args.shard), min_encounters=args.min_encounters)
    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
