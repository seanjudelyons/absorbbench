"""
F1 — Within-item user-half consistency. Per mp_08 §3.

Split the users who encountered each item into two halves; estimate ψᵢ
on each half; compute Spearman ρ across items.

Pre-registered threshold (mp_08 §3 F1): ρ ≥ 0.5 on items with ≥30
encounters.
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


def run(shard_path: Path | str, min_encounters: int = 30, seed: int = 0,
        **kwargs) -> dict:
    shard_path = Path(shard_path)
    df = pd.read_parquet(shard_path)
    rng = np.random.default_rng(seed)

    # per-item: split encounters into two halves
    item_psi_a = {}
    item_psi_b = {}
    item_n = {}
    for item_id, grp in df.groupby("item_id", sort=False):
        if len(grp) < min_encounters:
            continue
        idx = rng.permutation(len(grp))
        half = len(idx) // 2
        psi_arr = grp["psi"].to_numpy()
        item_psi_a[item_id] = float(psi_arr[idx[:half]].mean())
        item_psi_b[item_id] = float(psi_arr[idx[half:]].mean())
        item_n[item_id] = len(grp)

    if len(item_psi_a) < 30:
        return {"shard": str(shard_path), "skipped": "fewer than 30 items meet min_encounters",
                "n_items": len(item_psi_a)}

    a = pd.Series(item_psi_a)
    b = pd.Series(item_psi_b)
    rho, p = stats.spearmanr(a, b)
    pearson, _ = stats.pearsonr(a, b)

    verdict = "PASS" if rho >= 0.5 else "FAIL"
    return {
        "shard": str(shard_path),
        "test": "F1_user_half",
        "min_encounters": min_encounters,
        "n_items": len(item_psi_a),
        "spearman_rho": float(rho),
        "pearson_r": float(pearson),
        "p_value": float(p),
        "threshold": 0.5,
        "verdict": verdict,
        "median_n_per_item": float(pd.Series(item_n).median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--min-encounters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(Path(args.shard), min_encounters=args.min_encounters, seed=args.seed)
    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
