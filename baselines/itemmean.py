"""
itemmean baseline for the ψ-prediction task.

for each item in the test set, predict the per-item mean ψ computed on
the train set. items not seen in train get the global train-set mean.
this is the recsys-literature itemmean / itemavg baseline (sometimes
miscalled "popularity" — popularity is count-based, not mean-of-target;
see baselines/popularity.py for the count-based version).

trivial; cpu-only; runs in seconds on the full corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    item_mean = train.groupby("item_id")["psi"].mean()
    global_mean = float(train["psi"].mean())
    pred = test["item_id"].map(item_mean).fillna(global_mean)
    return pred.astype(float)


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(float).reset_index(drop=True)
    y_pred = y_pred.astype(float).reset_index(drop=True)
    err = y_true - y_pred
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "true_mean": float(y_true.mean()),
        "pred_mean": float(y_pred.mean()),
        "true_std": float(y_true.std()),
        "pred_std": float(y_pred.std()),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }


def run(corpus_root: Path = ROOT / "corpus") -> dict:
    """Run popularity baseline across all spine + appendix shards under each split."""
    from corpus_build.splits import items_split, time_split, users_split

    results = {}
    for shard_path in sorted((corpus_root / "spine").glob("*.parquet")):
        try:
            df = pd.read_parquet(shard_path)
        except Exception as e:
            print(f"  ⚠️ skipping unreadable shard {shard_path.name}: {e!r}")
            results[f"{shard_path.stem}/SKIPPED"] = {"reason": "unreadable_parquet", "error": repr(e)}
            continue
        for split_name, split_fn in [("items", items_split), ("users", users_split), ("time", time_split)]:
            mask = split_fn(df)
            train = df[mask == "train"]
            test = df[mask == "test"]
            if len(test) < 100 or len(train) < 100:
                continue
            pred = fit_predict(train, test)
            m = metrics(test["psi"], pred)
            results[f"{shard_path.stem}/{split_name}"] = m
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus"))
    ap.add_argument("--out", default=str(ROOT / "baselines" / "results_popularity.json"))
    args = ap.parse_args()

    results = run(Path(args.corpus))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")
    for k, v in results.items():
        print(f"  {k}: MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, Spearman={v['spearman']:.4f}, n={v['n']}")


if __name__ == "__main__":
    main()
