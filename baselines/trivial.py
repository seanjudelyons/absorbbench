"""
Trivial-control baselines:
  - random: uniform predictions in [0, max(ψ_train)]
  - global_mean: predict train-set mean ψ for every test row
  - user_mean: per-user mean ψ from train (for users-split this falls
               back to global mean; included only for items/time splits)
  - mlp_tabular: small MLP on (user_id, item_id) embeddings (deferred —
               needs torch; documented).

Each baseline reports MAE, RMSE, Spearman ρ, n.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = pd.Series(y_true.values, dtype=float)
    y_pred = pd.Series(y_pred.values, dtype=float)
    err = y_true - y_pred
    if y_pred.std() < 1e-12:
        spearman = float("nan")
    else:
        try:
            spearman = float(y_true.corr(y_pred, method="spearman"))
        except Exception:
            spearman = float("nan")
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "spearman": spearman,
        "true_mean": float(y_true.mean()),
        "pred_mean": float(y_pred.mean()),
    }


def random_predict(test: pd.DataFrame, train: pd.DataFrame, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(0, float(train["psi"].max()), size=len(test)))


def global_mean_predict(test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    m = float(train["psi"].mean())
    return pd.Series([m] * len(test))


def user_mean_predict(test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    user_mean = train.groupby("user_id")["psi"].mean()
    g = float(train["psi"].mean())
    return test["user_id"].map(user_mean).fillna(g).astype(float).reset_index(drop=True)


def run(corpus_root: Path = ROOT / "corpus") -> dict:
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
            for bl_name, bl_fn in [("random", random_predict),
                                   ("global_mean", global_mean_predict),
                                   ("user_mean", user_mean_predict)]:
                pred = bl_fn(test.reset_index(drop=True), train)
                m = metrics(test["psi"].reset_index(drop=True), pred)
                results[f"{shard_path.stem}/{split_name}/{bl_name}"] = m
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus"))
    ap.add_argument("--out", default=str(ROOT / "baselines" / "results_trivial.json"))
    args = ap.parse_args()

    results = run(Path(args.corpus))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}\n")
    print(f"| dataset | split | baseline | MAE | RMSE | Spearman | n |")
    print(f"|---|---|---|---|---|---|---|")
    for k, v in results.items():
        ds, split, bl = k.split("/")
        sp = f"{v['spearman']:.4f}" if not np.isnan(v['spearman']) else "n/a"
        print(f"| {ds} | {split} | {bl} | {v['mae']:.4f} | {v['rmse']:.4f} | {sp} | {v['n']:,} |")


if __name__ == "__main__":
    main()
