"""
corpus_build.splits
====================

Deterministic train/val/test splits per mp_07 §5:
- Held-out items: 20% test / 10% val / 70% train, by SHA-256(item_id) mod 1000
- Held-out users: 20% test / 10% val / 70% train, by SHA-256(user_id) mod 1000
- Held-out time:  10% test / 5% val / 85% train, by per-dataset timestamp/position rank

Vectorised — does NOT iterate over groupby groups.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _hash_to_bucket(values: pd.Series, seed: int = 0, modulus: int = 1000) -> pd.Series:
    """Vectorised: stable hash of each value into [0, modulus).

    For a Series of ~10^7 values, this should run in seconds, not minutes.
    """
    # compute hash on unique values only, then map back.
    unique = values.unique()

    def _h(s: str) -> int:
        return (int(hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:8], 16) ^ seed) % modulus

    bucket_lookup = {v: _h(v) for v in unique}
    return values.map(bucket_lookup).astype(np.int32)


def items_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 0) -> pd.Series:
    bucket = _hash_to_bucket(df["item_id"], seed=seed, modulus=1000)
    test_lo = 0
    test_hi = int(test_frac * 1000)
    val_lo = test_hi
    val_hi = test_hi + int(val_frac * 1000)
    out = np.full(len(df), "train", dtype=object)
    out[(bucket >= test_lo) & (bucket < test_hi)] = "test"
    out[(bucket >= val_lo) & (bucket < val_hi)] = "val"
    return pd.Series(out, index=df.index)


def users_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 0) -> pd.Series:
    bucket = _hash_to_bucket(df["user_id"], seed=seed, modulus=1000)
    test_lo = 0
    test_hi = int(test_frac * 1000)
    val_lo = test_hi
    val_hi = test_hi + int(val_frac * 1000)
    out = np.full(len(df), "train", dtype=object)
    out[(bucket >= test_lo) & (bucket < test_hi)] = "test"
    out[(bucket >= val_lo) & (bucket < val_hi)] = "val"
    return pd.Series(out, index=df.index)


def time_split(df: pd.DataFrame, test_frac: float = 0.10, val_frac: float = 0.05) -> pd.Series:
    """
    Last `test_frac` rows by per-dataset (timestamp or position) → test.
    Previous `val_frac` rows → val. Rest → train.
    """
    out = pd.Series("train", index=df.index)
    for source, idx in df.groupby("dataset_source").groups.items():
        sub = df.loc[idx]
        sort_col = "timestamp" if sub["timestamp"].notna().any() else "position"
        sorted_idx = sub.sort_values([sort_col, "user_id"]).index
        n = len(sorted_idx)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        if n_test > 0:
            out.loc[sorted_idx[-n_test:]] = "test"
        if n_val > 0:
            out.loc[sorted_idx[-(n_test + n_val):-n_test]] = "val"
    return out
