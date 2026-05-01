"""
corpus_build.psi
================

The locked formula from mp_00 §10. Single source of truth for ψ across
all pilot scripts and the corpus build.

ψ(u, i, t) = N_oesis(u, i, t) / K(i)
N_oesis  = c × (1 + R_norm)
c        = T_engaged / T_total          [0, 1]
R_norm   = min(R_count, R_p90) / R_p90  [0, 1]
K(i)     = max(K_raw, 1)                bits, R1 floor (mp_00 §11.3)

Aggregations (mp_00 §10.2):
- ψ_total(s) = Σ ψᵢ
- ψ_mean(s)  = mean ψᵢ
- (A_s, β_s) from OLS log ψᵢ vs position
- peak_position(s) = argmax_i ψᵢ
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# unified parquet schema (mp_05a §6.1, extended in §6.2 with `position`).
SCHEMA_COLS = [
    "user_id", "item_id", "modality", "dataset_source",
    "timestamp", "position",
    "c", "R_count", "K_raw", "K", "R_norm", "psi",
]


def k_floor(k_raw: float) -> float:
    """R1 from mp_00 §11.3 — 1-bit floor."""
    return max(float(k_raw), 1.0)


def normalise_R(R_count: pd.Series, percentile: float = 90.0) -> tuple[pd.Series, float]:
    """R̃_norm per mp_00 §10.1. Returns (R_norm series, R_p90 used)."""
    pos = R_count[R_count >= 1]
    if len(pos) == 0:
        return pd.Series(0.0, index=R_count.index), 0.0
    R_p = float(np.percentile(pos, percentile))
    if R_p <= 0:
        return pd.Series(0.0, index=R_count.index), 0.0
    return (R_count.clip(upper=R_p) / R_p).astype(float), R_p


def compute_psi(c: pd.Series, R_norm: pd.Series, K: pd.Series) -> pd.Series:
    """Vectorised ψ. K must already be floored."""
    return c * (1.0 + R_norm) / K


def session_descriptors(rows: pd.DataFrame, session_id_col: str = "session_id") -> pd.DataFrame:
    """Compute per-session aggregations (mp_00 §10.2). Returns DataFrame indexed by session id."""
    if session_id_col not in rows.columns:
        raise KeyError(f"rows must have a '{session_id_col}' column")

    out_rows = []
    for sid, grp in rows.groupby(session_id_col, sort=False):
        psi_arr = grp["psi"].to_numpy()
        pos_arr = grp["position"].to_numpy()
        if len(psi_arr) < 2:
            A, beta = np.nan, np.nan
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = np.log(np.clip(psi_arr, 1e-9, None))
            try:
                m, b = np.polyfit(pos_arr, lp, 1)
                A, beta = float(np.exp(b)), float(-m)  # ψ ≈ a exp(-β·pos)
            except (np.linalg.LinAlgError, ValueError):
                A, beta = np.nan, np.nan

        peak_pos = int(pos_arr[np.argmax(psi_arr)]) if len(psi_arr) else -1
        out_rows.append({
            "session_id": sid,
            "n_items": len(psi_arr),
            "psi_total": float(psi_arr.sum()),
            "psi_mean": float(psi_arr.mean()),
            "A": A,
            "beta": beta,
            "peak_position": peak_pos,
        })
    return pd.DataFrame(out_rows).set_index("session_id")
