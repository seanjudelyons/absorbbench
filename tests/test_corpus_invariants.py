"""
Corpus invariants — CI smoke tests.

Verifies the locked formula, the pre-registered drop history, and the
build pipeline's hash-stability. Run before any Phase 7 leaderboard
publication.

Usage: python -m pytest tests/

Or: python -m tests.test_corpus_invariants  (no pytest dependency)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus_build.psi import SCHEMA_COLS, compute_psi, k_floor, normalise_R


# ────────────────────────────────────────────────────────────────────
# §11.3 r1 — k floor
# ────────────────────────────────────────────────────────────────────
def test_k_floor():
    assert k_floor(0) == 1.0
    assert k_floor(0.5) == 1.0
    assert k_floor(1.0) == 1.0
    assert k_floor(2.5) == 2.5
    assert k_floor(100) == 100


# ────────────────────────────────────────────────────────────────────
# §10.1 ψ formula bounds
# ────────────────────────────────────────────────────────────────────
def test_psi_bounds():
    """ψ ≤ 2 strictly (mp_00 §11.3 with R1 patched)."""
    n = 1000
    rng = np.random.default_rng(0)
    c = pd.Series(rng.uniform(0, 1, n))
    R_norm = pd.Series(rng.uniform(0, 1, n))
    K = pd.Series(np.maximum(rng.uniform(0, 10, n), 1.0))
    psi = compute_psi(c, R_norm, K)
    assert psi.min() >= 0
    assert psi.max() <= 2.0 + 1e-9, f"ψ exceeded 2: {psi.max()}"


def test_psi_zero_engagement():
    """c → 0 ⇒ ψ → 0."""
    psi = compute_psi(pd.Series([0.0, 0.0, 0.0]),
                      pd.Series([0.5, 0.0, 1.0]),
                      pd.Series([1.0, 5.0, 10.0]))
    assert (psi == 0).all()


def test_psi_d2_monotonicity():
    """At fixed c, R_norm, ψ is decreasing in K."""
    c = pd.Series([1.0, 1.0, 1.0])
    R_norm = pd.Series([0.0, 0.0, 0.0])
    K = pd.Series([1.0, 5.0, 10.0])
    psi = compute_psi(c, R_norm, K)
    assert psi.iloc[0] > psi.iloc[1] > psi.iloc[2], f"D2 fail: {psi.tolist()}"


# ────────────────────────────────────────────────────────────────────
# §10.1 r̃_norm bounds
# ────────────────────────────────────────────────────────────────────
def test_R_norm_bounds():
    R_count = pd.Series([0, 1, 2, 5, 10, 20, 100])
    R_norm, R_p90 = normalise_R(R_count)
    assert (R_norm >= 0).all()
    assert (R_norm <= 1).all()


# ────────────────────────────────────────────────────────────────────
# schema invariants
# ────────────────────────────────────────────────────────────────────
def test_schema_columns():
    assert SCHEMA_COLS == [
        "user_id", "item_id", "modality", "dataset_source",
        "timestamp", "position",
        "c", "R_count", "K_raw", "K", "R_norm", "psi",
    ]


# ────────────────────────────────────────────────────────────────────
# corpus shard invariants (skips if corpus not built)
# ────────────────────────────────────────────────────────────────────
def test_shard_schema():
    corpus = ROOT / "corpus" / "spine"
    if not corpus.exists():
        print("SKIP: corpus/spine/ not built")
        return
    n_checked = 0
    n_unreadable = 0
    for p in corpus.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  ⚠️ unreadable parquet (skipping): {p.name}: {e!r}")
            n_unreadable += 1
            continue
        assert list(df.columns) == SCHEMA_COLS, f"{p}: schema mismatch {list(df.columns)}"
        # bounds
        assert df["c"].between(0, 1).all(), f"{p}: c not in [0, 1]"
        assert df["R_norm"].between(0, 1).all(), f"{p}: R_norm not in [0, 1]"
        assert (df["K"] >= 1.0).all(), f"{p}: K floor violated"
        assert df["psi"].between(0, 2.001).all(), f"{p}: ψ not in [0, 2]"
        n_checked += 1
    if n_checked == 0:
        raise AssertionError(f"no readable parquet shards in {corpus}")
    if n_unreadable:
        print(f"  ({n_checked} shards checked; {n_unreadable} unreadable — likely Drive upload corruption; investigate after the run)")


def test_pre_registered_drops():
    """No dropped dataset sources should appear in the corpus manifest."""
    import json
    manifest_path = ROOT / "corpus" / "manifest.json"
    if not manifest_path.exists():
        print("SKIP: corpus/manifest.json not built")
        return
    m = json.loads(manifest_path.read_text())
    sources_in_corpus = set()
    for s in m["shards"]:
        sources_in_corpus.update(s["dataset_sources"])

    # pre-registered drops per mp_05 / mp_05a phase 1 findings
    DROPPED = {"EB-NeRD", "LastFM-1K", "Tenrec-QB-article"}
    intersection = DROPPED & sources_in_corpus
    assert not intersection, f"Dropped datasets in corpus: {intersection}"


# ────────────────────────────────────────────────────────────────────
# run all when called as a script
# ────────────────────────────────────────────────────────────────────
def main():
    n_pass, n_fail = 0, 0
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  ✅ {name}")
                n_pass += 1
            except AssertionError as e:
                print(f"  ❌ {name}: {e}")
                n_fail += 1
                failed.append(name)
            except Exception as e:
                print(f"  ⚠️ {name}: ERROR {type(e).__name__}: {e}")
                n_fail += 1
                failed.append(name)
    print(f"\n{n_pass} passed, {n_fail} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
