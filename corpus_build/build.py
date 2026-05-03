"""
corpus_build.build
==================

Deterministic end-to-end corpus build. Reads raw on-disk dataset files,
applies the locked mp_00 §10 ψ formula via corpus_build.extractors, and
writes per-dataset parquet shards plus a manifest with content hashes.

Output structure:
  corpus/
    manifest.json                 — per-shard hashes + row counts
    spine/
      kuairand_pure.parquet
      tenrec_qk_article.parquet
    appendix/
      kuairec.parquet
      tenrec_qk_video.parquet      (sampled to 20M for ≤50GB target)
      tenrec_qb_video.parquet
      microlens_100k.parquet       (if available)
      otto.parquet                  (if available; conditional)

Determinism:
  - Source row counts logged in manifest.
  - Sort by (dataset_source, user_id, position) before writing.
  - Sample-with-seed for QK-video sampling.

Run:
  python -m corpus_build.build [--full] [--spine-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus_build.extractors import (
    DATASETS,
    kuairand_pure_rows,
    kuairec_rows,
    tenrec_qb_video_rows,
    tenrec_qk_article_rows,
    tenrec_qk_video_rows,
)
from corpus_build.psi import SCHEMA_COLS

CORPUS_OUT = ROOT / "corpus"
QK_VIDEO_SAMPLE_SIZE = 20_000_000  # 20M rows from 493M for ≤50GB target


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_shard(df: pd.DataFrame, out_path: Path, label: str) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Stable sort for determinism
    df = df.sort_values(["dataset_source", "user_id", "position"]).reset_index(drop=True)
    df = df[SCHEMA_COLS]
    t0 = time.time()
    df.to_parquet(out_path, index=False, compression="snappy")
    dt = time.time() - t0

    sha = file_sha256(out_path)
    size = out_path.stat().st_size
    return {
        "label": label,
        "path": str(out_path.relative_to(ROOT)),
        "rows": len(df),
        "sha256": sha,
        "size_bytes": size,
        "size_mb": round(size / 1e6, 2),
        "write_seconds": round(dt, 2),
        "modality_counts": df["modality"].value_counts().to_dict(),
        "dataset_sources": df["dataset_source"].unique().tolist(),
        "psi_summary": {
            "mean": round(float(df["psi"].mean()), 6),
            "std": round(float(df["psi"].std()), 6),
            "min": round(float(df["psi"].min()), 6),
            "max": round(float(df["psi"].max()), 6),
        },
        "k_summary": {
            "mean": round(float(df["K"].mean()), 4),
            "std": round(float(df["K"].std()), 4),
        },
    }


def build_spine(out_root: Path, full: bool) -> list:
    """Build the 2-dataset spine."""
    shards = []

    print("\n[spine] KuaiRand-Pure...")
    n = None if full else 200_000
    df = kuairand_pure_rows(n=n)
    shards.append(write_shard(df, out_root / "spine" / "kuairand_pure.parquet", "KuaiRand-Pure"))
    print(f"  → {shards[-1]['rows']:,} rows; {shards[-1]['size_mb']} MB; sha {shards[-1]['sha256'][:16]}...")

    print("\n[spine] Tenrec QK-article...")
    n = None if full else 500_000
    df = tenrec_qk_article_rows(n=n)
    shards.append(write_shard(df, out_root / "spine" / "tenrec_qk_article.parquet", "Tenrec-QK-article"))
    print(f"  → {shards[-1]['rows']:,} rows; {shards[-1]['size_mb']} MB; sha {shards[-1]['sha256'][:16]}...")

    return shards


def build_appendix(out_root: Path, full: bool) -> list:
    shards = []

    print("\n[appendix] KuaiRec...")
    n = None if full else 100_000
    df = kuairec_rows(n=n)
    shards.append(write_shard(df, out_root / "appendix" / "kuairec.parquet", "KuaiRec"))
    print(f"  → {shards[-1]['rows']:,} rows; {shards[-1]['size_mb']} MB")

    print("\n[appendix] Tenrec QB-video...")
    n = None if full else 100_000
    df = tenrec_qb_video_rows(n=n)
    shards.append(write_shard(df, out_root / "appendix" / "tenrec_qb_video.parquet", "Tenrec-QB-video"))
    print(f"  → {shards[-1]['rows']:,} rows; {shards[-1]['size_mb']} MB")

    print("\n[appendix] Tenrec QK-video (sampled)...")
    n = QK_VIDEO_SAMPLE_SIZE if full else 200_000
    df = tenrec_qk_video_rows(n=n)
    shards.append(write_shard(df, out_root / "appendix" / "tenrec_qk_video.parquet", "Tenrec-QK-video"))
    print(f"  → {shards[-1]['rows']:,} rows; {shards[-1]['size_mb']} MB")

    return shards


def write_manifest(out_root: Path, shards: list, full: bool) -> Path:
    total_rows = sum(s["rows"] for s in shards)
    total_size = sum(s["size_bytes"] for s in shards)

    manifest = {
        "build_id": "absorbbench_corpus",
        "version": "0.1.0-phase2-pilot",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_mode": "full" if full else "smoke",
        "psi_formula_source": "research_notes/mp_00_psi_derivation.md §10",
        "schema_columns": SCHEMA_COLS,
        "total_rows": total_rows,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / 1e6, 2),
        "shards": shards,
        "spine_dataset_sources": [
            "KuaiRand-Pure",
            "Tenrec-QK-article",
        ],
        "appendix_dataset_sources": [
            "KuaiRec",
            "Tenrec-QB-video",
            "Tenrec-QK-video",
        ],
        "phase1_findings_ref": "research_notes/mp_05a_phase1_findings.md",
        "construct_validity_status": "withdrawn per mp_03 §7; pillar 4 reframes to orthogonality-evidence",
        "modalities_in_corpus": ["short_video", "text_news"],
        "permanently_dropped": [
            "EB-NeRD (K_text fails K-validation; mp_05a Phase 1 Findings §1.2)",
            "Music modality (mp_05 §3.1a — foreground-attention scope)",
            "Tenrec QB-article (click-only schema)",
        ],
    }
    out_path = out_root / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Run full build (uses entire datasets)")
    ap.add_argument("--spine-only", action="store_true", help="Skip appendix datasets")
    ap.add_argument("--out", default=str(CORPUS_OUT), help="Output corpus root directory")
    args = ap.parse_args()

    out_root = Path(args.out)
    print(f"# AbsorbBench corpus build")
    print(f"Mode: {'FULL' if args.full else 'SMOKE'}")
    print(f"Output: {out_root.relative_to(ROOT) if out_root.is_absolute() else out_root}")
    print(f"Spine only: {args.spine_only}")

    shards = build_spine(out_root, full=args.full)
    if not args.spine_only:
        shards.extend(build_appendix(out_root, full=args.full))

    manifest_path = write_manifest(out_root, shards, full=args.full)
    print(f"\n## Build complete")
    print(f"Manifest written: {manifest_path.relative_to(ROOT)}")
    total_rows = sum(s["rows"] for s in shards)
    total_size = sum(s["size_bytes"] for s in shards) / 1e6
    print(f"Total rows: {total_rows:,}; total size: {total_size:.2f} MB across {len(shards)} shards")


if __name__ == "__main__":
    main()
