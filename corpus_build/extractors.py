"""
corpus_build.extractors
=======================

Per-dataset row extractors that produce the unified mp_05a §6.1 schema
under the locked mp_00 §10 ψ formula. One canonical place for the
ψ-extraction logic; pilot scripts import from here.

Public API per dataset:
  kuairand_pure_rows(n=None, slice_filter=None) -> DataFrame
  kuairec_rows(n=None) -> DataFrame
  ebnerd_rows(n=None, k_method="lexical_proxy") -> DataFrame
  tenrec_qk_article_rows(n=None) -> DataFrame
  tenrec_qk_video_rows(n=None) -> DataFrame
  tenrec_qb_video_rows(n=None) -> DataFrame
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .psi import (
    SCHEMA_COLS,
    compute_psi,
    k_floor,
    normalise_R,
)
from .lix import lix as compute_lix

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def _surprisal_marginal(values: pd.Series) -> dict:
    """
    Return {value: -log2 P(value | corpus marginal)}.

    NaN values are excluded from the marginal (pandas value_counts default).
    Lookups for values not in the dict should fall through to NaN in the
    caller, which then `.fillna(0).map(k_floor)` resolves to K=1 (R1 floor).
    """
    p = values.value_counts(normalize=True).to_dict()
    return {v: -math.log2(prob) for v, prob in p.items() if prob > 0}


def kuairand_pure_rows(n: int | None = None,
                       slice_filter: str | None = None) -> pd.DataFrame:
    """
    KuaiRand-Pure short-video rows.
    slice_filter: 'random' (is_rand==1), 'standard' (is_rand==0), or None (both).
    """
    base = DATASETS / "KuaiRand-Pure" / "data"
    paths = []
    if slice_filter in (None, "random"):
        paths.append(base / "log_random_4_22_to_5_08_pure.csv")
    if slice_filter in (None, "standard"):
        paths.append(base / "log_standard_4_08_to_4_21_pure.csv")
        paths.append(base / "log_standard_4_22_to_5_08_pure.csv")
    chunks = [pd.read_csv(p, nrows=n) for p in paths if p.exists()]
    log = pd.concat(chunks, ignore_index=True)
    if n is not None:
        log = log.head(n)

    vf = pd.read_csv(base / "video_features_basic_pure.csv")
    vf = vf.dropna(subset=["tag"]).copy()
    surp = _surprisal_marginal(vf["tag"])
    vf["K_raw"] = vf["tag"].map(surp)
    log = log.merge(vf[["video_id", "K_raw"]], on="video_id", how="left")

    log["c"] = (log["play_time_ms"].astype(float)
                / log["duration_ms"].clip(lower=1).astype(float)).clip(0, 1)
    refl_cols = ["is_like", "is_follow", "is_comment", "is_forward",
                 "long_view", "is_profile_enter"]
    log["R_count"] = log[refl_cols].sum(axis=1).astype(int)
    log["R_norm"], _ = normalise_R(log["R_count"])
    log["K"] = log["K_raw"].fillna(0).map(k_floor)
    log["psi"] = compute_psi(log["c"], log["R_norm"], log["K"])
    log["position"] = log.sort_values(["user_id", "time_ms"]).groupby("user_id").cumcount()

    return pd.DataFrame({
        "user_id": "kr_" + log["user_id"].astype(str),
        "item_id": "kr_" + log["video_id"].astype(str),
        "modality": "short_video",
        "dataset_source": "KuaiRand-Pure",
        "timestamp": pd.to_datetime(log["time_ms"], unit="ms"),
        "position": log["position"].astype(int),
        "c": log["c"].astype(float),
        "R_count": log["R_count"].astype(int),
        "K_raw": log["K_raw"].astype(float),
        "K": log["K"].astype(float),
        "R_norm": log["R_norm"].astype(float),
        "psi": log["psi"].astype(float),
    })[SCHEMA_COLS]


def kuairec_rows(n: int | None = None) -> pd.DataFrame:
    base = DATASETS / "KuaiRec" / "KuaiRec 2.0" / "data"
    log = pd.read_csv(base / "small_matrix.csv", nrows=n)
    cats = pd.read_csv(base / "item_categories.csv")
    cats["feat_list"] = cats["feat"].map(
        lambda s: [int(x) for x in re.findall(r"\d+", str(s))] if isinstance(s, str) else []
    )
    all_feats = [f for fl in cats["feat_list"] for f in fl]
    feat_freq = pd.Series(all_feats).value_counts(normalize=True).to_dict() if all_feats else {}

    def k_raw_from_feat(fl):
        if not fl:
            return 0.0
        if len(fl) == 1:
            return -math.log2(feat_freq.get(fl[0], 1e-6))
        return math.log2(len(fl))

    cats["K_raw"] = cats["feat_list"].map(k_raw_from_feat)
    log = log.merge(cats[["video_id", "K_raw"]], on="video_id", how="left")
    log["c"] = log["watch_ratio"].clip(0, 1)
    log["R_count"] = 0
    log["R_norm"] = 0.0
    log["K"] = log["K_raw"].fillna(0).map(k_floor)
    log["psi"] = compute_psi(log["c"], log["R_norm"], log["K"])
    log["position"] = log.sort_values(["user_id", "timestamp"]).groupby("user_id").cumcount()
    return pd.DataFrame({
        "user_id": "kc_" + log["user_id"].astype(str),
        "item_id": "kc_" + log["video_id"].astype(str),
        "modality": "short_video",
        "dataset_source": "KuaiRec",
        "timestamp": pd.to_datetime(log["timestamp"].astype(float), unit="s"),
        "position": log["position"].astype(int),
        "c": log["c"].astype(float),
        "R_count": log["R_count"].astype(int),
        "K_raw": log["K_raw"].astype(float),
        "K": log["K"].astype(float),
        "R_norm": log["R_norm"].astype(float),
        "psi": log["psi"].astype(float),
    })[SCHEMA_COLS]


def ebnerd_rows(n: int | None = None,
                k_method: str = "lexical_proxy",
                variant: str = "demo") -> pd.DataFrame:
    """
    EB-NeRD news rows.
    k_method: 'lexical_proxy' (fast, mp_05a §6.1 form) or 'lix' (canonical Danish anchor).
    variant: 'demo' or 'small'.
    """
    base = DATASETS / f"ebnerd_{variant}"
    beh = pd.read_parquet(base / "train" / "behaviors.parquet")
    arts = pd.read_parquet(base / "articles.parquet")

    if k_method == "lix":
        arts["K_raw"] = arts["body"].map(compute_lix)
    elif k_method == "lexical_proxy":
        stopwords = set("og er på som det at en jeg vi de en til af med har en det den".split())

        def k_raw_text(body: str) -> float:
            if not isinstance(body, str) or not body.strip():
                return 0.0
            words = re.findall(r"\b[a-zæøåA-ZÆØÅ]+\b", body.lower())
            words = [w for w in words if w not in stopwords and len(w) > 1]
            if len(words) < 5:
                return 0.0
            n_sent = max(1, body.count(".") + body.count("?") + body.count("!"))
            wc = Counter(words)
            total = sum(wc.values())
            H = -sum((c/total) * math.log2(c/total) for c in wc.values())
            return H / max(1, math.log2(n_sent + 1))

        arts["K_raw"] = arts["body"].map(k_raw_text)
    else:
        raise ValueError(f"unknown k_method: {k_method}")

    arts["word_count"] = arts["body"].fillna("").map(lambda s: len(s.split()))

    beh = beh[beh["article_id"].notna()].copy()
    if n is not None:
        beh = beh.head(n)
    beh = beh.merge(arts[["article_id", "K_raw", "word_count"]], on="article_id", how="left")
    beh["est_read_seconds"] = (beh["word_count"].fillna(0) / 250.0) * 60.0
    beh["c"] = (beh["read_time"].fillna(0)
                / beh["est_read_seconds"].clip(lower=1)).clip(0, 1)
    beh["R_count"] = 0
    beh["R_norm"] = 0.0
    beh["K"] = beh["K_raw"].fillna(0).map(k_floor)
    beh["psi"] = compute_psi(beh["c"], beh["R_norm"], beh["K"])
    beh["position"] = beh.sort_values(["user_id", "impression_time"]).groupby("user_id").cumcount()
    return pd.DataFrame({
        "user_id": "eb_" + beh["user_id"].astype(str),
        "item_id": "eb_" + beh["article_id"].astype(int).astype(str),
        "modality": "text_news",
        "dataset_source": "EB-NeRD",
        "timestamp": beh["impression_time"],
        "position": beh["position"].astype(int),
        "c": beh["c"].astype(float),
        "R_count": beh["R_count"].astype(int),
        "K_raw": beh["K_raw"].astype(float),
        "K": beh["K"].astype(float),
        "R_norm": beh["R_norm"].astype(float),
        "psi": beh["psi"].astype(float),
    })[SCHEMA_COLS]


def _tenrec_qk_article_load(n: int | None) -> pd.DataFrame:
    return pd.read_csv(DATASETS / "tenrec" / "QK-article.csv", nrows=n)


def tenrec_qk_article_rows(n: int | None = None) -> pd.DataFrame:
    df = _tenrec_qk_article_load(n)
    refl_cols = ["share", "like", "follow", "favorite"]
    df[refl_cols] = df[refl_cols].astype(bool).astype(int)
    df["R_count"] = df[refl_cols].sum(axis=1)
    df["read_bool"] = df["read"].astype(bool)
    df["c"] = np.where(df["read_bool"],
                       df["read_percentage"].fillna(0).astype(float) / 100.0,
                       0.0).clip(0, 1)
    df["cat_pair"] = df["category_first"].astype(str) + "_" + df["category_second"].astype(str)
    surp = _surprisal_marginal(df["cat_pair"])
    df["K_raw"] = df["cat_pair"].map(surp).fillna(0.0)
    df["R_norm"], _ = normalise_R(df["R_count"])
    df["K"] = df["K_raw"].map(k_floor)
    df["psi"] = compute_psi(df["c"], df["R_norm"], df["K"])
    df["position"] = df.groupby("user_id").cumcount()
    return pd.DataFrame({
        "user_id": "tr_qka_" + df["user_id"].astype(str),
        "item_id": "tr_qka_" + df["item_id"].astype(str),
        "modality": "text_news",
        "dataset_source": "Tenrec-QK-article",
        "timestamp": pd.NaT,
        "position": df["position"].astype(int),
        "c": df["c"].astype(float),
        "R_count": df["R_count"].astype(int),
        "K_raw": df["K_raw"].astype(float),
        "K": df["K"].astype(float),
        "R_norm": df["R_norm"].astype(float),
        "psi": df["psi"].astype(float),
    })[SCHEMA_COLS]


def _tenrec_video_rows(filename: str, source_name: str, n: int | None) -> pd.DataFrame:
    df = pd.read_csv(DATASETS / "tenrec" / filename, nrows=n)
    df[["follow", "like", "share"]] = df[["follow", "like", "share"]].astype(int)
    df["R_count"] = df[["follow", "like", "share"]].sum(axis=1)
    df["c"] = df["click"].astype(float).clip(0, 1)
    surp = _surprisal_marginal(df["video_category"])
    df["K_raw"] = df["video_category"].map(surp).fillna(0.0)
    df["R_norm"], _ = normalise_R(df["R_count"])
    df["K"] = df["K_raw"].map(k_floor)
    df["psi"] = compute_psi(df["c"], df["R_norm"], df["K"])
    df["position"] = df.groupby("user_id").cumcount()
    src_short = source_name.split('-')[-1].lower()
    return pd.DataFrame({
        "user_id": f"tr_{src_short}_" + df["user_id"].astype(str),
        "item_id": f"tr_{src_short}_" + df["item_id"].astype(str),
        "modality": "short_video",
        "dataset_source": source_name,
        "timestamp": pd.NaT,
        "position": df["position"].astype(int),
        "c": df["c"].astype(float),
        "R_count": df["R_count"].astype(int),
        "K_raw": df["K_raw"].astype(float),
        "K": df["K"].astype(float),
        "R_norm": df["R_norm"].astype(float),
        "psi": df["psi"].astype(float),
    })[SCHEMA_COLS]


def tenrec_qk_video_rows(n: int | None = None) -> pd.DataFrame:
    return _tenrec_video_rows("QK-video.csv", "Tenrec-QK-video", n)


def tenrec_qb_video_rows(n: int | None = None) -> pd.DataFrame:
    return _tenrec_video_rows("QB-video.csv", "Tenrec-QB-video", n)
