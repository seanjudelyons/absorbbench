"""
corpus_build.lix
================

LIX (Björnsson) Scandinavian readability index. Used as the Danish
external K-anchor for EB-NeRD per mp_07 §3.

LIX = (n_words / n_sentences) + (n_long_words / n_words) × 100
where long_word = >6 letters.

Implemented inline; no external dependency. Pure-Python; works on
short or long texts. Returns 0 for empty / one-word texts.
"""

from __future__ import annotations

import re

# Word boundary that handles Danish letters æøå plus standard ASCII.
_WORD_RE = re.compile(r"\b[\wÆØÅæøå]+\b", re.UNICODE)
_SENT_RE = re.compile(r"[.!?]+")


def lix(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = _WORD_RE.findall(text)
    n_words = len(words)
    if n_words < 2:
        return 0.0
    n_long = sum(1 for w in words if len(w) > 6)
    n_sentences = max(1, len(_SENT_RE.findall(text)))
    return (n_words / n_sentences) + (n_long / n_words) * 100.0
