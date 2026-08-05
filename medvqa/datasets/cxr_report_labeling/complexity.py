"""Unnormalized inverse-frequency complexity scores for reports."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def build_vocab_frequencies(texts: Iterable[str]) -> Dict[str, int]:
    freq: Counter = Counter()
    for text in texts:
        freq.update(tokenize(text))
    return dict(freq)


def complexity_score(text: str, vocab_freq: Dict[str, int]) -> float:
    """Unnormalized sum_i 1/freq[token_i]. Favors longer reports with rare tokens."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    score = 0.0
    for tok in tokens:
        f = vocab_freq.get(tok, 1)
        score += 1.0 / float(f)
    return score


def score_reports(reports: Sequence[dict], text_key: str = "report_text") -> List[float]:
    texts = [r[text_key] for r in reports]
    vocab = build_vocab_frequencies(texts)
    return [complexity_score(t, vocab) for t in texts]
