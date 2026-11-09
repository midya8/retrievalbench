"""Pure-Python BM25 indexing and scoring."""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[A-Za-z0-9]+")


def _finite(value) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(value)
    except (TypeError, OverflowError):
        return False


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization, deterministic across platforms."""
    return TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over an in-memory corpus.

    k1 controls term-frequency saturation and b controls document-length
    normalization. Scores are exact (no pruning), so results are fully
    reproducible and suitable for golden-metric regression tests.
    """

    def __init__(self, documents: list[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75):
        if not _finite(k1) or k1 <= 0:
            raise ValueError("k1 must be positive")
        if not _finite(b) or not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        seen: set[str] = set()
        for doc_id, text in documents:
            if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
                raise ValueError("document ids must be nonempty and unique")
            seen.add(doc_id)
            terms = tokenize(text)
            if not terms:
                continue
            self.doc_ids.append(doc_id)
            counts = Counter(terms)
            self._term_freqs.append(counts)
            self._lengths.append(len(terms))
            for term in counts:
                document_frequency[term] += 1
        if not self.doc_ids:
            raise ValueError("corpus has no non-empty documents")
        self._avg_length = sum(self._lengths) / len(self._lengths)
        total = len(self.doc_ids)
        # Positive log1p IDF variant. The +1 inside log keeps this positive;
        # max(0) is redundant for valid N/df, not a floor of raw RSJ log odds.
        self._idf = {
            term: max(0.0, math.log((total - df + 0.5) / (df + 0.5) + 1.0))
            for term, df in document_frequency.items()
        }

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def _score(self, query_terms: list[str], index: int) -> float:
        counts = self._term_freqs[index]
        length_norm = 1.0 - self.b + self.b * self._lengths[index] / self._avg_length
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            if self.k1 > 1e150:
                saturation = tf * (1.0 + 1.0 / self.k1) / (tf / self.k1 + length_norm)
                score += self._idf.get(term, 0.0) * saturation
            else:
                score += self._idf.get(term, 0.0) * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
        return score

    def rank(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Return (doc_id, score) pairs for all documents, best first.

        Ties break by document id so rankings are deterministic.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = tokenize(query)
        scored = sorted(
            ((self._score(terms, i), self.doc_ids[i]) for i in range(self.size)),
            key=lambda item: (-item[0], item[1]),
        )
        return [(doc_id, score) for score, doc_id in scored[:top_k]]
