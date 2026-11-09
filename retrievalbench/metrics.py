"""Ranking quality metrics computed per query and aggregated."""

from __future__ import annotations

import math


def _ranking(ranked: list[str], k: int | None = None) -> None:
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k <= 0):
        raise ValueError("k must be a positive integer")
    if len(set(ranked)) != len(ranked):
        raise ValueError("rankings must contain unique document ids")


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of the top-k results that are relevant (grade > 0)."""
    _ranking(ranked, k)
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of all relevant documents found in the top-k results."""
    _ranking(ranked, k)
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant result, 0 when none appears."""
    _ranking(ranked)
    for position, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    """Normalized discounted cumulative gain at k with graded relevance."""
    _ranking(ranked, k)

    if any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in grades.values()):
        raise ValueError("grades must be nonnegative integers")
    maximum = max(grades.values(), default=0)
    def gain(grade: int) -> float:
        if maximum <= 100:
            return 2.0 ** grade - 1.0
        delta = grade - maximum
        return (0.0 if delta < -1074 else 2.0 ** delta) - (0.0 if maximum > 1074 else 2.0 ** (-maximum))

    def dcg(doc_ids: list[str]) -> float:
        return sum(
            gain(grades.get(doc_id, 0)) / math.log2(position + 1)
            for position, doc_id in enumerate(doc_ids, 1)
        )

    ideal = sorted(grades, key=lambda doc_id: grades[doc_id], reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(ranked[:k]) / ideal_dcg


def evaluate_run(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    *,
    k: int = 10,
) -> dict:
    """Score a run (query_id -> ranked doc ids) against qrels.

    Returns per-query metric rows and macro-averaged aggregates. Queries
    without judgments are skipped; every judged query needs a ranking entry
    (an empty ranking scores zero rather than being silently dropped).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rows: list[dict] = []
    for query_id in sorted(qrels):
        grades = qrels[query_id]
        relevant = {doc_id for doc_id, grade in grades.items() if grade > 0}
        if not relevant:
            continue
        ranked = rankings.get(query_id)
        if ranked is None:
            raise ValueError(f"missing ranking for judged query {query_id!r}")
        rows.append(
            {
                "query_id": query_id,
                f"precision_at_{k}": precision_at_k(ranked, relevant, k),
                f"recall_at_{k}": recall_at_k(ranked, relevant, k),
                "reciprocal_rank": reciprocal_rank(ranked, relevant),
                f"ndcg_at_{k}": ndcg_at_k(ranked, grades, k),
                "relevant_count": len(relevant),
                "retrieved_count": len(ranked[:k]),
            }
        )
    if not rows:
        raise ValueError("no judged queries with relevant documents")
    metric_names = [name for name in rows[0] if name not in {"query_id", "relevant_count", "retrieved_count"}]
    averages = {name: sum(row[name] for row in rows) / len(rows) for name in metric_names}
    return {"query_count": len(rows), "k": k, "averages": averages, "queries": rows}
