"""Corpus, query and qrels loading plus document chunking."""

from __future__ import annotations

import json
from pathlib import Path

from .bm25 import tokenize


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of objects, skipping blank lines."""
    records: list[dict] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        records.append(record)
    return records


def load_documents(path: str | Path) -> list[tuple[str, str]]:
    """Load documents from JSONL with `id` (or `doc_id`) and `text` fields."""
    documents: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in load_jsonl(path):
        doc_id = record.get("id", record.get("doc_id"))
        text = record.get("text")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{path}: every document needs a non-empty string id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{doc_id}: document text must be a non-empty string")
        if doc_id in seen:
            raise ValueError(f"{path}: duplicate document id {doc_id!r}")
        seen.add(doc_id)
        documents.append((doc_id, text))
    if not documents:
        raise ValueError(f"{path}: no documents found")
    return documents


def load_queries(path: str | Path) -> list[tuple[str, str]]:
    """Load queries from JSONL with `id` (or `query_id`) and `text` fields."""
    queries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in load_jsonl(path):
        query_id = record.get("id", record.get("query_id"))
        text = record.get("text")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{path}: every query needs a non-empty string id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{query_id}: query text must be a non-empty string")
        if query_id in seen:
            raise ValueError(f"{path}: duplicate query id {query_id!r}")
        seen.add(query_id)
        queries.append((query_id, text))
    if not queries:
        raise ValueError(f"{path}: no queries found")
    return queries


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load relevance judgments: query_id -> {doc_id: grade}.

    JSONL records carry `query_id`, `doc_id` and an integer `relevance`
    (0 = not relevant, higher = more relevant).
    """
    qrels: dict[str, dict[str, int]] = {}
    for record in load_jsonl(path):
        query_id = record.get("query_id")
        doc_id = record.get("doc_id")
        relevance = record.get("relevance", 1)
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{path}: every qrel needs a non-empty query_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{path}: every qrel needs a non-empty doc_id")
        if not isinstance(relevance, int) or isinstance(relevance, bool) or relevance < 0:
            raise ValueError(f"{path}:{query_id}: relevance must be a non-negative integer")
        if doc_id in qrels.get(query_id, {}):
            raise ValueError(f"{path}: duplicate qrel pair {query_id!r}, {doc_id!r}")
        qrels.setdefault(query_id, {})[doc_id] = relevance
    if not qrels:
        raise ValueError(f"{path}: no relevance judgments found")
    return qrels


def chunk_records(documents: list[tuple[str, str]], *, max_tokens: int = 120, overlap: int = 20) -> list[dict]:
    """Windows with explicit parent identity and half-open token offsets.

    Preserve the historical trailing overlapping windows, including a final
    window wholly covered by its predecessor. Identity is never parsed by '#c'.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < max_tokens:
        raise ValueError("overlap must be in [0, max_tokens)")
    chunks, seen, parents = [], set(), set()
    for doc_id, text in documents:
        if not isinstance(doc_id, str) or not doc_id or doc_id in parents:
            raise ValueError("document ids must be nonempty and unique")
        parents.add(doc_id)
        terms = tokenize(text)
        starts = [0] if len(terms) <= max_tokens else range(0, len(terms), max_tokens-overlap)
        for i, start in enumerate(starts):
            end = min(len(terms), start+max_tokens)
            chunk_id = doc_id if len(terms) <= max_tokens else f"{doc_id}#c{i}"
            if chunk_id in seen:
                raise ValueError("chunk id collision; rename the conflicting document")
            seen.add(chunk_id)
            chunks.append({"id":chunk_id,"parent":doc_id,"start":start,"end":end,
                           "text":text if len(terms)<=max_tokens else " ".join(terms[start:end])})
    return chunks


def chunk_documents(documents: list[tuple[str, str]], *, max_tokens: int = 120, overlap: int = 20) -> list[tuple[str, str]]:
    """Historical token windows; use chunk_records for unambiguous parent ids."""
    return [(r["id"],r["text"]) for r in chunk_records(documents,max_tokens=max_tokens,overlap=overlap)]
