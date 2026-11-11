"""Exact, bounded weighted reciprocal rank fusion of explicitly supplied rankings.

Standard RRF (Cormack, Clarke, Buettcher, SIGIR 2009), not a relevance model.
No source score enters fusion. See FUSION.md for candidate and tie policies.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from .bm25 import BM25Index

SCHEMA = "retrievalbench-rrf/v1"


def _int(v, low, high, label):
    if type(v) is not int or not low <= v <= high:
        raise ValueError(f"{label} must be an integer in [{low},{high}]")
    return v


def _id(v, label):
    if not isinstance(v, str) or not 1 <= len(v) <= 64 or any(not 32 <= ord(c) <= 126 for c in v):
        raise ValueError(f"{label} must be 1..64 printable ASCII characters")
    return v


def _keys(v, keys, label):
    if not isinstance(v, dict) or set(v) != set(keys):
        raise ValueError(f"{label} needs exactly {','.join(keys)}")


def validate(spec):
    _keys(spec, ("schema", "constant", "k", "lists"), "spec")
    if spec["schema"] != SCHEMA:
        raise ValueError("unsupported fusion schema")
    c = _int(spec["constant"], 0, 1_000_000, "constant")
    k = _int(spec["k"], 0, 64, "k")
    lists = spec["lists"]
    if not isinstance(lists, list) or not 1 <= len(lists) <= 8:
        raise ValueError("need 1..8 lists")
    names = set(); universe = set(); clean = []
    for source in lists:
        _keys(source, ("id", "weight", "depth", "ranking"), "list")
        name = _id(source["id"], "list id")
        if name in names:
            raise ValueError("duplicate list id")
        names.add(name)
        weight = _int(source["weight"], 0, 100, "weight")
        depth = _int(source["depth"], 0, 64, "depth")
        ranking = source["ranking"]
        if not isinstance(ranking, list) or len(ranking) > 64:
            raise ValueError("ranking needs 0..64 document ids")
        ranking = [_id(d, "document id") for d in ranking]
        if len(set(ranking)) != len(ranking):
            raise ValueError("duplicate document in a source ranking")
        universe.update(ranking)
        clean.append(dict(id=name, weight=weight, depth=depth, ranking=ranking))
    if len(universe) > 64:
        raise ValueError("at most 64 unique document ids across all rankings")
    if not sum(s["weight"] for s in clean):
        raise ValueError("at least one positive list weight required")
    return dict(schema=SCHEMA, constant=c, k=k, lists=clean)


def fraction(value):
    """JSON-safe exact fraction: decimal strings, never JSON integer overflow."""
    return dict(n=str(value.numerator), d=str(value.denominator))


def fuse(spec):
    """Return all candidates, complete contribution matrix and exact cutoff gap.

    Candidate = union of the admitted prefixes, even for zero-weight lists.
    Unknown/beyond-depth documents have no contribution, not a last-place rank.
    A repeated ranking under a different source id is deliberately another vote.
    """
    spec = validate(spec); c = spec["constant"]; sources = spec["lists"]
    candidates = sorted({d for s in sources for d in s["ranking"][:s["depth"]]})
    positions = [{d: i for i, d in enumerate(s["ranking"], 1)} for s in sources]
    scores = {}; rows = []
    for doc in candidates:
        total = Fraction(); contributions = []
        for s, ranks in zip(sources, positions):
            rank = ranks.get(doc)
            admitted = rank is not None and rank <= s["depth"]
            value = Fraction(s["weight"], c + rank) if admitted else Fraction()
            total += value
            contributions.append(dict(source=s["id"], rank=rank, admitted=admitted,
                                      reason="included" if admitted else "missing" if rank is None else "beyond_depth",
                                      weight=s["weight"], denominator=c + rank if admitted else None,
                                      contribution=fraction(value)))
        scores[doc] = total
        rows.append(dict(doc=doc, score=fraction(total), contributions=contributions))
    rows.sort(key=lambda r: (-scores[r["doc"]], r["doc"]))
    for i, row in enumerate(rows, 1):
        row.update(rank=i, selected=i <= spec["k"])
    k = spec["k"]
    gap = None
    if 0 < k < len(rows):
        left = rows[k - 1]["doc"]; right = rows[k]["doc"]
        gap = dict(inside=left, outside=right, difference=fraction(scores[left] - scores[right]),
                   tie=scores[left] == scores[right])
    duplicates = []
    for i, s in enumerate(sources):
        for t in sources[:i]:
            if s["ranking"] == t["ranking"] and s["depth"] == t["depth"]:
                duplicates.append([t["id"], s["id"]])
    entries = [dict(id=s["id"], weight=s["weight"], depth=s["depth"],
                    entries=[dict(doc=d, rank=i, admitted=i <= s["depth"], candidate=d in scores)
                             for i, d in enumerate(s["ranking"], 1)]) for s in sources]
    return dict(schema="retrievalbench-rrf-report/v1", spec=spec, sources=entries, rows=rows,
                selected=[r["doc"] for r in rows if r["selected"]],
                excluded=sorted({d for s in sources for d in s["ranking"]} - set(candidates)),
                identical_rankings=duplicates, cutoff=gap,
                upper_bound=fraction(sum((Fraction(s["weight"], c + 1) for s in sources), Fraction())))


def load_spec(path):
    raw = Path(path).read_bytes()
    if len(raw) > 131072:
        raise ValueError("fusion input exceeds 128 KiB")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    def reject(value):
        raise ValueError("fusion JSON permits integer numbers only")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject, parse_float=reject)


def fuse_queries(documents, queries, *, constant=60, k=10, depth=10, weights=None, k1=1.5, b=.75):
    """Build actual exhaustive original BM25 rankings, then fuse their prefixes.

    Query variants are supplied, not generated. This does not create independent
    retrievers or an embedding pipeline. Original BM25 tie and empty-doc rules apply.
    """
    if not isinstance(documents, list) or not 1 <= len(documents) <= 64:
        raise ValueError("need 1..64 corpus documents")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 8:
        raise ValueError("need 1..8 query strings")
    if any(not isinstance(q, str) or len(q) > 4096 for q in queries):
        raise ValueError("queries must be strings of at most 4096 characters")
    for doc in documents:
        if not isinstance(doc, (tuple, list)) or len(doc) != 2 or not isinstance(doc[1], str) or len(doc[1]) > 65536:
            raise ValueError("documents need bounded id/text pairs")
        _id(doc[0], "document id")
    weights = [1] * len(queries) if weights is None else weights
    if not isinstance(weights, list) or len(weights) != len(queries):
        raise ValueError("one weight required per query")
    # Validate configuration before constructing potentially expensive index.
    validate(dict(schema=SCHEMA, constant=constant, k=k, lists=[dict(id=f"q{i}", weight=w, depth=depth, ranking=[]) for i, w in enumerate(weights, 1)]))
    index = BM25Index(documents, k1=k1, b=b)
    rankings = [index.rank(q, top_k=index.size) for q in queries]
    spec = dict(schema=SCHEMA, constant=constant, k=k, lists=[dict(id=f"q{i}", weight=w, depth=depth, ranking=[d for d, _ in ranking]) for i, (w, ranking) in enumerate(zip(weights, rankings), 1)])
    return dict(schema="retrievalbench-rrf-bm25/v1", documents=[dict(id=d, text=t) for d, t in documents],
                k1=k1, b=b, omitted=[d for d, _ in documents if d not in index.doc_ids],
                queries=[dict(id=f"q{i}", text=q, ranking=[dict(doc=d, score=v) for d, v in ranking]) for i, (q, ranking) in enumerate(zip(queries, rankings), 1)], fusion=fuse(spec))
