from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import __version__
from .bm25 import BM25Index
from .corpus import chunk_documents, chunk_records, load_documents, load_qrels, load_queries
from .metrics import evaluate_run
from .report import render_markdown, render_terminal

GATE_METRICS = ("recall", "precision", "ndcg", "mrr")


def _build_index(args) -> tuple[BM25Index, int]:
    documents = load_documents(args.corpus)
    if args.chunk_tokens:
        records = chunk_records(documents, max_tokens=args.chunk_tokens, overlap=args.chunk_overlap)
        args.parent_map = {r["id"]:r["parent"] for r in records}
        documents = [(r["id"],r["text"]) for r in records]
    if args.k <= 0:
        raise ValueError("k must be positive")
    return BM25Index(documents, k1=args.k1, b=args.b), len(documents)


def cmd_evaluate(args) -> int:
    try:
        for label in GATE_METRICS:
            gate = getattr(args, "min_"+label)
            if gate is not None and (not math.isfinite(gate) or gate < 0):
                raise ValueError("gates must be finite and nonnegative")
        index, corpus_size = _build_index(args)
        queries = load_queries(args.queries)
        qrels = load_qrels(args.qrels)
    except ValueError as exc:
        print(f"ERROR {exc}")
        return 2
    rankings = {}
    for query_id, text in queries:
        ranked = [doc_id for doc_id, _ in index.rank(text, top_k=args.k)]
        if args.chunk_tokens:
            # A document is retrieved when any of its chunks ranks; collapse
            # chunk ids to their parent document, keeping the best rank.
            seen: set[str] = set()
            collapsed = []
            for doc_id in ranked:
                parent = args.parent_map[doc_id]
                if parent not in seen:
                    seen.add(parent)
                    collapsed.append(parent)
            ranked = collapsed
        rankings[query_id] = ranked
    try:
        result = evaluate_run(rankings, qrels, k=args.k)
    except ValueError as exc:
        print(f"ERROR {exc}")
        return 2
    payload = {
        "schema_version": 1,
        "tool": f"retrievalbench {__version__}",
        "corpus_size": corpus_size,
        "parameters": {"k1": args.k1, "b": args.b, "chunk_tokens": args.chunk_tokens, "chunk_overlap": args.chunk_overlap},
        **result,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(result, corpus_size=corpus_size), encoding="utf-8", newline="\n")
    print(render_terminal(result, corpus_size=corpus_size))
    averages = result["averages"]
    k = args.k
    gates = {
        "recall": (args.min_recall, averages.get(f"recall_at_{k}")),
        "precision": (args.min_precision, averages.get(f"precision_at_{k}")),
        "ndcg": (args.min_ndcg, averages.get(f"ndcg_at_{k}")),
        "mrr": (args.min_mrr, averages.get("reciprocal_rank")),
    }
    failures = [
        f"{label} {actual:.4f} below gate {threshold:.4f}"
        for label, (threshold, actual) in gates.items()
        if threshold is not None and actual is not None and actual < threshold
    ]
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


def cmd_search(args) -> int:
    try:
        index, corpus_size = _build_index(args)
    except ValueError as exc:
        print(f"ERROR {exc}")
        return 2
    results = index.rank(args.query, top_k=args.k)
    print(f"corpus {corpus_size} document(s); top {len(results)} for {args.query!r}")
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"{rank:>3}. {doc_id}  {score:.4f}")
    return 0


def cmd_audit(args) -> int:
    from .audit import audit, load_spec
    try:
        result = audit(load_spec(args.spec))
        Path(args.json).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"ERROR {exc}")
        return 2
    print(f"audited {len(result['units'])} units, {len(result['terms'])} query terms")
    return 0


def cmd_fuse(args) -> int:
    from .fusion import fuse, load_spec
    try:
        result = fuse(load_spec(args.spec))
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"ERROR {exc}")
        return 2
    print(f"fused {len(result['sources'])} lists; {len(result['rows'])} candidates; {len(result['selected'])} selected")
    return 0


def cmd_fuse_search(args) -> int:
    from .fusion import fuse_queries
    try:
        result = fuse_queries(load_documents(args.corpus), args.query, constant=args.constant,
                              k=args.k, depth=args.depth, weights=args.weight, k1=args.k1, b=args.b)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"ERROR {exc}")
        return 2
    print(f"fused {len(result['queries'])} explicit BM25 query variants; selected {len(result['fusion']['selected'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrievalbench", description="Offline retrieval evaluation with exact BM25 and golden metrics.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_index_flags(command) -> None:
        command.add_argument("--corpus", required=True, help="JSONL documents with id and text")
        command.add_argument("--chunk-tokens", type=int, default=0, help="split documents into token windows of this size (0 disables)")
        command.add_argument("--chunk-overlap", type=int, default=20, help="overlap between chunk windows")
        command.add_argument("--k1", type=float, default=1.5, help="BM25 term-frequency saturation")
        command.add_argument("--b", type=float, default=0.75, help="BM25 length normalization")
        command.add_argument("-k", type=int, default=10, help="ranking cutoff")

    evaluate = sub.add_parser("evaluate", help="rank golden queries and score against qrels")
    add_index_flags(evaluate)
    evaluate.add_argument("--queries", required=True, help="JSONL queries with id and text")
    evaluate.add_argument("--qrels", required=True, help="JSONL relevance judgments with query_id, doc_id, relevance")
    evaluate.add_argument("--json", help="write the machine-readable JSON report")
    evaluate.add_argument("--markdown", help="write a Markdown report for CI job summaries")
    evaluate.add_argument("--min-recall", type=float, help="exit 1 when recall@k falls below this gate")
    evaluate.add_argument("--min-precision", type=float, help="exit 1 when precision@k falls below this gate")
    evaluate.add_argument("--min-ndcg", type=float, help="exit 1 when nDCG@k falls below this gate")
    evaluate.add_argument("--min-mrr", type=float, help="exit 1 when MRR falls below this gate")
    evaluate.set_defaults(func=cmd_evaluate)

    audit_parser = sub.add_parser("audit", help="complete BM25, parent budget and metric witnesses")
    audit_parser.add_argument("--spec", required=True)
    audit_parser.add_argument("--json", required=True)
    audit_parser.set_defaults(func=cmd_audit)

    fusion = sub.add_parser("fuse", help="exact weighted reciprocal rank fusion of supplied lists")
    fusion.add_argument("--spec", required=True)
    fusion.add_argument("--json", required=True)
    fusion.set_defaults(func=cmd_fuse)

    fused_search = sub.add_parser("fuse-search", help="original exhaustive BM25 query variants, then exact rank fusion")
    fused_search.add_argument("--corpus", required=True)
    fused_search.add_argument("--query", action="append", required=True, help="explicit query variant; repeat 1..8 times")
    fused_search.add_argument("--weight", type=int, action="append", help="integer weight per query, or omit all for equal weights")
    fused_search.add_argument("--constant", type=int, default=60)
    fused_search.add_argument("--depth", type=int, default=10)
    fused_search.add_argument("-k", type=int, default=10)
    fused_search.add_argument("--k1", type=float, default=1.5)
    fused_search.add_argument("--b", type=float, default=.75)
    fused_search.add_argument("--json", required=True)
    fused_search.set_defaults(func=cmd_fuse_search)

    search = sub.add_parser("search", help="run one ad-hoc query against the corpus")
    add_index_flags(search)
    search.add_argument("query", help="query text")
    search.set_defaults(func=cmd_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
