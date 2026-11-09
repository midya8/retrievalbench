# RetrievalBench

Offline retrieval evaluation for RAG and search pipelines.

RetrievalBench builds an exact Okapi BM25 index over a local JSONL corpus, ranks golden queries, and scores the rankings against relevance judgments with recall@k, precision@k, MRR and graded nDCG@k. It is dependency-free and runs fully offline, so retrieval quality becomes a reproducible CI gate instead of a vibes check.

## Why

Retrieval changes - chunk sizes, overlap, index parameters, document edits - silently shift which passages a pipeline surfaces. RetrievalBench turns those changes into measurable regressions:

- exact BM25 (no pruning, no ANN approximation), so scores and rankings are fully deterministic and diffable;
- token-window chunking with overlap, with chunk hits collapsed back to their parent document so qrels stay at document granularity;
- per-query and macro-averaged recall@k, precision@k, MRR and nDCG@k from graded qrels;
- quality gates (`--min-recall`, `--min-precision`, `--min-mrr`, `--min-ndcg`) that fail CI when a change drops retrieval quality;
- JSON reports for tooling and Markdown reports for CI job summaries.

## Commands

Evaluate a corpus against golden queries and qrels:

```bash
python3 -m retrievalbench evaluate \
  --corpus examples/data/corpus.jsonl \
  --queries examples/data/queries.jsonl \
  --qrels examples/data/qrels.jsonl \
  -k 3 --json eval.json --markdown eval.md
```

Gate on retrieval quality:

```bash
python3 -m retrievalbench evaluate \
  --corpus corpus.jsonl --queries queries.jsonl --qrels qrels.jsonl \
  -k 5 --min-recall 0.9 --min-mrr 0.8
```

Evaluate chunked indexing (chunks collapse to their parent document for scoring):

```bash
python3 -m retrievalbench evaluate \
  --corpus corpus.jsonl --queries queries.jsonl --qrels qrels.jsonl \
  -k 5 --chunk-tokens 120 --chunk-overlap 20
```

Run one ad-hoc query:

```bash
python3 -m retrievalbench search --corpus corpus.jsonl -k 5 "hybrid lexical and vector search"
```

Tune the ranking function with `--k1` (term-frequency saturation) and `--b` (length normalization).

## Data formats

All inputs are JSONL, one object per line.

- **Corpus:** `{"id": "doc-1", "text": "..."}` (`doc_id` also accepted)
- **Queries:** `{"id": "q-1", "text": "..."}` (`query_id` also accepted)
- **Qrels:** `{"query_id": "q-1", "doc_id": "doc-1", "relevance": 2}` - non-negative integer grades, `relevance` defaults to 1, grades above 0 count as relevant and nDCG uses the grade

## Verification

- 42 unit tests cover tokenization, BM25 scoring edge cases (ties, empty documents, unknown terms, length normalization), JSONL loading validation, chunking windows and overlap, every metric, report escaping and the CLI including gates and chunked evaluation.
- `bash examples/demo.sh` runs fully offline against a small AI/ML corpus and asserts the aggregate metrics, the Markdown report, chunked-evaluation parent collapsing, and both sides of the quality gates.
- CI runs the same suite across Python 3.10-3.13.

BM25 is a lexical baseline: it measures term-matching retrieval, not embedding quality. Graded qrels are illustrative judgments for the example corpus, not a research benchmark.

## License

MIT
