#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out_dir="${TMPDIR:-/tmp}/retrievalbench-demo-$$"
mkdir -p "$out_dir"

# Evaluate the example corpus at document granularity.
python3 -m retrievalbench evaluate \
  --corpus examples/data/corpus.jsonl \
  --queries examples/data/queries.jsonl \
  --qrels examples/data/qrels.jsonl \
  -k 3 \
  --json "$out_dir/eval.json" \
  --markdown "$out_dir/eval.md"

# Chunked indexing should keep perfect recall on this corpus.
python3 -m retrievalbench evaluate \
  --corpus examples/data/corpus.jsonl \
  --queries examples/data/queries.jsonl \
  --qrels examples/data/qrels.jsonl \
  -k 3 --chunk-tokens 30 --chunk-overlap 5 \
  --json "$out_dir/eval-chunked.json" > /dev/null

# An ad-hoc search against the same corpus.
python3 -m retrievalbench search --corpus examples/data/corpus.jsonl -k 3 \
  "what metric rewards placing highly relevant documents first" > "$out_dir/search.txt"

python3 - <<PY
import json
result = json.load(open("$out_dir/eval.json", encoding="utf-8"))
assert result["corpus_size"] == 12
assert result["query_count"] == 6
averages = result["averages"]
assert averages["recall_at_3"] == 1.0, averages
assert averages["reciprocal_rank"] == 1.0, averages
assert averages["ndcg_at_3"] > 0.9, averages
per_query = {row["query_id"]: row for row in result["queries"]}
assert per_query["q-hybrid"]["recall_at_3"] == 1.0
md = open("$out_dir/eval.md", encoding="utf-8").read()
assert md.startswith("# RetrievalBench evaluation\n")
assert "| Judged queries | 6 |" in md
assert "| q-bm25 |" in md
chunked = json.load(open("$out_dir/eval-chunked.json", encoding="utf-8"))
assert chunked["corpus_size"] > 12
assert chunked["averages"]["recall_at_3"] == 1.0, chunked["averages"]
search = open("$out_dir/search.txt", encoding="utf-8").read()
assert "evaluation-metrics" in search.splitlines()[1], search
PY

# Quality gates: perfect recall passes, an impossible gate fails.
python3 -m retrievalbench evaluate \
  --corpus examples/data/corpus.jsonl \
  --queries examples/data/queries.jsonl \
  --qrels examples/data/qrels.jsonl \
  -k 3 --min-recall 1.0 --min-mrr 0.99 > /dev/null
if python3 -m retrievalbench evaluate \
  --corpus examples/data/corpus.jsonl \
  --queries examples/data/queries.jsonl \
  --qrels examples/data/qrels.jsonl \
  -k 3 --min-ndcg 1.01 > "$out_dir/gate.txt" 2>&1; then
  echo "expected the impossible nDCG gate to fail" >&2
  exit 1
fi
cat "$out_dir/gate.txt"
printf 'demo artifacts: %s\n' "$out_dir"
