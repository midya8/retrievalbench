# RetrievalBench

RetrievalBench ranks documents with BM25 and evaluates supplied queries
against graded relevance judgments. Inspect every term contribution, chunk and
parent selection, and per-rank metric calculation. It runs offline without
runtime dependencies.

[Live retrieval inspector](https://midya8.github.io/retrievalbench/) · [Evidence contract, sources and limits](EVIDENCE.md) · [Original README](docs/original-readme.md)

## BM25 evaluation and rank fusion

`search` and `evaluate` use the original exhaustive BM25 index. `audit` exposes
bounded ranking and metric arithmetic. `fuse` combines supplied ranked lists;
`fuse-search` first ranks explicit query variants with BM25. Fusion scores are
exact rational rank calculations, not evidence of better relevance.

## Exact multi-ranking fusion

[Open the rank-fusion inspector](https://midya8.github.io/retrievalbench/fusion.html) · [RRF contract and prior work](FUSION.md)

The opt-in `fuse` and `fuse-search` commands implement standard weighted reciprocal rank fusion. `fuse-search` runs the original exhaustive BM25 index on explicit query variants before fusing their prefixes. Every candidate/source fraction, missing or beyond-depth rank, repeated vote, zero-weight candidate and exact cutoff tie stays visible. Python Fraction and independent BigInt replay all 230 complete reports exactly. This is rank arithmetic, not a new algorithm or evidence of relevance improvement. Existing BM25 commands remain unchanged.

```bash
python -m retrievalbench fuse --spec examples/fusion.json --json fusion.json
python -m retrievalbench fuse-search --corpus examples/data/corpus.jsonl \
  --query 'bread yeast' --query 'fermentation' --depth 3 -k 3 --json fused.json
```

The current suite has 80 Python tests, including all 68 earlier tests, and seven Node/actual-Chrome tests, including all three earlier tests. These checks cover 432 exhaustive rank controls, 200 conservation/order/scaling controls, 100 real-BM25 pipeline comparisons, 230 full native/browser reports and isolated installed-wheel reproduction. [Full bounds, reproducible counts and limits](FUSION.md).

## BM25 and parent-selection audits

The original CLI builds an exhaustive BM25 index over JSONL documents, evaluates golden queries against graded judgments and gates precision/recall/MRR/nDCG. The bounded `audit` command exposes the complete ranked list, query-term contributions, corpus statistics, token windows, parent collapse and per-rank/ideal DCG rows. Independent scalar JavaScript recomputes the entire report in an editable static inspector; no API, model or stored-score lookup.

Taking k chunks before collapse can return fewer than k parents. Scanning until k distinct parents is a different budget. Both are shown on the same fixed ranking. Whole-document versus chunk indexing also changes N, df and average length, so it is not an isolated chunk-quality effect. Zero-score units remain eligible. Judgments are illustrative, not a general retrieval benchmark.

## Run

```bash
python -m retrievalbench evaluate --corpus examples/data/corpus.jsonl \
  --queries examples/data/queries.jsonl --qrels examples/data/qrels.jsonl \
  -k 3 --min-recall 1 --json eval.json --markdown eval.md
python -m retrievalbench search --corpus examples/data/corpus.jsonl -k 3 'BM25 relevance'
python -m retrievalbench audit --spec examples/audit.json --json audit.json
```

Chunked evaluation adds `--chunk-tokens 30 --chunk-overlap 5`. The default CLI retains k-units-before-collapse; it is not silently changed to the distinct-parent alternative. Tune `--k1` and `--b`. Gates: `--min-recall`, `--min-precision`, `--min-mrr`, `--min-ndcg`. All gates compare full-precision values, while terminal/Markdown display rounds.

Data remains JSONL: corpus/query records have id/text; qrels have query_id/doc_id/relevance (default 1, nonnegative integer). Duplicate ids/pairs are errors. See the original README for the full command examples and formats.

## Original BM25 checks

- At the BM25 audit release, 68 Python tests included all 42 originals. 35 complete reports include all six original queries under whole/chunk indexing and 23 explicit controls.
- Independent JavaScript compares 27,694 numeric and 12,052 other scalar values. That release's three Node/actual-Chrome tests check every ranked unit, contribution/metric row, source text, both policies, edits/export and 320/390/1280 containment.
- Python 3.10-3.13 regeneration and isolated installed-wheel verification with eight module origins.
- Original corpus, queries, qrels, demo and tests unchanged; original whole/chunked evaluation JSON remains byte-identical. Hashes and original artifacts are retained under docs/.

```bash
python -m unittest discover -s tests -v
bash examples/demo.sh
PYTHONPATH=. python tools/build_audits.py
npm test  # Node 22+ and Chrome; CHROME_BIN can select Chrome
bash tools/wheel_smoke.sh
```

## Explicit fixes and limits

Parent ids are carried as data rather than split on '#c'. Generated-id collisions, duplicate qrels/rankings and nonfinite parameters/gates are rejected. High-grade nDCG and very large finite k1 avoid intermediate overflow; the inspector deliberately uses narrower input bounds. Original overlapping tail windows and original macro skip policy remain.

Lexical BM25 is prior work, not embedding-quality evidence. Positive IDF, linear repeated-query weighting, tokenization, tie breaks, gain/discount, requested-k denominator, unjudged-as-zero and query exclusions are stated in [EVIDENCE.md](EVIDENCE.md). No Lucene/trec_eval parity, new-algorithm or quality-improvement claim. Browser parity uses a bounded ASCII profile, not unrestricted Unicode. MIT license.
