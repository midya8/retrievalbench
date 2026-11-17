# retrievalbench

BM25 search plus the usual retrieval metrics (precision, recall, MRR, nDCG), for checking a small document set against hand-written relevance judgments. No dependencies, runs offline. I mostly use it to sanity-check chunking settings before reaching for anything fancier.

```bash
python -m retrievalbench search --corpus examples/data/corpus.jsonl -k 3 'BM25 relevance'

python -m retrievalbench evaluate --corpus examples/data/corpus.jsonl \
  --queries examples/data/queries.jsonl --qrels examples/data/qrels.jsonl \
  -k 3 --min-recall 1
```

The toy example is 12 documents and 6 queries:

```
nDCG@3       1.0000
Precision@3  0.4444
Recall@3     1.0000
MRR          1.0000
```

Precision is low because most queries only have one relevant doc and k is 3, which is a decent reminder of how much the metric depends on the judgments.

Data is JSONL. Corpus and queries are `{"id", "text"}`, qrels are `{"query_id", "doc_id", "relevance"}`. Add `--chunk-tokens 30 --chunk-overlap 5` to index chunks instead of whole documents, and `--k1` / `--b` to tune BM25. The `--min-*` flags turn it into a CI check.

One thing that tripped me up: with chunking, taking the top k chunks and then collapsing to their parent documents can give you fewer than k documents. The tool keeps that behaviour as the default and the `audit` command shows both ways side by side.

## Rank fusion

`fuse` and `fuse-search` do reciprocal rank fusion over several ranked lists, or over BM25 results for several phrasings of the same query:

```bash
python -m retrievalbench fuse-search --corpus examples/data/corpus.jsonl \
  --query 'bread yeast' --query 'fermentation' --depth 3 -k 3 --json fused.json
```

More detail in FUSION.md. There are browser pages for both the BM25 scoring and the fusion at https://midya8.github.io/retrievalbench/.

## Tests

`python -m unittest discover -s tests` (83 tests), and `npm test` for the browser pages.
