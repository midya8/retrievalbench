# Exact weighted reciprocal rank fusion

This is standard weighted reciprocal rank fusion (RRF), not a new algorithm, trained retriever, embedding model or relevance guarantee. It adds an opt-in API and two CLI commands. Original `search`, `evaluate`, `audit`, BM25, corpus, metrics and original fixtures retain their behavior.

[Local inspector](https://midya8.github.io/retrievalbench/fusion.html) · [Source](https://github.com/midya8/retrievalbench)

## Two actual workflows

```bash
# Explicit ranked lists, with strict JSON loading and exact rational output.
python -m retrievalbench fuse --spec examples/fusion.json --json fusion.json
# Actual original exhaustive BM25 rankings for supplied query variants.
python -m retrievalbench fuse-search --corpus examples/data/corpus.jsonl \
  --query 'bread yeast' --query 'fermentation' --weight 2 --weight 1 \
  --depth 3 --constant 60 -k 3 --json fused-search.json
```

The API is `retrievalbench.fusion.fuse(spec)` and `fuse_queries(documents, queries, *, constant=60, k=10, depth=10, weights=None, k1=1.5, b=.75)`. Documents are `(id, text)` pairs. Query variants are explicit strings, not generated expansions, independent retrievers or learned features. A duplicate query is another vote. `fuse-search` includes source texts, original unrounded BM25 scores, complete rankings, empty-token document omissions, the replay spec and the exact fusion report. It uses whole documents, not hidden chunking or parent collapse. There is no qrels tuning or claimed relevance improvement.

## Exact input contract

`retrievalbench-rrf/v1` has exactly `schema`, `constant`, `k`, `lists`. Each list has exactly `id`, `weight`, `depth`, `ranking`.

- 1..8 uniquely named sources. 0..64 unique document ids per ranking; at most 64 unique ids across all raw rankings, including excluded tails.
- Source/document ids: 1..64 printable ASCII characters, case-sensitive, including spaces. Within-source duplicate document ids are rejected. A shared document in different sources is expected.
- Integer rank constant 0..1,000,000; integer weight 0..100 with at least one positive weight; integer source depth 0..64; final k 0..64.
- Integer JSON numbers only: `60.0`, `6e1`, booleans, NaN and infinity are not integer spellings. Duplicate object keys and unknown/missing fields are errors. Files/text up to 128 KiB. Browser additionally rejects JSON nesting above 16; valid specs need only four structural levels.
- API callers also supply actual integers. No implicit strings, coercion or default missing fields.
- `fuse_queries`: 1..64 corpus records, source ids as above, texts up to 65,536 characters each, 1..8 supplied query strings up to 4,096 characters. Original BM25 rejects duplicate ids, invalid k1/b and all-empty token corpora; empty-token documents are otherwise omitted and reported. Query text may use Unicode subject to the original ASCII BM25 token pattern, unlike bounded ASCII ids.

Input validation completes before output writes. Invalid commands leave an existing output untouched and exit 2. Valid output uses Python JSON serialization; filesystem/OS failure during writing is not claimed crash-atomic. The inspector clears all results and disables download as soon as an input changes or validation fails.

## Formula, candidates and ties

For a candidate d, sum `weight_s / (constant + rank_s(d))` over sources where d is in the first `depth_s` entries. Ranks are 1-based and never re-numbered when weight is zero. Missing or beyond-depth entries contribute exactly zero; there is no synthetic last rank. Source score magnitudes are ignored.

Candidates are the union of **all admitted prefixes, including zero-weight sources**. This deliberate policy can retain exact-zero-score candidates. A document outside every prefix is excluded and listed separately. All candidates are sorted by full exact rational score descending, then printable ASCII document id ascending. Final k is sliced after fusion. `k=0` selects none, depth 0 admits none, k larger than candidate count selects all. There is no implicit relation between source depth and final k, unlike some vendor APIs.

Every candidate has one contribution record per source, including raw rank or null, admission, reason, weight, denominator or null and reduced fraction. Every raw source entry also remains visible, with admission and candidate membership. Report fractions are reduced and use decimal string numerator/denominator pairs, avoiding JSON integer precision loss. Native Python uses `fractions.Fraction`; the independent browser accumulates source columns using BigInt, GCD and rational cross-products. Exact fraction comparisons decide order. There is no rounded-score tie decision.

An inside/outside pair exists only for `0 < k < candidate count`. Its exact nonnegative gap and tie flag are shown. The upper bound `sum(weight_s / (constant + 1))` is a loose mathematical bound, not a probability or calibrated confidence. It includes positive weights even for empty/zero-depth sources. Identical raw-ranking + depth pairs are reported regardless of weight; duplication is not removed. Duplicate sources increase influence and are not independent evidence.

## Evidence and reproduction

```bash
python -m unittest discover -s tests -v
bash examples/demo.sh
PYTHONPATH=. python tools/build_audits.py
PYTHONPATH=. python tools/build_fusion.py
npm test
bash tools/wheel_smoke.sh
```

80 Python tests retain all 68 existing tests unchanged. The 12 new tests include 432 exhaustive permutation/depth/constant controls, 200 seeded conservation/source-order/weight-scaling controls, 100 random actual-BM25 pipeline comparisons, strict parsing, original BM25 zero-score and empty-token behavior, both CLI paths and invalid-output preservation. No expected score is stored by hand in generated reports.

230 complete native/browser reports cover 14 explicit controls, 200 seeded bounded profiles and 16 actual original-BM25 pipelines (4 source depths × 2 constants × 2 weight profiles). There are 2,006 candidate rows, 9,674 candidate/source contribution cells and 6,416 raw source entries. Independent Node and actual Chrome compare every complete field exactly. Chrome also checks every rendered ranking, contribution and raw-source cell, exact score bound, stale/error states, trusted keyboard edits, two real downloads replayed by the native algorithm, and 90 constructed/actual-pipeline control-width layouts at 320, 390 and 1280 pixels. The browser does not recompute source BM25 logs: it exactly replays supplied rankings. An installed wheel verifies nine module origins, all 230 reports, all 16 actual pipelines, both commands and invalid-output preservation outside the source checkout.

The original 68 tests, existing eight modules except additive CLI, example corpus/queries/qrels, original demo, complete BM25 audit fixtures and original whole/chunked outputs remain unchanged. Feature/PR/main/deployment verification is release evidence, not an algorithmic contribution.

## Prior work and limits

1. Cormack, Clarke and Buettcher, SIGIR 2009, [original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf). This is the source of RRF and the commonly used pilot-selected constant 60. Their experimental improvements are not evidence about these controls.
2. [Google publication record](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/) confirms attribution and venue.
3. [Elastic RRF API](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) documents rank constant, source windows and final size. Local zero constant/depth/k, candidate and tie policies differ; no Elasticsearch parity claim.
4. [Elastic weighted RRF](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf) documents configurable weighted fusion. Weights express influence, not confidence.
5. [Azure hybrid ranking](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) documents multiple ranked lists and score dependence on list count. This implementation does not call Azure or implement vector search.
6. [Apache Doris RRF](https://doris.apache.org/docs/4.x/key-features/reciprocal-rank-fusion/) is another standard implementation, not a target for parity.
7. [Python Fraction](https://docs.python.org/3/library/fractions.html) documents native exact rational arithmetic.
8. [Independent explanation](https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/) supplies design context, not quality evidence.

Inspected November 11, 2026. No new method, semantic-quality gain, score calibration, causality, retriever independence, speed/memory improvement, vendor parity or security guarantee is claimed. Full pairwise witness space is bounded O(candidates × sources), rankings are fully sorted, and generated fixtures are development controls. All work is local, without paid APIs, model downloads or private data. Downloaded evidence can contain user-supplied ids/text in native pipeline reports; replay checks arithmetic consistency, not authenticity or provenance.
