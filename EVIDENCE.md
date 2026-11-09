# Retrieval arithmetic contract

RetrievalBench is an exhaustive floating-point lexical baseline, not mathematically exact real arithmetic or a semantic relevance oracle. BM25, relevance metrics and chunking are prior work. The inspector makes the implementation's definitions inspectable rather than proposing a new ranking algorithm.

## Original evidence stays visible

All 42 original tests, the 12-document corpus, six queries, graded judgments and offline demo are byte-preserved. Their hashes are retained in `docs/original-sha256.json`; the original README is copied verbatim to `docs/original-readme.md`. Original whole-document and chunked evaluation JSON are retained as `docs/original-eval.json` and `docs/original-chunked.json`. The modified CLI reproduces both byte-for-byte. Illustrative perfect recall remains an example, not evidence of broad retrieval quality.

## Corrections, not hidden protocol changes

- Chunk parent identity is carried explicitly. The original CLI split the first `#c` substring, corrupting legal ids such as `a#c-old`. Chunk records retain half-open token windows and the original tail-window rule. A final overlapping window wholly covered by its predecessor is still generated. Generated-id collisions are rejected instead of silently merging units. Short unchunked documents preserve their full original text/id.
- Duplicate native document ids, duplicate qrel pairs and duplicate metric rankings are rejected. Repeated ranking ids previously let recall and nDCG exceed one; duplicate judgment pairs silently replaced earlier grades.
- Nonfinite/Boolean BM25 parameters and nonfinite/negative CLI gates are rejected. Gates above one remain supported, including the original impossible-gate test. Very large finite k1 uses an algebraically scaled saturation expression to avoid an intermediate infinity.
- High-grade nDCG scales all gains by a common power of two before summing when the maximum grade exceeds 100. Normal-sized grades retain the original arithmetic. This prevents `2.0 ** grade` overflow, not arbitrary-precision exactness; extreme relative grade differences underflow in binary64. The bounded inspector accepts grades only 0..8.
- Markdown per-query tables now have the correct separator-column count.

The default CLI still takes k retrieval units before parent collapsing. It is not silently changed to return k distinct parents. The alternative exists only as an explicit audit policy. Original example JSON remains unchanged.

## The full witness

`python -m retrievalbench audit --spec examples/audit.json --json audit.json`

Specs include documents (id/text), query, grades, k, k1, b, window, overlap, retrieval `unit` (`documents` or `chunks`) and selected `budget` (`chunks` or `parents`). Defaults are k=3, k1=1.5, b=.75, window=12, overlap=2, whole documents and k-units-before-collapse. The selected budget controls display; every report includes both budget computations.

For every retrieval unit the report keeps its full text, parent id, half-open token window, every token occurrence with offsets in that unit's text, full score, rank and every distinct query term's tf/qtf/df/IDF, length normalization, denominator, saturation and score contribution. Chunk text is the original lowercased token window joined with spaces, so occurrence character offsets refer to this canonical unit text, not the original document's character offsets. Original parent text remains in the spec. Empty-token units are listed as omitted and excluded from N, df and average length.

The ranking is complete, including zeros, sorted by descending score then ascending unit id. Duplicate query occurrences are summed linearly in original order. The term table groups those occurrences for explanation; `weighted` is qtf times the single-occurrence contribution, which can differ from repeated binary64 additions by rounding. Numeric comparisons use tolerance, not a cross-runtime bit-identity claim.

The formula is:

```
IDF(t) = ln(1 + (N - df(t) + .5)/(df(t) + .5))
norm(d) = 1 - b + b * length(d)/avgdl
score(q,d) = sum over query occurrences of IDF(t) * tf(t,d) * (k1+1)/(tf(t,d)+k1*norm(d))
```

This is the positive-IDF variant, not raw RSJ log odds with negative common-term values clipped. The old max(0) remains algebraically redundant. No stemming or stopword removal. Tokenization lowercases, then extracts `[a-z0-9]+`. Lucene's storage norms, defaults, analysis and optional query-side saturation differ; no Lucene score parity is claimed.

## Two budgets and metric conventions

Both budget controls use the identical complete ranked list and index:

- `chunks`: consider only the first k units, then keep each parent once.
- `parents`: scan until k distinct parents have been kept, or exhaust all units.

Every raw rank has a complete admission/duplicate/kept/parent-rank witness. A parent-rank value after the budget closes means that parent was kept earlier, not that the later unit was admitted. Duplicates and units beyond the budget remain visible. Zero scores remain eligible, so unrelated units can fill the list. This is a defined baseline behavior, not evidence of relevance.

Indexing whole documents versus chunks changes N, df and average length as well as text granularity. It is not an isolated causal effect of shorter text. Comparing budgets holds these statistics and scores fixed. Neither comparison measures search quality without credible judgments.

Per rank, the report shows judged status, grade, binary hit, exponential gain `2^grade - 1`, discount `1/log2(rank+1)`, DCG contribution, cumulative DCG and hits. The ideal denominator includes all judged ids, even absent corpus ids, sorted by grade descending then id. Precision divides by requested k even when fewer parents return. Recall divides by all positive-grade judgments. Unjudged documents have grade zero for evaluation, not proven irrelevance. nDCG is zero when ideal DCG is zero.

The audit's RR is cut at k. The library's `reciprocal_rank` and `evaluate_run` examine all supplied ranks for RR; the CLI only supplies a cutoff ranking. The original macro average excludes queries without positive judgments. The inspector displays zero-valued per-control metrics for these queries and explicitly says they are excluded, not zero contributions. Missing rankings for positive judged queries remain an error. No trec_eval or ir_measures output-equivalence claim is made.

## Bounded independent browser

Python and independent scalar JavaScript accept 1..16 unique documents, 2,048 ASCII characters each, 1,024 total corpus tokens, a 512-character/64-token query, 128 retrieval units, 32 judgments with grades 0..8, k in 1..32, k1 in [.01,100], b in [0,1], window in 1..256 and overlap below window. ASCII excludes control characters except TAB/LF/CR. Identifiers are nonempty and at most 64 characters. Numeric integer fields accept finite integral JSON values. Unknown fields are rejected. All corpus/ranking/score/metric rows are retained, not top-matches-only excerpts.

The Python file loader also limits input to 128 KiB and rejects duplicate JSON keys/nonfinite JSON. Browser JSON editors use standard parsing, which keeps the final duplicate-key value; exported specs reveal the actual object evaluated. The original CLI is not restricted to this profile. ASCII offsets are code-point and UTF-16 offsets simultaneously; no unrestricted Unicode parity is claimed.

The browser recalculates after edits and hides stale results/disables export until recomputed. Export contains the actual current full report, not the retained fixture. Source text is rendered as text nodes, not HTML. No paid API, downloaded model or external inference.

## Reproduction

- 68 Python tests, including all 42 originals unchanged. New tests include 200 seeded BM25 comparisons against 70-digit Decimal arithmetic, all 65 partial permutations of four ids at five cutoffs against direct metric formulas, exact token spans, both parent budgets, literal-id/collision regressions, overflow controls and invalid input.
- 35 retained full reports: 23 controls plus the six original queries under whole-document/chunk indexing. Controls include repeated queries, crowding, both budgets, missing judgments, no positive judgments, empty/unknown queries, zero-score ties, overlap, b/k1 extremes, exponential grades and HTML as data.
- Independent JavaScript compares all report leaves: 27,694 numeric and 12,052 other scalar values with relative/absolute tolerance 1e-11.
- Three Node/actual-Chrome tests inspect all reports, 268 ranked units, 1,384 contribution rows, 89 selected-policy metric rows, every full unit/parent text, both policy tables, ideal gains, input invalidation, JSON export and 320/390/1280 page containment.
- Python 3.10-3.13 runs test and regenerate artifacts. An isolated installed wheel outside the checkout verifies eight module origins, all 35 reports, audit CLI and original gated evaluation CLI.

Same-environment byte regeneration does not promise all-platform determinism. These are synthetic arithmetic/policy controls, not a benchmark or confidence interval. Judgment incompleteness and lexical mismatch remain.

## Prior work

- [Apache Lucene BM25 implementation](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java) documents the IDF expression and cites Robertson, Walker, Jones, Hancock-Beaulieu and Gatford, Okapi at TREC-3. No Lucene code copied.
- [Lucene 9.11 BM25 API](https://lucene.apache.org/core/9_11_0/core/org/apache/lucene/search/similarities/BM25Similarity.html) specifies that implementation's parameters/norms. This project does not reproduce them exactly.
- [Elastic Practical BM25, Part 2](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) explains saturation and length normalization.
- [NIST trec_eval](https://github.com/usnistgov/trec_eval/blob/main/README) is the established TREC evaluation tool; its completeness/configuration options are not silently assumed here.
- [ir_measures conventions](https://ir-measur.es/en/latest/measures.html) distinguish recall from success, optional RR cutoffs, judgment policies and gain variants.
- [Trectools duplicate-qrels discussion](https://github.com/joaopalotti/trectools/issues/52) illustrates why duplicate handling must be explicit. It is a community bug report, not validation of this implementation.

Implementation, original synthetic controls and inspector by Midya Rahmani. No external evaluator code or benchmark data copied.
