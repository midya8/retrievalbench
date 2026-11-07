import json
import tempfile
import unittest
from pathlib import Path

from retrievalbench.bm25 import BM25Index, tokenize
from retrievalbench.corpus import chunk_documents, load_documents, load_qrels, load_queries
from retrievalbench.metrics import evaluate_run, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from retrievalbench.report import render_markdown, render_terminal

DOCS = [
    ("d1", "the cat sat on the warm mat"),
    ("d2", "dogs chase cats through the garden"),
    ("d3", "quantum entanglement enables state teleportation"),
    ("d4", "sourdough bread needs a wild yeast starter"),
]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Hello, World! BM25's k1=1.5"), ["hello", "world", "bm25", "s", "k1", "1", "5"])

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("!!!"), [])


class BM25IndexTests(unittest.TestCase):
    def test_relevant_document_ranks_first(self):
        index = BM25Index(DOCS)
        ranked = index.rank("quantum teleportation", top_k=4)
        self.assertEqual(ranked[0][0], "d3")

    def test_ranking_is_deterministic(self):
        index = BM25Index(DOCS)
        first = index.rank("cat", top_k=4)
        second = index.rank("cat", top_k=4)
        self.assertEqual(first, second)

    def test_zero_scores_are_included_with_stable_order(self):
        index = BM25Index(DOCS)
        ranked = index.rank("cat", top_k=4)
        zero_ids = [doc_id for doc_id, score in ranked if score == 0.0]
        self.assertEqual(zero_ids, sorted(zero_ids))

    def test_top_k_limits_results(self):
        index = BM25Index(DOCS)
        self.assertEqual(len(index.rank("the", top_k=2)), 2)

    def test_top_k_must_be_positive(self):
        index = BM25Index(DOCS)
        with self.assertRaises(ValueError):
            index.rank("cat", top_k=0)

    def test_empty_documents_are_skipped(self):
        index = BM25Index(DOCS + [("empty", "   ")])
        self.assertEqual(index.size, 4)

    def test_empty_corpus_raises(self):
        with self.assertRaises(ValueError):
            BM25Index([("x", "")])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            BM25Index(DOCS, k1=0)
        with self.assertRaises(ValueError):
            BM25Index(DOCS, b=1.5)

    def test_unknown_query_terms_score_zero(self):
        index = BM25Index(DOCS)
        ranked = index.rank("zebra xylophone", top_k=4)
        self.assertTrue(all(score == 0.0 for _, score in ranked))

    def test_longer_document_not_favored(self):
        index = BM25Index([("short", "apple pie"), ("long", "apple " + "filler " * 200)])
        ranked = index.rank("apple pie", top_k=2)
        self.assertEqual(ranked[0][0], "short")


class CorpusLoadTests(unittest.TestCase):
    def test_load_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.jsonl"
            write_jsonl(path, [{"id": "a", "text": "alpha"}, {"doc_id": "b", "text": "beta"}])
            self.assertEqual(load_documents(path), [("a", "alpha"), ("b", "beta")])

    def test_load_documents_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.jsonl"
            write_jsonl(path, [{"id": "a", "text": "alpha"}, {"id": "a", "text": "again"}])
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_load_documents_rejects_missing_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.jsonl"
            write_jsonl(path, [{"id": "a"}])
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_load_documents_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_load_documents_rejects_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_load_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"
            write_jsonl(path, [{"id": "q1", "text": "alpha"}])
            self.assertEqual(load_queries(path), [("q1", "alpha")])

    def test_load_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.jsonl"
            write_jsonl(path, [
                {"query_id": "q1", "doc_id": "a", "relevance": 2},
                {"query_id": "q1", "doc_id": "b"},
            ])
            self.assertEqual(load_qrels(path), {"q1": {"a": 2, "b": 1}})

    def test_load_qrels_rejects_negative_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.jsonl"
            write_jsonl(path, [{"query_id": "q1", "doc_id": "a", "relevance": -1}])
            with self.assertRaises(ValueError):
                load_qrels(path)


class ChunkTests(unittest.TestCase):
    def test_short_documents_keep_their_id(self):
        chunks = chunk_documents([("d1", "one two three")], max_tokens=10, overlap=2)
        self.assertEqual(chunks, [("d1", "one two three")])

    def test_long_documents_split_with_overlap(self):
        text = " ".join(f"w{i}" for i in range(25))
        chunks = chunk_documents([("d1", text)], max_tokens=10, overlap=2)
        self.assertTrue(all(chunk_id.startswith("d1#c") for chunk_id, _ in chunks))
        self.assertGreaterEqual(len(chunks), 3)
        first_window = chunks[0][1].split()
        second_window = chunks[1][1].split()
        self.assertEqual(first_window[-2:], second_window[:2])

    def test_overlap_must_be_smaller_than_window(self):
        with self.assertRaises(ValueError):
            chunk_documents([("d1", "a b c")], max_tokens=5, overlap=5)

    def test_max_tokens_must_be_positive(self):
        with self.assertRaises(ValueError):
            chunk_documents([("d1", "a b c")], max_tokens=0)


class MetricTests(unittest.TestCase):
    def test_precision_at_k(self):
        self.assertAlmostEqual(precision_at_k(["a", "b", "c"], {"a", "c"}, 3), 2 / 3)
        self.assertAlmostEqual(precision_at_k(["a", "b", "c"], {"a", "c"}, 1), 1.0)

    def test_recall_at_k(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 3), 2 / 3)
        self.assertEqual(recall_at_k(["a"], set(), 1), 0.0)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "a"], {"a"}), 0.5)
        self.assertEqual(reciprocal_rank(["x", "y"], {"a"}), 0.0)

    def test_ndcg_perfect_and_zero(self):
        grades = {"a": 2, "b": 1}
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], grades, 2), 1.0)
        self.assertEqual(ndcg_at_k(["x", "y"], grades, 2), 0.0)

    def test_ndcg_penalizes_inversion(self):
        grades = {"a": 2, "b": 1}
        self.assertLess(ndcg_at_k(["b", "a"], grades, 2), 1.0)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            precision_at_k(["a"], {"a"}, 0)
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], {"a": 1}, 0)

    def test_evaluate_run_aggregates(self):
        qrels = {"q1": {"a": 2}, "q2": {"b": 1, "c": 1}}
        run = {"q1": ["a", "x"], "q2": ["x", "b", "c"]}
        result = evaluate_run(run, qrels, k=2)
        self.assertEqual(result["query_count"], 2)
        self.assertAlmostEqual(result["averages"]["reciprocal_rank"], 0.75)
        self.assertAlmostEqual(result["averages"]["recall_at_2"], (1.0 + 0.5) / 2)

    def test_evaluate_run_requires_rankings_for_judged_queries(self):
        with self.assertRaises(ValueError):
            evaluate_run({}, {"q1": {"a": 1}}, k=5)

    def test_evaluate_run_skips_unjudged_and_zero_grade_queries(self):
        result = evaluate_run({"q1": ["a"], "q2": ["b"]}, {"q1": {"a": 1}, "q2": {"b": 0}}, k=1)
        self.assertEqual(result["query_count"], 1)

    def test_evaluate_run_needs_at_least_one_relevant_query(self):
        with self.assertRaises(ValueError):
            evaluate_run({"q1": ["a"]}, {"q1": {"a": 0}}, k=1)


class ReportTests(unittest.TestCase):
    def _result(self):
        qrels = {"q1": {"a": 2, "b": 1}, "q2": {"c": 1}}
        run = {"q1": ["a", "b", "x"], "q2": ["x", "c", "y"]}
        return evaluate_run(run, qrels, k=3)

    def test_markdown_report_structure(self):
        text = render_markdown(self._result(), corpus_size=10)
        self.assertTrue(text.startswith("# RetrievalBench evaluation\n"))
        self.assertIn("| Corpus documents | 10 |", text)
        self.assertIn("| Judged queries | 2 |", text)
        self.assertIn("| q1 |", text)
        self.assertIn("MRR", text)

    def test_markdown_escapes_pipe_characters(self):
        result = self._result()
        result["queries"][0]["query_id"] = "q|1"
        text = render_markdown(result, corpus_size=10)
        self.assertIn("q\\|1", text)

    def test_terminal_report_names_weakest_query(self):
        text = render_terminal(self._result(), corpus_size=10)
        self.assertIn("weakest query q2", text)
        self.assertIn("k=3", text)


class CliTests(unittest.TestCase):
    def _write_example(self, tmp):
        corpus = Path(tmp) / "corpus.jsonl"
        queries = Path(tmp) / "queries.jsonl"
        qrels = Path(tmp) / "qrels.jsonl"
        write_jsonl(corpus, [{"id": doc_id, "text": text} for doc_id, text in DOCS])
        write_jsonl(queries, [{"id": "q1", "text": "quantum teleportation"}])
        write_jsonl(qrels, [{"query_id": "q1", "doc_id": "d3", "relevance": 2}])
        return corpus, queries, qrels

    def test_evaluate_writes_reports(self):
        from retrievalbench.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries, qrels = self._write_example(tmp)
            out_json = str(Path(tmp) / "out.json")
            out_md = str(Path(tmp) / "out.md")
            rc = main(["evaluate", "--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels),
                       "-k", "3", "--json", out_json, "--markdown", out_md])
            self.assertEqual(rc, 0)
            payload = json.loads(Path(out_json).read_text(encoding="utf-8"))
            self.assertEqual(payload["corpus_size"], 4)
            self.assertAlmostEqual(payload["averages"]["reciprocal_rank"], 1.0)
            self.assertIn("RetrievalBench", Path(out_md).read_text(encoding="utf-8"))

    def test_gates_fail_below_threshold(self):
        from retrievalbench.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries, qrels = self._write_example(tmp)
            rc = main(["evaluate", "--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels),
                       "-k", "3", "--min-precision", "1.0"])
            self.assertEqual(rc, 1)
            rc = main(["evaluate", "--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels),
                       "-k", "3", "--min-recall", "1.0", "--min-mrr", "1.0", "--min-ndcg", "1.0"])
            self.assertEqual(rc, 0)

    def test_bad_inputs_exit_2(self):
        from retrievalbench.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            corpus, queries, qrels = self._write_example(tmp)
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text("nope\n", encoding="utf-8")
            rc = main(["evaluate", "--corpus", str(bad), "--queries", str(queries), "--qrels", str(qrels)])
            self.assertEqual(rc, 2)

    def test_search_command(self):
        from retrievalbench.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            corpus, _, _ = self._write_example(tmp)
            rc = main(["search", "--corpus", str(corpus), "-k", "2", "yeast bread"])
            self.assertEqual(rc, 0)

    def test_chunked_evaluation_collapses_to_parent_documents(self):
        from retrievalbench.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            long_text = "filler " * 50 + "zebra stripe pattern " + "filler " * 50
            write_jsonl(corpus, [{"id": "d1", "text": long_text}, {"id": "d2", "text": "unrelated words"}])
            queries = Path(tmp) / "queries.jsonl"
            write_jsonl(queries, [{"id": "q1", "text": "zebra stripe"}])
            qrels = Path(tmp) / "qrels.jsonl"
            write_jsonl(qrels, [{"query_id": "q1", "doc_id": "d1", "relevance": 1}])
            out_json = str(Path(tmp) / "out.json")
            rc = main(["evaluate", "--corpus", str(corpus), "--queries", str(queries), "--qrels", str(qrels),
                       "-k", "3", "--chunk-tokens", "20", "--chunk-overlap", "5", "--json", out_json])
            self.assertEqual(rc, 0)
            payload = json.loads(Path(out_json).read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["averages"]["recall_at_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
