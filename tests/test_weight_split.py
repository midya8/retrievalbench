"""Exact weight-splitting controls for standard reciprocal rank fusion."""
import copy
from fractions import Fraction
import unittest

from retrievalbench.fusion import SCHEMA, fuse


def ratio(value):
    return Fraction(int(value["n"]), int(value["d"]))


def spec(weight, constant, k):
    return {"schema": SCHEMA, "constant": constant, "k": k, "lists": [
        {"id": "original", "weight": weight, "depth": 2, "ranking": ["c", "a", "b"]},
        {"id": "control", "weight": 2, "depth": 3, "ranking": ["b", "a", "c"]},
        {"id": "zero", "weight": 0, "depth": 1, "ranking": ["z"]},
    ]}


class WeightSplitTests(unittest.TestCase):
    def test_splitting_weight_preserves_full_ranking_and_cutoff(self):
        count = 0
        for weight in (1, 2, 7, 100):
            for left in sorted({0, weight // 2, weight}):
                for constant in (0, 60, 1000000):
                    for k in (0, 1, 2, 4, 64):
                        original = spec(weight, constant, k)
                        split = copy.deepcopy(original)
                        source = split["lists"].pop(0)
                        split["lists"][:0] = [dict(source, id="left", weight=left),
                                               dict(source, id="right", weight=weight-left)]
                        before, after = fuse(original), fuse(split)
                        with self.subTest(weight=weight, left=left, constant=constant, k=k):
                            self.assertEqual(before["selected"], after["selected"])
                            self.assertEqual(before["cutoff"], after["cutoff"])
                            self.assertEqual(before["excluded"], after["excluded"])
                            self.assertEqual(before["upper_bound"], after["upper_bound"])
                            self.assertEqual(after["identical_rankings"], [["left", "right"]])
                            for old, new in zip(before["rows"], after["rows"]):
                                for key in ("doc", "score", "rank", "selected"):
                                    self.assertEqual(old[key], new[key])
                                self.assertEqual(ratio(old["contributions"][0]["contribution"]),
                                                 sum((ratio(row["contribution"])
                                                      for row in new["contributions"][:2]), Fraction()))
                                for row in new["contributions"][:2]:
                                    for key in ("rank", "admitted", "reason", "denominator"):
                                        self.assertEqual(row[key], old["contributions"][0][key])
                                self.assertEqual(old["contributions"][1:], new["contributions"][2:])
                        count += 1
        self.assertEqual(count, 165)

    def test_duplicating_without_splitting_really_changes_the_vote(self):
        original = {"schema": SCHEMA, "constant": 0, "k": 1, "lists": [
            {"id": "x", "weight": 1, "depth": 2, "ranking": ["a", "b"]},
            {"id": "y", "weight": 1, "depth": 2, "ranking": ["b", "a"]},
        ]}
        duplicated = copy.deepcopy(original)
        duplicated["lists"].append(dict(original["lists"][1], id="again"))
        before, after = fuse(original), fuse(duplicated)
        self.assertTrue(before["cutoff"]["tie"])
        self.assertEqual(before["selected"], ["a"])
        self.assertEqual(after["selected"], ["b"])
        self.assertFalse(after["cutoff"]["tie"])
        self.assertEqual(after["identical_rankings"], [["y", "again"]])

    def test_zero_weight_prefix_still_adds_a_candidate(self):
        original = spec(1, 60, 64)
        after = fuse(original)
        without_zero = copy.deepcopy(original)
        without_zero["lists"].pop()
        before = fuse(without_zero)
        self.assertEqual(after["selected"], before["selected"] + ["z"])
        self.assertEqual(after["upper_bound"], before["upper_bound"])
        row = after["rows"][-1]
        self.assertEqual(row["score"], {"n": "0", "d": "1"})
        self.assertTrue(row["contributions"][-1]["admitted"])
        self.assertEqual(row["contributions"][-1]["reason"], "included")
