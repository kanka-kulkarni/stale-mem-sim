"""Exact McNemar and Holm tests against hand-computed values."""

from __future__ import annotations

import unittest

import pandas as pd

from simulator import stats


class TestMcNemar(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(10, 0), 2 / 1024)
        self.assertAlmostEqual(stats.mcnemar_exact(0, 10), 2 / 1024)
        self.assertEqual(stats.mcnemar_exact(5, 5), 1.0)
        # n=12, k=2: 2 * (1 + 12 + 66) / 4096
        self.assertAlmostEqual(stats.mcnemar_exact(2, 10), 2 * 79 / 4096)

    def test_large_counts_are_finite(self):
        p = stats.mcnemar_exact(1500, 1400)
        self.assertTrue(0 < p < 1)

    def test_discordant_pairs_on_shared_keys(self):
        rows = []
        for s, (a, b) in enumerate([(1, 0), (1, 0), (0, 1), (1, 1), (0, 0)]):
            rows += [{"seed": 0, "user": 0, "session": s, "policy": "A", "correct": bool(a)},
                     {"seed": 0, "user": 0, "session": s, "policy": "B", "correct": bool(b)}]
        self.assertEqual(stats.discordant(pd.DataFrame(rows), "A", "B"), (2, 1, 5))


class TestHolm(unittest.TestCase):
    def test_known_values(self):
        adj = stats.holm([0.01, 0.04, 0.03])
        for got, want in zip(adj, [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(got, want)

    def test_capped_and_monotone(self):
        adj = stats.holm([0.5, 0.2, 0.9])
        self.assertTrue(all(a <= 1 for a in adj))
        self.assertEqual(stats.holm([]), [])


if __name__ == "__main__":
    unittest.main()
