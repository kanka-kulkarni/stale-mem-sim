"""Numerical checks of Proposition 1 on small random separable instances (brute force
over subsets), and of the assumption behind 1(b): each decision rule reads only its
dependency list."""

import itertools
import unittest

import numpy as np

from simulator.domains import commute, meals
from simulator.theory import (above_threshold, independent_bound, separable_gain, top_k,
                              union_bound)


def _instance(rng: np.random.Generator, n: int) -> dict[str, float]:
    """A separable instance: at most one fact is decisive per session, so asking Q
    fixes the answer iff it contains that fact."""
    w = rng.dirichlet(np.ones(n + 1))
    return {f"a{i}": float(w[i]) for i in range(n)}


def _gain_by_outcomes(r: dict[str, float], q: set[str]) -> float:
    """Expected corrections from asking q, by enumerating the decisive-fact outcomes
    (independent of `separable_gain`, which uses the closed form)."""
    return sum(prob for a, prob in r.items() if a in q)


def _subsets(items: list[str]):
    for size in range(len(items) + 1):
        yield from (set(c) for c in itertools.combinations(items, size))


class TestPropositionA(unittest.TestCase):
    def test_gain_is_additive_on_instances(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            r = _instance(rng, int(rng.integers(2, 7)))
            for q in _subsets(list(r)):
                self.assertAlmostEqual(_gain_by_outcomes(r, q), separable_gain(r, q))

    def test_top_k_is_optimal_for_every_budget(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            r = _instance(rng, int(rng.integers(2, 8)))
            for k in range(len(r) + 1):
                best = max(_gain_by_outcomes(r, q) for q in _subsets(list(r)) if len(q) <= k)
                self.assertAlmostEqual(_gain_by_outcomes(r, top_k(r, k)), best)

    def test_threshold_maximizes_gain_minus_price(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            r = _instance(rng, int(rng.integers(2, 8)))
            lam = float(rng.uniform(0, 0.5))
            best = max(_gain_by_outcomes(r, q) - lam * len(q) for q in _subsets(list(r)))
            q = above_threshold(r, lam)
            self.assertAlmostEqual(_gain_by_outcomes(r, q) - lam * len(q), best)


class TestPropositionC(unittest.TestCase):
    def test_top_k_loses_at_most_2k_delta(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            r = _instance(rng, int(rng.integers(2, 8)))
            delta = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.5]))
            r_hat = {a: float(np.clip(v + rng.uniform(-delta, delta), 0, 1)) for a, v in r.items()}
            for k in range(len(r) + 1):
                best = max(_gain_by_outcomes(r, q) for q in _subsets(list(r)) if len(q) <= k)
                got = _gain_by_outcomes(r, top_k(r_hat, k))
                self.assertGreaterEqual(got, best - 2 * k * delta - 1e-12)

    def test_threshold_loss_bounded_by_near_threshold_facts(self):
        rng = np.random.default_rng(4)
        for _ in range(300):
            r = _instance(rng, int(rng.integers(2, 8)))
            delta = float(rng.choice([0.01, 0.05, 0.1, 0.2]))
            lam = float(rng.uniform(0, 0.4))
            r_hat = {a: float(np.clip(v + rng.uniform(-delta, delta), 0, 1)) for a, v in r.items()}

            def net(q):
                return _gain_by_outcomes(r, q) - lam * len(q)

            best = max(net(q) for q in _subsets(list(r)))
            near = sum(abs(v - lam) <= delta for v in r.values())
            self.assertGreaterEqual(net(above_threshold(r_hat, lam)), best - delta * near - 1e-12)


class TestPropositionB(unittest.TestCase):
    def test_bounds_ordered(self):
        p = {"a": 0.1, "b": 0.3, "c": 0.0}
        self.assertAlmostEqual(union_bound(p, p), 0.4)
        self.assertAlmostEqual(independent_bound(p, p), 1 - 0.9 * 0.7)
        self.assertLessEqual(independent_bound(p, p), union_bound(p, p))

    def test_rules_read_only_their_dependencies(self):
        """1(b) needs: if memory equals the truth on D_k, the answer is correct."""
        rng = np.random.default_rng(5)
        for dom in (commute.DOMAIN, meals.DOMAIN):
            for _ in range(300):
                state = {a.name: a.values[int(rng.integers(len(a.values)))] for a in dom.attributes}
                for task in dom.tasks:
                    base = task.answer(state)
                    for a in dom.attributes:
                        if a.name in task.depends_on:
                            continue
                        for v in a.values:
                            self.assertEqual(task.answer({**state, a.name: v}), base)


if __name__ == "__main__":
    unittest.main()
