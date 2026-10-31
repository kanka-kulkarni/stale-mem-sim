"""Baselines: oracle (upper bound), always_all (reference), info_gain (task-blind
uncertainty reduction)."""

from __future__ import annotations

import unittest

import pandas as pd

from simulator import core
from simulator.core import Memory
from simulator.domains import get_domain
from simulator.domains.commute import DOMAIN
from simulator.policies import (AlwaysAll, Always, InfoGain, Never, Oracle, Volatility,
                                build_families, make_policy)


def run(domain, policies, seed=0, users=40, sessions=20, mult=1.0) -> pd.DataFrame:
    tls = core.generate_timelines(domain, seed, users, sessions, mult)
    return pd.DataFrame(core.run(domain, tls, policies))


def sets(df):
    return [set(a.split(";")) - {""} for a in df.asked.fillna("")]


class TestOracle(unittest.TestCase):
    def test_asks_exactly_the_stale_answer_changing_facts(self):
        for name in ("commute", "meals_calibrated"):
            d = get_domain(name)
            df = run(d, [Oracle(d, 20)])
            self.assertTrue((df.n_questions == df.n_answer_changing).all(), name)
            self.assertTrue((df.n_answer_changing_asked == df.n_answer_changing).all(), name)
            self.assertTrue((df.n_asked_stale == df.n_questions).all(), name)

    def test_beats_never_and_errors_only_from_remaining_stale_facts(self):
        df = run(DOMAIN, [Never(), Oracle(DOMAIN, 20)])
        acc = df.groupby("policy").correct.mean()
        self.assertGreater(acc["oracle"], acc["never"])
        self.assertGreaterEqual(acc["oracle"], 0.95)
        # The single-fact test holds other facts at their stored values, so interacting
        # stale facts can stay wrong after the oracle's re-asks; every remaining error
        # must still involve a stale dependency.
        wrong = df[(df.policy == "oracle") & ~df.correct]
        self.assertTrue((wrong.n_stale_deps > 0).all())

    def test_uses_true_multiplier_and_checks_alignment(self):
        df = run(DOMAIN, [Oracle(DOMAIN, 20, rate_multiplier=2.0)], mult=2.0)
        self.assertTrue((df.n_questions == df.n_answer_changing).all())
        oracle = Oracle(DOMAIN, 20)
        oracle.start_user(0, 0)
        tl = core.generate_timeline(DOMAIN, 0, 0, 20)
        wrong = next(k for k in DOMAIN.tasks if k.name != tl.tasks[1])
        with self.assertRaises(RuntimeError):
            oracle.decide(Memory.onboard(tl.states[0]), wrong, 1)


class TestAlwaysAll(unittest.TestCase):
    def test_memory_equals_truth(self):
        for name in ("commute", "meals_calibrated"):
            d = get_domain(name)
            df = run(d, [AlwaysAll(d)])
            self.assertEqual(df.correct.mean(), 1.0)
            self.assertTrue((df.n_questions == len(d.attributes)).all())
            self.assertEqual(int(df.n_stale_deps.sum()), 0)


class TestInfoGain(unittest.TestCase):
    def test_equals_blind_volatility_between_class_rates_on_commute(self):
        # slow: 1 - 0.97^d > 0.1 first at d = 4; fast (0.15) always above 0.1
        a = run(DOMAIN, [InfoGain(DOMAIN, 0.1)])
        b = run(DOMAIN, [Volatility(DOMAIN, 4, blind=True)])
        self.assertEqual(sets(a), sets(b))

    def test_differs_above_fast_rate_and_ignores_task(self):
        a = run(DOMAIN, [InfoGain(DOMAIN, 0.2)])
        b = run(DOMAIN, [Volatility(DOMAIN, 5, blind=True)])
        self.assertNotEqual(sets(a), sets(b))
        outside = sum(bool(s - set(DOMAIN.task(k).depends_on)) for s, k in zip(sets(a), a.task))
        self.assertGreater(outside, 0)

    def test_monotone_in_threshold(self):
        costs = [run(DOMAIN, [InfoGain(DOMAIN, x)]).n_questions.mean()
                 for x in (0.01, 0.05, 0.1, 0.3, 0.6)]
        self.assertEqual(costs, sorted(costs, reverse=True))

    def test_family_only_on_request(self):
        self.assertNotIn("info_gain", build_families(DOMAIN))
        fam = build_families(DOMAIN, include_info_gain=True)["info_gain"]
        self.assertEqual((fam.kind, fam.make(0.1).family), ("continuous", "info_gain"))

    def test_rate_domain_used_for_assumed_rates(self):
        cal, base = get_domain("meals_calibrated"), get_domain("meals")
        p = InfoGain(cal, 0.1, rate_domain=base)
        self.assertAlmostEqual(p.rates["cuisine_likes"], 0.03)


class TestSpecs(unittest.TestCase):
    def test_parse(self):
        self.assertIsInstance(make_policy("always_all", DOMAIN), AlwaysAll)
        self.assertIsInstance(make_policy("info_gain(0.05)", DOMAIN), InfoGain)
        o = make_policy("oracle", DOMAIN, 1.0, n_sessions=20, true_multiplier=2.0)
        self.assertEqual((o.n_sessions, o.mult), (20, 2.0))
        with self.assertRaises(KeyError):
            make_policy("oracle", DOMAIN)
        self.assertIsInstance(make_policy("always"), Always)


if __name__ == "__main__":
    unittest.main()
