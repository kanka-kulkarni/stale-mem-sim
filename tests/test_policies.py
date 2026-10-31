"""Policy tests for `never` and `always`, including the accuracy sanity checks."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from simulator import core
from simulator.core import Memory
from simulator.domains.commute import DOMAIN
from simulator.policies import (TTL, Always, Budget, Never, StalenessXImpact, Volatility,
                                make_policy)


def run_df(policies, seed=0, users=50, sessions=20, mult=1.0) -> pd.DataFrame:
    tls = core.generate_timelines(DOMAIN, seed, users, sessions, mult)
    return pd.DataFrame(core.run(DOMAIN, tls, policies))


class TestDecide(unittest.TestCase):
    def test_never_and_always_outputs(self):
        mem = Memory.onboard(core.generate_timeline(DOMAIN, 0, 0, 1).states[0])
        task = DOMAIN.task("commute_plan")
        self.assertEqual(Never().decide(mem, task, 3), [])
        self.assertEqual(Always().decide(mem, task, 3), list(task.depends_on))


class TestSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = run_df([Never(), Always()])

    def test_always_is_100_percent(self):
        always = self.df[self.df.policy == "always"]
        self.assertEqual(always.correct.mean(), 1.0)
        self.assertEqual(always.stale_error.sum(), 0)

    def test_never_asks_nothing_and_declines(self):
        never = self.df[self.df.policy == "never"]
        self.assertEqual(never.n_questions.sum(), 0)
        acc = never.groupby("session").correct.mean()
        slope = np.polyfit(acc.index.to_numpy(), acc.to_numpy(), 1)[0]
        self.assertLess(slope, 0)
        self.assertLess(acc.iloc[-5:].mean(), acc.iloc[:5].mean())

    def test_never_decline_holds_in_expectation(self):
        """Larger sample: accuracy at the last session is well below the first."""
        acc = run_df([Never()], seed=7, users=1000).groupby("session").correct.mean()
        self.assertGreater(acc.loc[1] - acc.loc[20], 0.1)

    def test_every_error_is_a_stale_error(self):
        """Rule-based answerer: a wrong answer implies an outdated dependency."""
        wrong = self.df[~self.df.correct]
        self.assertTrue(wrong.stale_error.all())
        self.assertTrue((wrong.n_stale_deps > 0).all())


def asked(policy, seed=0, users=40, sessions=20, mult=1.0) -> pd.DataFrame:
    tls = core.generate_timelines(DOMAIN, seed, users, sessions, mult)
    return pd.DataFrame(core.run(DOMAIN, tls, [policy]))


def sets(df: pd.DataFrame) -> list[set[str]]:
    return [set(a.split(";")) - {""} for a in df.asked]


STATIC = {a.name for a in DOMAIN.attributes if a.volatility == "static"}
FAST = {a.name for a in DOMAIN.attributes if a.volatility == "fast"}


class TestM2Policies(unittest.TestCase):
    def test_parse_and_family(self):
        cases = {"ttl(3)": ("ttl", TTL), "volatility(2)": ("volatility", Volatility),
                 "sxi(0.05,exact)": ("sxi[exact]", StalenessXImpact),
                 "budget(2,coarse)": ("budget[coarse]", Budget)}
        for spec, (fam, cls) in cases.items():
            p = make_policy(spec, DOMAIN)
            self.assertIsInstance(p, cls)
            self.assertEqual((p.name, p.family, p.stateful), (spec, fam, False))
        with self.assertRaises(KeyError):
            make_policy("ttl(3,exact)", DOMAIN)

    def test_ttl_extremes(self):
        self.assertEqual(sets(asked(TTL(DOMAIN, 0))), sets(asked(Always())))
        self.assertEqual(asked(TTL(DOMAIN, 20)).n_questions.sum(), 0)

    def test_ttl_asks_only_old_relevant_attributes(self):
        mem = Memory.onboard(core.generate_timeline(DOMAIN, 0, 0, 1).states[0])
        mem.update("office_days", "remote", 8)
        task = DOMAIN.task("commute_plan")
        out = TTL(DOMAIN, 3).decide(mem, task, 10)  # office_days age 2, others age 10
        self.assertEqual(set(out), set(task.depends_on) - {"office_days"})

    def test_volatility(self):
        for df in (asked(Volatility(DOMAIN, 1)), asked(Volatility(DOMAIN, 21))):
            for s, deps in zip(sets(df), df.task.map(lambda k: set(DOMAIN.task(k).depends_on))):
                self.assertFalse(s & STATIC)
                self.assertLessEqual(s, deps)
                self.assertGreaterEqual(s, deps & FAST)
        only_fast = asked(Volatility(DOMAIN, 21))
        self.assertTrue(all(s <= FAST for s in sets(only_fast)))
        all_nonstatic = asked(Volatility(DOMAIN, 1))
        for s, k in zip(sets(all_nonstatic), all_nonstatic.task):
            self.assertEqual(s, set(DOMAIN.task(k).depends_on) - STATIC)

    def test_sxi_extremes_and_impact_versions(self):
        self.assertEqual(asked(StalenessXImpact(DOMAIN, 1.0, "exact")).n_questions.sum(), 0)
        coarse0 = asked(StalenessXImpact(DOMAIN, 0.0, "coarse"))
        for s, k in zip(sets(coarse0), coarse0.task):
            self.assertEqual(s, set(DOMAIN.task(k).depends_on) - STATIC)

    def test_exact_policies_only_ask_attributes_with_positive_exact_impact(self):
        tl = core.generate_timeline(DOMAIN, 3, 0, 20)
        for policy in (StalenessXImpact(DOMAIN, 0.0, "exact"), Budget(DOMAIN, 4, "exact")):
            mem = Memory.onboard(tl.states[0])
            for t in range(1, 21):
                task = DOMAIN.task(tl.tasks[t])
                for a in policy.decide(mem, task, t):
                    self.assertGreater(core.impact_exact(DOMAIN, task, a, mem.values), 0)
                for a in task.depends_on:  # keep memory fresh-ish to vary states
                    if t % 3 == 0:
                        mem.update(a, tl.states[t][a], t)

    def test_sxi_threshold_monotone(self):
        tl = core.generate_timeline(DOMAIN, 5, 0, 1)
        mem = Memory.onboard(tl.states[0])
        for task in DOMAIN.tasks:
            for t in (1, 5, 15):
                prev = None
                for x in (0.0, 0.01, 0.05, 0.2, 0.6):
                    cur = set(StalenessXImpact(DOMAIN, x, "coarse").decide(mem, task, t))
                    if prev is not None:
                        self.assertLessEqual(cur, prev)
                    prev = cur

    def test_budget_top_k(self):
        mem = Memory.onboard(core.generate_timeline(DOMAIN, 0, 0, 1).states[0])
        mem.update("home_area", mem.values["home_area"], 9)  # age 1 at t=10
        task = DOMAIN.task("commute_plan")
        b = Budget(DOMAIN, 2, "coarse")
        # coarse score = P(changed); office_days (fast, age 10) > slow attrs at age 10 >
        # home_area (slow, age 1); ties among has_car/work_location by declaration order.
        self.assertEqual(b.decide(mem, task, 10), ["office_days", "work_location"])
        self.assertEqual(Budget(DOMAIN, 0, "coarse").decide(mem, task, 10), [])
        self.assertEqual(len(Budget(DOMAIN, 9, "coarse").decide(mem, task, 10)), 4)
        static_task = DOMAIN.task("commute_podcast")
        self.assertNotIn("preferred_language", Budget(DOMAIN, 3, "coarse").decide(
            mem, static_task, 10))  # zero score -> never asked

    def test_task_blind_variants(self):
        every = {a.name for a in DOMAIN.attributes}
        ttl0 = asked(TTL(DOMAIN, 0, blind=True))
        self.assertTrue(all(s == every for s in sets(ttl0)))  # incl. static, irrelevant
        vol1 = asked(Volatility(DOMAIN, 1, blind=True))
        self.assertTrue(all(s == every - STATIC for s in sets(vol1)))
        vol_fast = asked(Volatility(DOMAIN, 21, blind=True))
        self.assertTrue(all(s == FAST for s in sets(vol_fast)))
        outside = sum(bool(s - set(DOMAIN.task(k).depends_on))
                      for s, k in zip(sets(asked(TTL(DOMAIN, 4, blind=True))),
                                      asked(TTL(DOMAIN, 4, blind=True)).task))
        self.assertGreater(outside, 0)

    def test_blind_names_and_families(self):
        p = make_policy("ttl(3,blind)", DOMAIN)
        self.assertEqual((p.name, p.family, p.blind), ("ttl(3,blind)", "ttl[blind]", True))
        q = make_policy("volatility(2,blind)", DOMAIN)
        self.assertEqual((q.name, q.family), ("volatility(2,blind)", "volatility[blind]"))
        with self.assertRaises(KeyError):
            make_policy("sxi(0.1,blind)", DOMAIN)

    def test_llm_impact_equal_to_coarse_table_reproduces_coarse(self):
        table = {(k.name, a.name): core.impact_coarse(k, a.name)
                 for k in DOMAIN.tasks for a in DOMAIN.attributes}
        for x in (0.01, 0.1):
            llm = asked(StalenessXImpact(DOMAIN, x, "llm", impact_table=table))
            coarse = asked(StalenessXImpact(DOMAIN, x, "coarse"))
            self.assertEqual(sets(llm), sets(coarse))
        self.assertEqual(sets(asked(Budget(DOMAIN, 2, "llm", impact_table=table))),
                         sets(asked(Budget(DOMAIN, 2, "coarse"))))

    def test_llm_impact_can_ask_outside_dependencies(self):
        table = {(k.name, a.name): 1.0 for k in DOMAIN.tasks for a in DOMAIN.attributes}
        df = asked(StalenessXImpact(DOMAIN, 0.0, "llm", impact_table=table))
        self.assertTrue(all(s == {a.name for a in DOMAIN.attributes} - STATIC
                            for s in sets(df)))
        with self.assertRaises(ValueError):
            StalenessXImpact(DOMAIN, 0.1, "llm").decide(
                Memory.onboard(core.generate_timeline(DOMAIN, 0, 0, 1).states[0]),
                DOMAIN.tasks[0], 1)

    def test_assumed_multiplier_changes_only_rate_based_scores(self):
        mis = StalenessXImpact(DOMAIN, 0.1, "coarse", rate_multiplier=1.0)
        right = StalenessXImpact(DOMAIN, 0.1, "coarse", rate_multiplier=2.0)
        self.assertAlmostEqual(mis.rates["office_days"], 0.15)
        self.assertAlmostEqual(right.rates["office_days"], 0.30)

    def test_rates_follow_multiplier(self):
        p = TTL(DOMAIN, 1, rate_multiplier=2.0)
        self.assertAlmostEqual(p.rates["office_days"], 0.30)
        self.assertAlmostEqual(TTL(DOMAIN, 1).rates["office_days"], 0.15)


if __name__ == "__main__":
    unittest.main()
