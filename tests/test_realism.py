"""Realism study S1: individual rates, the learned-rates update and the references."""

import unittest

import numpy as np

from experiments.realism import LearnedRates, OracleRates, event_rate, individual_rates
from simulator import core
from simulator.core import Memory
from simulator.domains import get_domain
from simulator.policies import StalenessXImpact


class TestIndividualRates(unittest.TestCase):
    def test_cv0_is_the_current_setting(self):
        for name in ("commute", "meals_calibrated"):
            d = get_domain(name)
            for u in range(5):
                r = individual_rates(d, 1000, u, 0.0)
                self.assertEqual(r, {a.name: a.base_rate for a in d.attributes})
                self.assertEqual(core.generate_timeline(d, 1000, u, 20, own_rates=r),
                                 core.generate_timeline(d, 1000, u, 20))

    def test_gamma_mean_and_cv(self):
        d = get_domain("commute")
        for cv in (0.5, 1.0, 2.0):  # a slow fact: the cap at 1 never binds
            draws = np.array([individual_rates(d, 7, u, cv)["weekly_budget"] for u in range(4000)])
            self.assertAlmostEqual(draws.mean(), 0.03, delta=0.03 * 4 * cv / np.sqrt(4000) * 2)
            self.assertAlmostEqual(draws.std() / draws.mean(), cv, delta=0.2 * cv)
        # fast fact at CV 2: the cap lowers the mean (about -8.6%)
        fast = np.array([individual_rates(d, 7, u, 2.0)["office_days"] for u in range(4000)])
        self.assertLess(fast.mean(), 0.15)
        self.assertGreater(fast.mean(), 0.15 * 0.8)
        self.assertTrue(all(individual_rates(d, 7, u, 2.0)["preferred_language"] == 0
                            for u in range(50)))

    def test_rates_capped_at_one(self):
        d = get_domain("commute")
        self.assertTrue(all(v <= 1.0 for u in range(500)
                            for v in individual_rates(d, 3, u, 2.0).values()))


class TestPolicies(unittest.TestCase):
    def test_oracle_rates_at_cv0_equals_population(self):
        d = get_domain("meals_calibrated")
        tls = core.generate_timelines(d, 1000, 10, 20)
        rates = {(1000, u): individual_rates(d, 1000, u, 0.0) for u in range(10)}
        a = [r["asked"] for r in core.run(d, tls, [OracleRates(d, 0.05, rates)])]
        b = [r["asked"] for r in core.run(d, tls, [StalenessXImpact(d, 0.05, "coarse")])]
        self.assertEqual(a, b)
        self.assertGreater(event_rate(d, "shopping_day"), 0)

    def test_learned_update_rule(self):
        d = get_domain("commute")
        pol = LearnedRates(d, 0.0)
        pol.start_user(0, 0)
        p0 = pol.population["office_days"]
        mem = Memory.onboard({a.name: a.values[0] for a in d.attributes})
        task = d.task("commute_plan")
        pol.decide(mem, task, 4)  # snapshot at t=4; office_days confirmed at 0
        mem.update("office_days", "full_office", 4)  # re-asked at 4: changed (d = 4)
        mem.update("home_area", mem.values["home_area"], 4)  # re-asked at 4: unchanged
        pol.decide(mem, task, 5)
        a, b = pol._ab["office_days"]
        self.assertAlmostEqual(a, 5 * p0 + 1)
        self.assertAlmostEqual(b, 5 * (1 - p0) + 3)
        ph = pol.population["home_area"]
        self.assertAlmostEqual(pol._ab["home_area"][1], 5 * (1 - ph) + 4)
        self.assertAlmostEqual(pol.rates["office_days"], a / (a + b))

    def test_learned_with_huge_prior_equals_population(self):
        d = get_domain("commute")
        tls = core.generate_timelines(d, 1000, 10, 20)
        a = [r["asked"] for r in core.run(d, tls, [LearnedRates(d, 0.05, strength=1e9)])]
        b = [r["asked"] for r in core.run(d, tls, [StalenessXImpact(d, 0.05, "coarse")])]
        self.assertEqual(a, b)


class TestS2S3(unittest.TestCase):
    def test_zero_error_changes_nothing(self):
        d = get_domain("commute")
        tls = core.generate_timelines(d, 1000, 10, 20)
        base = core.run(d, tls, [StalenessXImpact(d, 0.05, "coarse")])
        with core.wrong_answers(0.0):
            again = core.run(d, tls, [StalenessXImpact(d, 0.05, "coarse")])
        import pandas as pd  # NaN target_cost: compare as frames (NaN == NaN there)
        pd.testing.assert_frame_equal(pd.DataFrame(base), pd.DataFrame(again))

    def test_wrong_answers_rate_and_pairing(self):
        d = get_domain("meals_calibrated")
        tls = core.generate_timelines(d, 1000, 40, 20)
        from simulator.policies import Always, AlwaysAll
        with core.wrong_answers(0.2):
            recs = core.run(d, tls, [AlwaysAll(d)], record_memory=True)
        wrong = sum(r["memory"][a] != r["true_state"][a] for r in recs for a in r["memory"])
        total = sum(len(r["memory"]) for r in recs)
        self.assertAlmostEqual(wrong / total, 0.2, delta=0.02)
        # the same (user, session, fact) re-ask is wrong under every policy
        with core.wrong_answers(0.2):
            a = core.run(d, tls[:5], [AlwaysAll(d)], record_memory=True)
            b = core.run(d, tls[:5], [Always()], record_memory=True)
        for ra, rb in zip(a, b):
            task = d.task(ra["task"])
            for x in task.depends_on:
                self.assertEqual(ra["memory"][x], rb["memory"][x])
        self.assertEqual(core._WRONG_ANSWER_RATE, 0.0)

    def test_s3_factor_one_is_the_current_domain(self):
        from experiments.realism import scaled_events_domain
        d0, d1 = get_domain("meals_calibrated"), scaled_events_domain(1.0)
        for u in range(5):
            self.assertEqual(core.generate_timeline(d1, 1000, u, 20),
                             core.generate_timeline(d0, 1000, u, 20))
        d4 = scaled_events_domain(4.0)
        totals = {a.name: a.base_rate + event_rate(d4, a.name) for a in d4.attributes}
        self.assertAlmostEqual(totals["shopping_day"], 0.18)
        self.assertAlmostEqual(totals["weekly_grocery_budget"], 0.08)  # cap binds


if __name__ == "__main__":
    unittest.main()
