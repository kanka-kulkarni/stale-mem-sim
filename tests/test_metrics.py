"""Metric tests: headline metrics on hand-built records, and per-session counts."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from simulator import core, metrics
from simulator.domains.commute import DOMAIN
from simulator.policies import Always, Never


def records(policy, correct, n_q, asked_stale, changing, changing_asked,
            pre_correct=None, family=None):
    pre_correct = pre_correct or [1] * len(correct)
    return [
        {"domain": "d", "policy": policy, "family": family or metrics_family(policy),
         "role": "sweep", "target_cost": float("nan"), "session": i + 1,
         "correct": bool(c), "n_questions": q, "stale_error": not c, "n_asked_stale": s,
         "n_answer_changing": ch, "n_answer_changing_asked": ca, "pre_ask_correct": bool(pc)}
        for i, (c, q, s, ch, ca, pc) in enumerate(zip(correct, n_q, asked_stale, changing,
                                                      changing_asked, pre_correct))
    ]


def metrics_family(policy: str) -> str:
    return policy.split("(")[0]


def toy() -> pd.DataFrame:
    rows = (
        records("never", [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0],
                pre_correct=[1, 0, 0, 1])
        + records("always", [1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0])
        + records("ttl(1)", [1, 1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0])
        + records("ttl(2)", [1, 1, 1, 1], [2, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0])
    )
    return pd.DataFrame(rows)


class TestSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = metrics.summarize(toy()).set_index("policy")

    def test_gap_closed_and_cost(self):
        s = self.s
        self.assertEqual((s.loc["never", "gap_closed"], s.loc["always", "gap_closed"]), (0, 1))
        self.assertAlmostEqual(s.loc["ttl(1)", "gap_closed"], 0.5)
        self.assertAlmostEqual(s.loc["ttl(1)", "question_cost"], 0.25)
        self.assertAlmostEqual(s.loc["ttl(2)", "question_cost"], 0.5)
        self.assertEqual(s.loc["always", "question_cost"], 1.0)

    def test_precision_recall(self):
        s = self.s
        self.assertAlmostEqual(s.loc["always", "staleness_precision"], 2 / 8)
        self.assertAlmostEqual(s.loc["always", "decision_precision"], 2 / 8)
        self.assertEqual(s.loc["always", "reask_recall"], 1.0)
        self.assertAlmostEqual(s.loc["ttl(1)", "staleness_precision"], 1 / 2)
        self.assertAlmostEqual(s.loc["ttl(1)", "decision_precision"], 1 / 2)
        self.assertAlmostEqual(s.loc["ttl(1)", "reask_recall"], 1 / 2)
        self.assertTrue(math.isnan(s.loc["never", "staleness_precision"]))
        self.assertTrue(math.isnan(s.loc["never", "decision_precision"]))
        self.assertEqual(s.loc["never", "reask_recall"], 0.0)

    def test_decision_precision_below_staleness_precision_when_stale_is_harmless(self):
        rows = (records("never", [1, 1], [0, 0], [0, 0], [0, 0], [0, 0])
                + records("always", [1, 1], [2, 2], [1, 1], [0, 1], [0, 1])
                + records("x(1)", [1, 1], [2, 2], [2, 1], [0, 1], [0, 1]))
        s = metrics.summarize(pd.DataFrame(rows)).set_index("policy")
        self.assertAlmostEqual(s.loc["x(1)", "staleness_precision"], 3 / 4)
        self.assertAlmostEqual(s.loc["x(1)", "decision_precision"], 1 / 4)

    def test_edge_case_counts(self):
        rows = records("never", [0, 1, 1, 0], [0] * 4, [0] * 4, [0, 1, 0, 1], [0] * 4,
                       pre_correct=[0, 1, 1, 0])
        s = metrics.summarize(pd.DataFrame(rows)).iloc[0]
        # session 1: wrong with no answer-changing fact; session 2: right despite one
        self.assertEqual((s.joint_only_n, s.cancelling_n), (1, 1))
        self.assertEqual((s.joint_only_frac, s.cancelling_frac), (0.25, 0.25))

    def test_marginal_accuracy_per_question(self):
        self.assertTrue(math.isnan(self.s.loc["never", "marginal_acc_per_question"]))
        self.assertAlmostEqual(self.s.loc["ttl(1)", "marginal_acc_per_question"], 0.25 / 0.5)

    def test_missing_baseline_gives_nan(self):
        s = metrics.summarize(toy().query("policy != 'always'")).set_index("policy")
        self.assertTrue(s.gap_closed.isna().all() and s.question_cost.isna().all())

    def test_matched_cost_interpolates_within_family_only(self):
        m = metrics.at_matched_cost(self.s.reset_index(), [0.25, 0.375, 0.75])
        ttl = m[m.family == "ttl"].set_index("question_cost")
        self.assertAlmostEqual(ttl.loc[0.375, "accuracy"], 0.875)
        self.assertAlmostEqual(ttl.loc[0.375, "gap_closed"], 0.75)
        self.assertAlmostEqual(ttl.loc[0.25, "gap_closed"], 0.5)
        self.assertTrue(math.isnan(ttl.loc[0.75, "accuracy"]))  # no extrapolation
        self.assertTrue(m[m.family == "always"].accuracy.isna().all())

    def test_exact_and_coarse_curves_stay_separate(self):
        """Same base name, different impact version -> separate interpolation curves."""
        rows = (records("never", [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], pre_correct=[0, 0])
                + records("always", [1, 1], [2, 2], [1, 1], [1, 1], [1, 1])
                + records("b(1,exact)", [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                          family="b[exact]")
                + records("b(2,exact)", [1, 1], [2, 2], [1, 1], [1, 1], [1, 1],
                          family="b[exact]")
                + records("b(1,coarse)", [0, 0], [1, 1], [0, 0], [1, 1], [0, 0],
                          family="b[coarse]")
                + records("b(2,coarse)", [0, 1], [2, 2], [1, 1], [1, 1], [1, 1],
                          family="b[coarse]"))
        m = metrics.at_matched_cost(metrics.summarize(pd.DataFrame(rows)), [0.75])
        m = m.set_index("family")
        self.assertAlmostEqual(m.loc["b[exact]", "accuracy"], 1.0)
        self.assertAlmostEqual(m.loc["b[coarse]", "accuracy"], 0.25)

    def test_matched_role_excluded_from_interpolation(self):
        df = toy()
        df.loc[df.policy == "ttl(2)", "role"] = "matched"
        m = metrics.at_matched_cost(metrics.summarize(df), [0.375])
        self.assertTrue(math.isnan(m[m.family == "ttl"].accuracy.iloc[0]))


def summ_row(policy, family, role, target, cost, acc):
    return {"domain": "d", "policy": policy, "family": family, "role": role,
            "target_cost": target, "question_cost": cost, "accuracy": acc}


class TestMatchedComparisons(unittest.TestCase):
    def test_local_slope(self):
        pts = [(0.0, 0.5), (0.5, 0.7), (1.0, 1.0)]
        self.assertAlmostEqual(metrics.local_slope(pts, 0.25), 0.4)
        self.assertAlmostEqual(metrics.local_slope(pts, 0.75), 0.6)
        self.assertAlmostEqual(metrics.local_slope(pts, 1.4), 0.6)  # beyond end: last segment
        self.assertAlmostEqual(metrics.local_slope([(0.25, 0.6), (0.75, 0.8)], 0.5), 0.4)
        self.assertTrue(math.isnan(metrics.local_slope([(0.5, 0.7)], 0.5)))

    def table(self, a_cost, a_acc, b_cost, b_acc, a_curve=True):
        """Family A has matched points at 0.25/0.75 (slope 0.4 around 0.5)."""
        rows = [summ_row("A@0.5", "A", "matched", 0.5, a_cost, a_acc),
                summ_row("B@0.5", "B", "matched", 0.5, b_cost, b_acc),
                summ_row("B@0.25", "B", "matched", 0.25, 0.25, 0.5)]
        if a_curve:
            rows += [summ_row("A@0.25", "A", "matched", 0.25, 0.26, 0.70),
                     summ_row("A@0.75", "A", "matched", 0.75, 0.76, 0.90)]
        c = metrics.matched_comparisons(pd.DataFrame(rows))
        return c[c.target_cost == 0.5].iloc[0]

    def test_costlier_leader_with_real_lead_not_flagged(self):
        r = self.table(0.52, 0.85, 0.49, 0.80)  # lead 0.05 > 0.4 * 0.03 = 0.012
        self.assertEqual((r.leader, r.leader_costlier, bool(r.flagged)), ("A@0.5", True, False))
        self.assertAlmostEqual(r.leader_slope, 0.4)
        self.assertAlmostEqual(r.lead_from_cost, 0.012)

    def test_costlier_leader_within_cost_effect_flagged(self):
        r = self.table(0.52, 0.81, 0.49, 0.80)  # lead 0.01 < 0.012
        self.assertTrue(r.flagged)
        self.assertEqual(r.note, "lead within cost effect")

    def test_cheaper_leader_not_flagged(self):
        r = self.table(0.48, 0.85, 0.51, 0.80)
        self.assertFalse(r.leader_costlier or r.flagged)
        self.assertEqual(r.note, "leader not costlier")

    def test_no_slope_flagged(self):
        r = self.table(0.52, 0.85, 0.49, 0.80, a_curve=False)
        self.assertTrue(r.flagged)
        self.assertEqual(r.note, "no slope available")

    def test_same_family_and_sweep_rows_not_compared(self):
        rows = [summ_row("A@0.5", "A", "matched", 0.5, 0.5, 0.8),
                summ_row("A(3)", "A", "sweep", float("nan"), 0.5, 0.7)]
        self.assertEqual(len(metrics.matched_comparisons(pd.DataFrame(rows))), 0)


class TestSessionCounts(unittest.TestCase):
    def test_answer_changing_stale(self):
        task = DOMAIN.task("weekend_outing")
        mem = {a.name: a.values[0] for a in DOMAIN.attributes}
        mem.update(weekly_budget="low", has_car="yes")
        truth = {**mem, "weekly_budget": "medium", "has_car": "no"}
        # budget alone flips local_park -> nearby_hike; car alone keeps local_park
        self.assertEqual(core.answer_changing_stale(task, mem, truth), ["weekly_budget"])

    def test_counts_on_real_run(self):
        tls = core.generate_timelines(DOMAIN, 0, 50, 20)
        df = pd.DataFrame(core.run(DOMAIN, tls, [Never(), Always()]))
        always, never = df[df.policy == "always"], df[df.policy == "never"]
        self.assertEqual(
            (always.n_answer_changing_asked == always.n_answer_changing).all(), True)
        self.assertEqual(never.n_asked_stale.sum() + never.n_answer_changing_asked.sum(), 0)
        self.assertTrue((always.n_asked_stale <= always.n_questions).all())
        self.assertGreater(never.n_answer_changing.sum(), 0)


if __name__ == "__main__":
    unittest.main()
