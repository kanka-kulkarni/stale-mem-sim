"""Matched-cost tests: mixture streams, calibration, tuning/held-out separation.

Uses toy policy families defined here; the real policy families plug into the same code.
"""

from __future__ import annotations

import argparse
import unittest

import pandas as pd

from experiments.run import run_experiment
from simulator import core, matching
from simulator.core import Memory, Task
from simulator.domains.commute import DOMAIN
from simulator.policies import Family, Matched, Mixture, Never, is_stateful

TUNE = core.generate_timelines(DOMAIN, 1000, 50, 20)
HELD = core.generate_timelines(DOMAIN, 0, 50, 20)


class FirstK:
    """Toy integer family: ask the first k dependencies."""

    family = "first_k"

    def __init__(self, k: float) -> None:
        self.k, self.name = int(k), f"first_k({int(k)})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        return list(task.depends_on[: self.k])


class StatefulFirstK(FirstK):
    """Same decisions as FirstK but declared stateful (stands in for e.g. an LLM policy)."""

    family = "stateful_first_k"
    stateful = True


class StaleOver:
    """Toy continuous family: ask dependencies with P(changed) above a threshold."""

    family = "stale_over"

    def __init__(self, x: float) -> None:
        self.x, self.name = x, f"stale_over({x:.6f})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        return [a for a in task.depends_on
                if core.p_changed(core.effective_rate(DOMAIN, a), memory.age(a, t)) > self.x]


class Probe:
    """Asks one fixed attribute or nothing, regardless of memory: exposes mixture coins."""

    family = "probe"

    def __init__(self, ask: bool) -> None:
        self.ask, self.name = ask, f"probe({ask})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        return [task.depends_on[0]] if self.ask else []


FIRST_K = Family("first_k", "integer", FirstK, values=(0, 1, 2, 3, 4))
STALE_OVER = Family("stale_over", "continuous", StaleOver, bounds=(0.0, 1.0))


def used_a(w: float, tls=HELD) -> set[tuple[int, int]]:
    """(user, session) where Mixture(probe(True), probe(False), w) followed component a."""
    recs = core.run(DOMAIN, tls, [Mixture(Probe(True), Probe(False), w)])
    return {(r["user"], r["session"]) for r in recs if r["n_questions"]}


class TestMixture(unittest.TestCase):
    def test_extreme_weights_equal_components(self):
        for w, inner in ((1.0, FirstK(2)), (0.0, FirstK(1))):
            mix = core.run(DOMAIN, HELD, [Mixture(FirstK(2), FirstK(1), w)])
            ref = core.run(DOMAIN, HELD, [inner])
            self.assertEqual([r["asked"] for r in mix], [r["asked"] for r in ref])

    def test_coins_shared_across_weights_and_reproducible(self):
        lo, hi = used_a(0.3), used_a(0.6)
        self.assertLess(lo, hi)  # nested: same coins, higher weight uses a more often
        self.assertEqual(lo, used_a(0.3))
        n = 50 * 20
        self.assertAlmostEqual(len(hi) / n, 0.6, delta=0.05)

    def test_coins_depend_on_seed_and_user(self):
        other = core.generate_timelines(DOMAIN, 1, 50, 20)
        self.assertNotEqual(used_a(0.5), used_a(0.5, other))
        per_user = [{t for u, t in used_a(0.5) if u == k} for k in range(50)]
        self.assertGreater(len({frozenset(s) for s in per_user}), 40)

    def test_mixture_keeps_paired_timelines(self):
        recs = pd.DataFrame(core.run(DOMAIN, HELD, [Never(), Mixture(FirstK(3), FirstK(0), .5)]))
        key, cols = ["user", "session"], ["task", "state_hash", "true_answer"]
        views = [g.set_index(key)[cols] for _, g in recs.groupby("policy")]
        self.assertTrue(views[0].equals(views[1]))

    def test_mixture_must_stay_in_family(self):
        with self.assertRaises(ValueError):
            Mixture(FirstK(1), StaleOver(0.5), 0.5)

    def test_stateful_components_rejected(self):
        stateful = StatefulFirstK(1)
        with self.assertRaises(ValueError):
            Mixture(stateful, StatefulFirstK(2), 0.5)
        with self.assertRaises(ValueError):  # no nesting: a mixture is itself stateful
            Mixture(Mixture(FirstK(1), FirstK(2), 0.5), FirstK(3), 0.5)

    def test_stateful_flag_propagates(self):
        self.assertFalse(is_stateful(FirstK(1)))
        self.assertFalse(is_stateful(Never()))
        self.assertTrue(is_stateful(Matched(StatefulFirstK(1), "sfk", 0.3)))
        self.assertTrue(is_stateful(Mixture(FirstK(1), FirstK(2), 0.5)))

    def test_matched_wrapper_logs_role_family_target(self):
        rec = core.run(DOMAIN, HELD[:1], [Matched(FirstK(1), "first_k", 0.3)])[0]
        self.assertEqual((rec["policy"], rec["family"], rec["role"], rec["target_cost"]),
                         ("first_k@0.3", "first_k", "matched", 0.3))


class TestCalibrate(unittest.TestCase):
    def test_integer_family_uses_mixture_of_neighbours(self):
        cal = matching.calibrate(DOMAIN, TUNE, FIRST_K, 0.5, tol=0.01)
        self.assertEqual((cal.kind, cal.param_a, cal.param_b), ("mixture", 1.0, 2.0))
        self.assertLessEqual(abs(cal.tuned_cost - 0.5), 0.01)

    def test_integer_family_hits_exact_setting(self):
        always_q = matching.mean_questions(DOMAIN, TUNE, FirstK(4))
        target = matching.mean_questions(DOMAIN, TUNE, FirstK(2)) / always_q
        cal = matching.calibrate(DOMAIN, TUNE, FIRST_K, target, tol=0.01)
        self.assertEqual((cal.kind, cal.param_a), ("single", 2.0))

    def test_continuous_family(self):
        for target in (0.25, 0.5, 0.75):
            cal = matching.calibrate(DOMAIN, TUNE, STALE_OVER, target, tol=0.01)
            self.assertIn(cal.kind, ("single", "mixture"))
            self.assertLessEqual(abs(cal.tuned_cost - target), 0.01, (target, cal))

    def test_stateful_family_never_mixed(self):
        fam = Family("stateful_first_k", "integer", StatefulFirstK, values=(0, 1, 2, 3, 4))
        with self.assertRaises(ValueError):
            matching.calibrate(DOMAIN, TUNE, fam, 0.5)  # would need a mixture
        always_q = matching.mean_questions(DOMAIN, TUNE, FirstK(4))
        exact = matching.mean_questions(DOMAIN, TUNE, FirstK(2)) / always_q
        self.assertEqual(matching.calibrate(DOMAIN, TUNE, fam, exact).kind, "single")

    def test_unreachable_target(self):
        cal = matching.calibrate(DOMAIN, TUNE, FIRST_K, 1.5)
        self.assertEqual(cal.kind, "unreachable")
        self.assertIsNone(cal.build(FIRST_K))

    def test_tuned_policy_generalises_to_held_out(self):
        for fam in (FIRST_K, STALE_OVER):
            cal = matching.calibrate(DOMAIN, TUNE, fam, 0.5)
            always_q = matching.mean_questions(DOMAIN, HELD, FirstK(4))
            held = matching.question_cost(DOMAIN, HELD, cal.build(fam), always_q)
            self.assertAlmostEqual(held, 0.5, delta=0.05, msg=fam.name)


def args(**kw) -> argparse.Namespace:
    base = dict(domain="commute", policies=["never", "always"], users=30, sessions=20,
                seeds=[0], tune_seeds=[1000], rate_multiplier=1.0,
                cost_levels=[0.25, 0.5], match_families=["first_k", "stale_over"], tol=0.01,
                sweep=False)
    return argparse.Namespace(**{**base, **kw})


FAMS = {"first_k": FIRST_K, "stale_over": STALE_OVER}


class TestTuneHeldOut(unittest.TestCase):
    def test_overlapping_seeds_rejected(self):
        with self.assertRaises(ValueError):
            run_experiment(args(seeds=[0, 1000]), FAMS)

    def test_calibration_independent_of_evaluation_seeds(self):
        _, cal_a = run_experiment(args(seeds=[0]), FAMS)
        _, cal_b = run_experiment(args(seeds=[1, 2]), FAMS)
        self.assertEqual(cal_a, cal_b)
        _, cal_c = run_experiment(args(tune_seeds=[2000]), FAMS)
        self.assertNotEqual(cal_a, cal_c)

    def test_results_only_on_held_out_seeds_and_paired(self):
        df, cals = run_experiment(args(), FAMS)
        self.assertEqual(set(df.seed), {0})
        matched = df[df.role == "matched"]
        self.assertEqual(set(matched.policy),
                         {f"{c.family}@{c.target_cost:g}" for c in cals if c.kind != "unreachable"})
        key, cols = ["seed", "user", "session"], ["task", "state_hash"]
        views = [g.set_index(key)[cols] for _, g in df.groupby("policy")]
        self.assertTrue(all(v.equals(views[0]) for v in views[1:]))


if __name__ == "__main__":
    unittest.main()
