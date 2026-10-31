"""Calibration: VitaBench rate derivation, calibrated domain, assumed-rate and
frozen-calibration conditions."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from calibration.vitabench_rates import derive, wilson
from experiments.run import run_experiment
from simulator import core
from simulator.domains import get_domain
from simulator.domains.meals import CALIBRATED_RATES, CALIBRATION_SOURCE
from simulator.policies import StalenessXImpact

ROOT = Path(__file__).resolve().parents[1]
FOOD, SHOP = "饮食偏好", "消费购物偏好"


def subtask(day: str, current: dict, history: list[tuple[str, str]]) -> dict:
    return {"current_time": f"{day} 10:00:00", "user_scenario": {
        "personalized_preference_memory": {
            "current": current,
            "preference_tag_change_history": [{"content": c, "type": t, "source": ""}
                                              for c, t in history]}}}


class TestDerive(unittest.TestCase):
    def test_counts_on_hand_built_user(self):
        user = {"subtasks": [
            subtask("2026-01-01", {FOOD: ["a", "b"], SHOP: ["x"]},
                    [("a", "create"), ("b", "create"), ("x", "create")]),
            # b updated to b2, x deleted, new tag c created
            subtask("2026-01-11", {FOOD: ["a", "b2", "c"], SHOP: []},
                    [("a", "unchanged"), ("b2", "update"), ("c", "create"), ("x", "delete")]),
            subtask("2026-01-31", {FOOD: ["a", "b2", "c"], SHOP: []},
                    [("a", "unchanged"), ("b2", "unchanged"), ("c", "unchanged")]),
        ]}
        df, facts = derive([user])
        food = df.set_index("category").loc["food"]
        shop = df.set_index("category").loc["shopping_spending"]
        self.assertEqual((food["update"], food["delete"], food["create"]), (1, 0, 1))
        self.assertEqual(food.tag_exposure, 2 + 3)  # tags held at start of each transition
        self.assertAlmostEqual(food.per_tag_rate, 1 / 5)
        self.assertAlmostEqual(food.p_category_changed, 1 / 2)
        self.assertEqual((shop["delete"], shop["tag_exposure"]), (1, 1))
        self.assertEqual((facts["transitions"], facts["gap_days_median"],
                          facts["unmapped_entries"]), (2, 20, 0))

    def test_wilson(self):
        lo, hi = wilson(10, 100)
        self.assertTrue(lo < 0.1 < hi)
        self.assertAlmostEqual(lo, 0.0552, places=3)


class TestCalibratedDomain(unittest.TestCase):
    @unittest.skipUnless((ROOT / "calibration" / "vitabench_rates.csv").exists(),
                         "run configs/calibration.yaml (stage vitabench) first")
    def test_rates_match_derived_csv(self):
        df = pd.read_csv(ROOT / "calibration" / "vitabench_rates.csv").set_index("category")
        for attr, rate in CALIBRATED_RATES.items():
            self.assertAlmostEqual(rate, df.loc[CALIBRATION_SOURCE[attr], "per_tag_rate"],
                                   places=3, msg=attr)

    def test_only_rates_differ(self):
        cal, base = get_domain("meals_calibrated"), get_domain("meals")
        self.assertEqual([k.name for k in cal.tasks], [k.name for k in base.tasks])
        for a, b in zip(cal.attributes, base.attributes):
            self.assertEqual((a.name, a.values, a.volatility), (b.name, b.values, b.volatility))
            self.assertEqual(a.base_rate, CALIBRATED_RATES.get(a.name, b.base_rate))

    def test_policy_can_assume_default_rates(self):
        cal, base = get_domain("meals_calibrated"), get_domain("meals")
        right = StalenessXImpact(cal, 0.1, "coarse")
        wrong = StalenessXImpact(cal, 0.1, "coarse", rate_domain=base)
        self.assertAlmostEqual(right.rates["cuisine_likes"], 0.059)
        self.assertAlmostEqual(wrong.rates["cuisine_likes"], 0.03)
        self.assertEqual(right.rates["shopping_day"], wrong.rates["shopping_day"])


def args(**kw) -> argparse.Namespace:
    base = dict(domain="commute", policies=["never", "always"], users=20, sessions=20,
                seeds=[0], tune_seeds=[1000], rate_multiplier=1.0, assumed_multiplier=None,
                assumed_rates="true", frozen_calibration=None, llm_impact=None,
                cost_levels=[0.25, 0.5], match_families=["sxi[exact]", "budget[coarse]"],
                tol=0.01, sweep=False)
    return argparse.Namespace(**{**base, **kw})


class TestFrozen(unittest.TestCase):
    def test_frozen_reproduces_tuned_policies_in_same_environment(self):
        df, cals = run_experiment(args())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps([c.to_dict() for c in cals]))
            df2, cals2 = run_experiment(args(frozen_calibration=path))
        self.assertEqual(cals, cals2)
        cols = ["policy", "seed", "user", "session", "asked", "correct"]
        a = df[df.role == "matched"][cols].reset_index(drop=True)
        b = df2[df2.role == "matched"][cols].reset_index(drop=True)
        self.assertTrue(a.equals(b))

    def test_frozen_at_other_multiplier_does_not_retune(self):
        _, cals = run_experiment(args())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps([c.to_dict() for c in cals]))
            df, cals2 = run_experiment(args(frozen_calibration=path, rate_multiplier=2.0,
                                            assumed_multiplier=1.0))
        self.assertEqual(cals, cals2)  # same thresholds / weights, no tuning at 2x
        self.assertEqual(set(df.rate_multiplier), {2.0})


if __name__ == "__main__":
    unittest.main()
