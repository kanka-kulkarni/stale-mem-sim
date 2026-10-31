"""Meal-planning case study: domain validity, life events, sanity checks."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from simulator import core, diagnostics
from simulator.domains.meals import DOMAIN, EVENTS
from simulator.policies import Always, Never

STATES = diagnostics.reference_states(DOMAIN, n_samples=3000, seed=1)
FAST = {a.name for a in DOMAIN.attributes if a.volatility == "fast"}


class TestDomain(unittest.TestCase):
    def test_answers_valid(self):
        for task in DOMAIN.tasks:
            for s in STATES:
                task.answer(s)

    def test_task_mix(self):
        self.assertTrue(6 <= len(DOMAIN.tasks) <= 8)
        kinds = {("fast" if set(k.depends_on) <= FAST else
                  "slow" if not set(k.depends_on) & FAST else "mix") for k in DOMAIN.tasks}
        self.assertEqual(kinds, {"fast", "slow", "mix"})

    def test_every_dependency_matters_and_no_dead_pairs(self):
        for task in DOMAIN.tasks:
            for a in task.depends_on:
                self.assertTrue(any(core.impact_exact(DOMAIN, task, a, s) > 0 for s in STATES),
                                f"{task.name}.{a}")
            for a in DOMAIN.attributes:
                if a.name not in task.depends_on:
                    self.assertTrue(all(core.impact_exact(DOMAIN, task, a.name, s) == 0
                                        for s in STATES[:300]))
        self.assertEqual(diagnostics.dead_value_pairs(DOMAIN, STATES), [])

    def test_no_health_framing(self):
        words = " ".join([a.name for a in DOMAIN.attributes]
                         + [v for a in DOMAIN.attributes for v in a.values]
                         + [k.description for k in DOMAIN.tasks]).lower()
        for banned in ("calorie", "weight", "diabet", "allerg", "nutrition", "health"):
            self.assertNotIn(banned, words)


class TestEvents(unittest.TestCase):
    def test_events_touch_only_declared_attributes_with_valid_values(self):
        rng = np.random.default_rng(0)
        for ev in EVENTS:
            for s in STATES[:500]:
                out = ev.apply(dict(s), rng)
                self.assertLessEqual(set(out), set(ev.touches), ev.name)
                for a, v in out.items():
                    self.assertIn(v, DOMAIN.attribute(a).values)

    def test_new_household_member_direction(self):
        ev = next(e for e in EVENTS if e.name == "new_household_member")
        out = ev.apply({"household_size": "2", "weekly_grocery_budget": "moderate",
                        "cooking_time_weeknights": "30min"}, np.random.default_rng(0))
        self.assertEqual(out, {"household_size": "3-4", "weekly_grocery_budget": "tight",
                               "cooking_time_weeknights": "15min"})

    def test_effective_rates_include_events(self):
        self.assertAlmostEqual(core.effective_rate(DOMAIN, "shopping_day"), 0.15 + 0.01 + 0.02)
        self.assertAlmostEqual(core.effective_rate(DOMAIN, "household_size"), 0.03 + 0.01)
        self.assertAlmostEqual(core.effective_rate(DOMAIN, "cuisine_likes"), 0.03)

    def test_events_fire_at_expected_rate_and_move_attributes_together(self):
        tls = core.generate_timelines(DOMAIN, 2, 300, 40)
        joint = sum(a["nearby_stores"] != b["nearby_stores"] and
                    a["shopping_day"] != b["shopping_day"]
                    for tl in tls for a, b in zip(tl.states, tl.states[1:]))
        # independent: 0.03 * 0.15 ~ 0.0045; with moved_house (0.01 * 2/3) clearly more
        self.assertGreater(joint / (300 * 40), 0.008)

    def test_multipliers_keep_tasks_aligned(self):
        for u in range(20):
            a = core.generate_timeline(DOMAIN, 3, u, 30, 0.5)
            b = core.generate_timeline(DOMAIN, 3, u, 30, 2.0)
            self.assertEqual(a.tasks, b.tasks)


class TestSanity(unittest.TestCase):
    def test_always_perfect_never_declines(self):
        tls = core.generate_timelines(DOMAIN, 0, 300, 20)
        df = pd.DataFrame(core.run(DOMAIN, tls, [Never(), Always()]))
        self.assertEqual(df[df.policy == "always"].correct.mean(), 1.0)
        acc = df[df.policy == "never"].groupby("session").correct.mean()
        self.assertGreater(acc.iloc[:5].mean() - acc.iloc[-5:].mean(), 0.1)


if __name__ == "__main__":
    unittest.main()
