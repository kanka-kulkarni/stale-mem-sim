"""Core model tests: rates, P(changed), memory, impact, domain validity."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from simulator import core
from simulator.core import Attribute, Domain, LifeEvent, Memory, Task
from simulator.domains.commute import DOMAIN


def all_states(domain: Domain):
    """Every combination of attribute values (commute domain is small enough)."""
    names = [a.name for a in domain.attributes]
    for combo in itertools.product(*(a.values for a in domain.attributes)):
        yield dict(zip(names, combo))


class TestRates(unittest.TestCase):
    def test_default_rates(self):
        self.assertEqual(Attribute("a", ("x", "y"), "static").base_rate, 0.0)
        self.assertEqual(Attribute("a", ("x", "y"), "slow").base_rate, 0.03)
        self.assertEqual(Attribute("a", ("x", "y"), "fast").base_rate, 0.15)
        self.assertEqual(Attribute("a", ("x", "y"), "fast", rate=0.5).base_rate, 0.5)

    def test_p_changed(self):
        self.assertEqual(core.p_changed(0.15, 0), 0.0)
        self.assertAlmostEqual(core.p_changed(0.15, 1), 0.15)
        self.assertAlmostEqual(core.p_changed(0.1, 3), 1 - 0.9**3)
        self.assertEqual(core.p_changed(0.0, 50), 0.0)

    def test_effective_rate_includes_events(self):
        attrs = (Attribute("a", ("x", "y"), "slow"), Attribute("b", ("x", "y"), "static"))
        ev = LifeEvent("e", 0.02, ("a",), lambda s, r: {"a": "y"})
        d = Domain("d", attrs, (), (ev,))
        self.assertAlmostEqual(core.effective_rate(d, "a"), 0.05)
        self.assertAlmostEqual(core.effective_rate(d, "b"), 0.0)
        self.assertAlmostEqual(core.effective_rate(d, "a", 2.0), 0.10)


class TestMemory(unittest.TestCase):
    def test_onboard_equals_truth_and_ages(self):
        s0 = {"a": "x", "b": "y"}
        m = Memory.onboard(s0)
        self.assertEqual(m.values, s0)
        self.assertIsNot(m.values, s0)
        self.assertEqual(m.age("a", 4), 4)
        m.update("a", "z", 4)
        self.assertEqual((m.values["a"], m.age("a", 6), m.age("b", 6)), ("z", 2, 6))


class TestImpact(unittest.TestCase):
    def test_exact_impact_of_irrelevant_attributes_is_zero(self):
        """For every task and every memory state, non-dependencies have exact impact 0."""
        for task in DOMAIN.tasks:
            irrelevant = [a.name for a in DOMAIN.attributes if a.name not in task.depends_on]
            for state in all_states(DOMAIN):
                for a in irrelevant:
                    self.assertEqual(core.impact_exact(DOMAIN, task, a, state), 0.0)

    def test_every_dependency_matters_somewhere(self):
        """Declared dependencies are not dead: each has positive impact in some state."""
        for task in DOMAIN.tasks:
            for a in task.depends_on:
                self.assertTrue(
                    any(core.impact_exact(DOMAIN, task, a, s) > 0 for s in all_states(DOMAIN)),
                    f"{task.name}.{a}",
                )

    def test_exact_impact_value(self):
        task = DOMAIN.task("weekend_outing")
        mem = {s: v for s, v in next(all_states(DOMAIN)).items()}
        mem.update(weekly_budget="low", has_car="yes")
        # low -> local_park; medium -> nearby_hike; high -> day_trip_by_car: both alts flip
        self.assertEqual(core.impact_exact(DOMAIN, task, "weekly_budget", mem), 1.0)
        # with low budget the car doesn't matter
        self.assertEqual(core.impact_exact(DOMAIN, task, "has_car", mem), 0.0)

    def test_coarse_impact(self):
        task = DOMAIN.task("weekend_outing")
        self.assertEqual(core.impact_coarse(task, "has_car"), 1.0)
        self.assertEqual(core.impact_coarse(task, "office_days"), 0.0)


class TestCommuteDomain(unittest.TestCase):
    def test_answers_are_valid_options_everywhere(self):
        for task in DOMAIN.tasks:
            for s in all_states(DOMAIN):
                task.answer(s)  # raises on unknown option

    def test_has_task_where_fast_attribute_is_irrelevant(self):
        fast = {a.name for a in DOMAIN.attributes if a.volatility == "fast"}
        self.assertTrue(any(not fast & set(k.depends_on) for k in DOMAIN.tasks))

    def test_every_value_change_can_change_some_answer(self):
        """No dead value pairs: switching any attribute between any two of its values
        changes some task's answer in some state."""
        from simulator.diagnostics import dead_value_pairs
        self.assertEqual(dead_value_pairs(DOMAIN), [])

    def test_task_count(self):
        self.assertTrue(3 <= len(DOMAIN.tasks) <= 5)


class TestDrift(unittest.TestCase):
    def test_changes_go_to_a_different_value_and_static_never_changes(self):
        tl = core.generate_timeline(DOMAIN, seed=1, user=0, n_sessions=400)
        lang = {s["preferred_language"] for s in tl.states}
        self.assertEqual(len(lang), 1)
        office_changes = sum(
            a["office_days"] != b["office_days"] for a, b in zip(tl.states, tl.states[1:])
        )
        # fast rate 0.15 over 400 sessions -> ~60 changes
        self.assertTrue(35 < office_changes < 90, office_changes)

    def test_empirical_rate_matches_p(self):
        tls = core.generate_timelines(DOMAIN, seed=3, n_users=200, n_sessions=50)
        for attr, p in (("office_days", 0.15), ("home_area", 0.03)):
            changes = np.mean([
                a[attr] != b[attr] for tl in tls for a, b in zip(tl.states, tl.states[1:])
            ])
            self.assertAlmostEqual(changes, p, delta=0.015, msg=attr)

    def test_rate_multiplier_zero_freezes_state(self):
        tl = core.generate_timeline(DOMAIN, seed=0, user=0, n_sessions=30, rate_multiplier=0.0)
        self.assertTrue(all(s == tl.states[0] for s in tl.states))

    def test_unknown_dependency_rejected(self):
        with self.assertRaises(ValueError):
            Domain("d", (Attribute("a", ("x", "y"), "slow"),),
                   (Task("k", ("b",), ("o",), lambda v: "o"),))


if __name__ == "__main__":
    unittest.main()
