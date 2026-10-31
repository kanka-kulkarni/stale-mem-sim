"""Value-study cells: reused cells reproduce the existing policies decision for decision;
AP / AUC on known cases; the random reference policy is deterministic per user."""

import unittest

import numpy as np

from experiments.value_study import (RandomPolicy, ValuePolicy, average_precision, roc_auc,
                                     rating_table)
from simulator import core
from simulator.domains import get_domain
from simulator.policies import TTL, Budget, InfoGain, StalenessXImpact


def decisions(domain, policy, timelines):
    return [r["asked"] for r in core.run(domain, timelines, [policy])]


class TestReusedCells(unittest.TestCase):
    def setUp(self):
        self.cases = []
        for name in ("commute", "meals_calibrated"):
            d = get_domain(name)
            self.cases.append((d, core.generate_timelines(d, 1000, 15, 20)))

    def test_cells_equal_existing_policies(self):
        for d, tls in self.cases:
            table = rating_table(d, "llm-7b")
            pairs = [
                (ValuePolicy(d, "rate", "coarse", "threshold", 0.05), StalenessXImpact(d, 0.05, "coarse")),
                (ValuePolicy(d, "rate", "exact", "threshold", 0.02), StalenessXImpact(d, 0.02, "exact")),
                (ValuePolicy(d, "rate", "exact", "budget", 2), Budget(d, 2, "exact")),
                (ValuePolicy(d, "rate", "coarse", "budget", 1), Budget(d, 1, "coarse")),
                (ValuePolicy(d, "rate", "none", "threshold", 0.1), InfoGain(d, 0.1)),
                (ValuePolicy(d, "age", "coarse", "threshold", 4), TTL(d, 4)),
                (ValuePolicy(d, "rate", "llm-7b", "threshold", 0.03, table),
                 StalenessXImpact(d, 0.03, "llm", impact_table=table)),
                (ValuePolicy(d, "rate", "llm-7b", "budget", 2, table),
                 Budget(d, 2, "llm", impact_table=table)),
            ]
            for new, old in pairs:
                self.assertEqual(decisions(d, new, tls), decisions(d, old, tls), new.name)

    def test_none_none_budget_takes_declaration_order(self):
        d, tls = self.cases[0]
        first_two = ";".join(a.name for a in d.attributes[:2])
        self.assertTrue(all(x == first_two for x in decisions(d, ValuePolicy(d, "none", "none", "budget", 2), tls)))


class TestRanking(unittest.TestCase):
    def test_average_precision(self):
        y = np.array([True, False, True, False])
        self.assertAlmostEqual(average_precision(np.array([4., 3., 2., 1.]), y), (1 + 2 / 3) / 2)
        self.assertAlmostEqual(average_precision(np.array([4., 1., 3., 0.]), y), 1.0)
        self.assertAlmostEqual(average_precision(np.ones(4), y), 0.5)  # all tied

    def test_roc_auc(self):
        y = np.array([True, False, True, False])
        self.assertAlmostEqual(roc_auc(np.array([4., 1., 3., 0.]), y), 1.0)
        self.assertAlmostEqual(roc_auc(np.array([0., 3., 1., 4.]), y), 0.0)
        self.assertAlmostEqual(roc_auc(np.ones(4), y), 0.5)


class TestRandomPolicy(unittest.TestCase):
    def test_deterministic_and_rate(self):
        d = get_domain("commute")
        tls = core.generate_timelines(d, 1000, 20, 20)
        a = decisions(d, RandomPolicy(d, 0.3), tls)
        self.assertEqual(a, decisions(d, RandomPolicy(d, 0.3), tls))
        asked = sum(len(x.split(";")) if x else 0 for x in a)
        self.assertAlmostEqual(asked / (len(a) * len(d.attributes)), 0.3, delta=0.03)



class TestE345Policies(unittest.TestCase):
    def test_unperturbed_levels_equal_exact_cell(self):
        from experiments.value_study_e345 import Perturbed
        for name in ("commute", "meals_calibrated"):
            d = get_domain(name)
            tls = core.generate_timelines(d, 1000, 15, 20)
            for sel, param in (("threshold", 0.03), ("budget", 2)):
                as_sets = lambda p: [set(x.split(";")) - {""} for x in decisions(d, p, tls)]  # noqa: E731
                ref = as_sets(ValuePolicy(d, "rate", "exact", sel, param))
                for kind in ("noise", "over"):
                    self.assertEqual(as_sets(Perturbed(d, kind, 0.0, sel, param)), ref)

    def test_noise_is_fixed_per_user_and_mixable(self):
        from experiments.value_study_e345 import Perturbed
        from simulator.policies import Mixture
        d = get_domain("commute")
        tls = core.generate_timelines(d, 1000, 10, 20)
        a = Perturbed(d, "noise", 0.2, "budget", 2)
        self.assertEqual(decisions(d, a, tls), decisions(d, Perturbed(d, "noise", 0.2, "budget", 2), tls))
        Mixture(Perturbed(d, "noise", 0.2, "budget", 1), Perturbed(d, "noise", 0.2, "budget", 2), 0.5)

    def test_error_target_leaves_at_most_eps(self):
        from experiments.value_study_e345 import ErrorTarget
        d = get_domain("meals_calibrated")
        pol = ErrorTarget(d, 0.1)
        mem = core.Memory.onboard(core.generate_timeline(d, 1000, 0, 20).states[0])
        for task in d.tasks:
            asked = pol.decide(mem, task, 15)
            left = sum(pol.p_changed(mem, a, 15) for a in task.depends_on if a not in asked)
            self.assertLessEqual(left, 0.1 + 1e-12)


if __name__ == "__main__":
    unittest.main()
