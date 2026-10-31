"""Paired-design tests: determinism and identical timelines across policies."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from simulator import core
from simulator.core import Attribute, Domain, LifeEvent, Memory, Task
from simulator.domains.commute import DOMAIN
from simulator.policies import Always, Never

ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    """Wraps a policy and records the (memory-free) task it sees each session."""

    def __init__(self, inner):
        self.inner, self.name, self.seen = inner, inner.name, []

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        self.seen.append((t, task.name))
        return self.inner.decide(memory, task, t)


class TestPaired(unittest.TestCase):
    def test_same_seed_same_timeline(self):
        a = core.generate_timelines(DOMAIN, 5, 10, 20)
        b = core.generate_timelines(DOMAIN, 5, 10, 20)
        self.assertEqual(a, b)
        c = core.generate_timelines(DOMAIN, 6, 10, 20)
        self.assertNotEqual(a, c)

    def test_users_independent_of_population_size(self):
        """User u's timeline does not depend on how many users are generated."""
        small = core.generate_timelines(DOMAIN, 5, 3, 20)
        big = core.generate_timelines(DOMAIN, 5, 30, 20)
        self.assertEqual(small, big[:3])

    def test_policies_see_identical_states_and_tasks(self):
        tls = core.generate_timelines(DOMAIN, 0, 20, 20)
        never, always = Recorder(Never()), Recorder(Always())
        recs = pd.DataFrame(core.run(DOMAIN, tls, [never, always]))
        self.assertEqual(never.seen, always.seen)
        key = ["seed", "user", "session"]
        cols = ["task", "state_hash", "true_answer"]
        a = recs[recs.policy == "never"].set_index(key)[cols]
        b = recs[recs.policy == "always"].set_index(key)[cols]
        self.assertTrue(a.equals(b))
        for tl in tls:  # logged hashes match the stored timeline
            for t in range(1, tl.n_sessions + 1):
                self.assertEqual(a.loc[(0, tl.user, t), "state_hash"],
                                 core.state_hash(tl.states[t]))

    def test_running_a_policy_does_not_mutate_timeline(self):
        tls = core.generate_timelines(DOMAIN, 0, 5, 20)
        before = [tuple(dict(s) for s in tl.states) for tl in tls]
        core.run(DOMAIN, tls, [Always(), Never()])
        self.assertEqual(before, [tuple(tl.states) for tl in tls])

    def test_cli_same_seed_same_output(self):
        """End to end: two CLI runs with the same seed write identical sessions.csv."""
        outs = []
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                out = Path(tmp) / str(i)
                subprocess.run(
                    [sys.executable, "-m", "experiments.run", "--users", "10",
                     "--sessions", "10", "--seeds", "3", "--out", str(out)],
                    cwd=ROOT, check=True, capture_output=True,
                )
                outs.append((out / "sessions.csv").read_bytes())
        self.assertEqual(outs[0], outs[1])


MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 5.0)


def changes(tl: core.Timeline) -> set[tuple[int, str]]:
    """(session, attribute) pairs where the true value differs from the previous session."""
    return {(t, a) for t in range(1, tl.n_sessions + 1)
            for a in tl.states[t] if tl.states[t][a] != tl.states[t - 1][a]}


def event_domain() -> Domain:
    """Toy domain whose life event uses its own randomness when it fires."""
    attrs = (Attribute("x", ("a", "b", "c"), "fast"), Attribute("y", tuple("pqrs"), "slow"))
    ev = LifeEvent("move", 0.1, ("y",),
                   lambda s, rng: {"y": "pqrs"[int(rng.integers(4))]})
    task = Task("k", ("x", "y"), tuple(f"{i}{j}" for i in "abc" for j in "pqrs"),
                lambda v: v["x"] + v["y"])
    return Domain("toy", attrs, (task, Task("k2", ("x",), ("a", "b", "c"), lambda v: v["x"])),
                  (ev,))


class TestRateMultipliers(unittest.TestCase):
    def test_task_sequence_identical_across_multipliers(self):
        for domain in (DOMAIN, event_domain()):
            for user in range(30):
                tls = [core.generate_timeline(domain, 4, user, 40, m) for m in MULTIPLIERS]
                for tl in tls[1:]:
                    self.assertEqual(tl.tasks, tls[0].tasks, (domain.name, user))
                    self.assertEqual(tl.states[0], tls[0].states[0], (domain.name, user))

    def test_changes_at_lower_multiplier_are_subset_of_higher(self):
        strict = 0
        for user in range(100):
            sets = [changes(core.generate_timeline(DOMAIN, 4, user, 50, m)) for m in MULTIPLIERS]
            for lo, hi in zip(sets, sets[1:]):
                self.assertLessEqual(lo, hi, f"user {user}")
                strict += lo < hi
        self.assertGreater(strict, 100)  # nesting is not trivially equal sets

    def test_probabilities_capped_at_one(self):
        self.assertEqual(core.scaled(0.6, 2.0), 1.0)
        self.assertEqual(core.scaled(0.15, 0.0), 0.0)
        toy = event_domain()
        self.assertEqual(core.effective_rate(toy, "y", 40.0), 1.0)
        tl = core.generate_timeline(DOMAIN, 0, 0, 30, rate_multiplier=10.0)  # fast: 1.5 -> 1
        self.assertEqual(sum(a == "office_days" for _, a in changes(tl)), 30)


if __name__ == "__main__":
    unittest.main()
