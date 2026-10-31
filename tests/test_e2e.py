"""End-to-end study: answerer prompt, parsing, metrics, and cache-only replay."""

from __future__ import annotations

import unittest

import pandas as pd

from experiments import e2e
from experiments.e2e import ROOT
from simulator import core
from simulator.domains.commute import DOMAIN


class TestPrompt(unittest.TestCase):
    def test_matches_fixed_template(self):
        tl = core.generate_timeline(DOMAIN, 0, 0, 3)
        task = DOMAIN.task("transit_fare")
        lines = e2e.answer_prompt(DOMAIN, tl.states[1], task).split("\n")
        self.assertEqual(lines[0], "You are a personal assistant. The user has made a request. "
                                   "Here is what you know about the user:")
        self.assertEqual(len(lines), 1 + len(DOMAIN.attributes) + 3)
        first = DOMAIN.attributes[0].name
        self.assertTrue(lines[1].startswith(f"{first}: "))
        self.assertNotIn("_", lines[1].split(": ", 1)[1])  # values shown with spaces
        self.assertEqual(lines[-3], "Request: Which public transit ticket or pass should I buy?")
        self.assertEqual(lines[-2], "Choose the single best option for this user from this "
                                    "list: none_needed, pay_per_ride, ten_ride_bundle, monthly_pass")
        self.assertEqual(lines[-1], 'Reply with JSON only: {"answer": "<one option exactly as written>"}')
        self.assertNotIn("session", "\n".join(lines[1:-3]))  # no ages

    def test_parse(self):
        opts = ("a_b", "c")
        self.assertEqual(e2e.parse_answer('{"answer": "a_b"}', opts), "a_b")
        for bad in ('{"answer": "a b"}', '{"answer": 1}', "c", '{"x": "c"}', "nope"):
            self.assertIsNone(e2e.parse_answer(bad, opts), bad)


def toy_records() -> pd.DataFrame:
    """Two policies over 4 sessions of one user; tasks repeat for responsiveness."""
    tl = core.generate_timeline(DOMAIN, 0, 0, 4)
    rows = []
    for pol, answers in (("never", ["x", "y", "x", "x"]), ("always", ["x", "x", "x", "y"]),
                         ("always_all", ["r1", "r2", "r3", "r4"])):
        for t in range(1, 5):
            rows.append({"policy": pol, "seed": 0, "user": 0, "session": t,
                         "task": "weekend_outing", "true_state": tl.states[t],
                         "n_questions": {"never": 0, "always": 2, "always_all": 6}[pol],
                         "correct": True, "true_answer": f"r{t}",
                         "policy_answer": answers[t - 1] if pol != "always_all" else f"r{t}",
                         "ref_answer": f"r{t}"})
    return pd.DataFrame(rows)


class TestEvaluate(unittest.TestCase):
    def test_metrics(self):
        df = toy_records()
        df.loc[df.policy == "always", "policy_answer"] = ["r1", "r2", "x", "r4"]
        summ, comp, facts = e2e.evaluate(df, DOMAIN)
        s = summ.set_index("policy")
        self.assertEqual(s.loc["always_all", "self_agreement"], 1.0)
        self.assertEqual(s.loc["always", "self_agreement"], 0.75)
        self.assertEqual(s.loc["never", "self_agreement"], 0.0)
        self.assertAlmostEqual(s.loc["always", "gap_closed"], 0.75)
        self.assertEqual(s.loc["always_all", "question_cost"], 3.0)
        self.assertAlmostEqual(facts["irrelevant_fact_sway"], 0.25)
        self.assertEqual(len(comp), 3)
        self.assertEqual(facts["reference_rule_agreement"], 1.0)

    def test_invalid_is_wrong(self):
        df = toy_records()
        df.loc[(df.policy == "always_all") & (df.session == 1), "policy_answer"] = None
        s = e2e.evaluate(df, DOMAIN)[0].set_index("policy")
        self.assertEqual(s.loc["always_all", "self_agreement"], 0.75)
        self.assertEqual(s.loc["always_all", "invalid"], 1)


class TestReplay(unittest.TestCase):
    @unittest.skipUnless((ROOT / "results" / "m2_commute" / "m1" / "calibration.json").exists(),
                         "run configs/main.yaml first")
    def test_commute_replays_from_cache_and_always_all_equals_reference(self):
        df = e2e.records("commute")
        pols = set(df.policy)
        self.assertIn("llm(qwen2.5:7b)", pols)
        self.assertIn("llm-guided(qwen2.5:7b)", pols)
        self.assertIn("sxi[exact]@0.5", pols)
        self.assertEqual(len(df[df.policy == "never"]), e2e.EVAL_USERS * e2e.SESSIONS)
        aa = df[df.policy == "always_all"]
        self.assertTrue((aa.policy_prompt == aa.reference_prompt).all())
        nv = df[df.policy == "never"]
        self.assertFalse((nv.policy_prompt == nv.reference_prompt).all())


if __name__ == "__main__":
    unittest.main()
