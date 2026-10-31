"""Per-session LLM policy and driver tests with a fake transport (no model needed)."""

from __future__ import annotations

import argparse
import json
import unittest

from experiments import m3
from experiments.rater_report import spearman
from simulator import core
from simulator.domains.commute import DOMAIN
from simulator.llm import OllamaClient
from simulator.llm_policy import LLMPolicy, build_prompt, parse_recheck
from simulator.policies import Mixture, TTL, is_stateful

import pandas as pd

NAMES = {a.name for a in DOMAIN.attributes}


def fake_client(reply) -> OllamaClient:
    def transport(url, payload, timeout):
        return {"response": reply(payload["prompt"]), "prompt_eval_count": 300,
                "eval_count": 12}
    return OllamaClient("fake", transport=transport)


def ask_office_days(prompt: str) -> str:
    return json.dumps({"recheck": ["office_days"]})


class TestParse(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_recheck('{"recheck": ["has_car"]}', NAMES), ["has_car"])
        self.assertEqual(parse_recheck('{"recheck": []}', NAMES), [])
        self.assertEqual(parse_recheck('{"recheck": ["has_car", "has_car"]}', NAMES),
                         ["has_car"])
        for bad in ('{"recheck": ["salary"]}', '{"recheck": "has_car"}', '{"ask": []}',
                    "nope", '["has_car"]', '{"recheck": [1]}'):
            self.assertIsNone(parse_recheck(bad, NAMES), bad)

    def test_prompt_shows_values_and_ages_not_dependencies(self):
        tl = core.generate_timeline(DOMAIN, 0, 0, 5)
        mem = core.Memory.onboard(tl.states[0])
        mem.update("office_days", tl.states[3]["office_days"], 3)
        prompt = build_prompt(DOMAIN, mem, DOMAIN.task("weekend_outing"), 5)
        self.assertIn("office_days", prompt)
        self.assertIn("last confirmed 2 sessions ago", prompt)
        self.assertIn("last confirmed 5 sessions ago", prompt)
        self.assertIn("Plan a cheap weekend outing", prompt)
        self.assertNotIn("depends", prompt.lower())
        self.assertNotIn("0.15", prompt)  # no rates


class TestGuidedPrompt(unittest.TestCase):
    def test_guided_replaces_only_the_instruction(self):
        from simulator.llm_policy import GUIDED_INSTRUCTION, PLAIN_INSTRUCTION
        tl = core.generate_timeline(DOMAIN, 0, 0, 5)
        mem = core.Memory.onboard(tl.states[0])
        task = DOMAIN.task("commute_plan")
        plain = build_prompt(DOMAIN, mem, task, 3)
        guided = build_prompt(DOMAIN, mem, task, 3, "guided")
        self.assertTrue(plain.startswith(PLAIN_INSTRUCTION + "\n\n"))
        self.assertTrue(guided.startswith(GUIDED_INSTRUCTION + "\n\n"))
        self.assertEqual(plain[len(PLAIN_INSTRUCTION):], guided[len(GUIDED_INSTRUCTION):])
        self.assertIn("Step 3: re-check only facts", guided)

    def test_policy_name_and_style(self):
        pol = LLMPolicy(DOMAIN, fake_client(ask_office_days), "guided")
        self.assertEqual((pol.name, pol.family), ("llm-guided(fake)", "llm"))
        with self.assertRaises(ValueError):
            LLMPolicy(DOMAIN, fake_client(ask_office_days), "cot")


class TestPolicy(unittest.TestCase):
    def test_decisions_invalid_and_log(self):
        replies = iter(['{"recheck": ["office_days"]}', "garbage"] * 100)
        pol = LLMPolicy(DOMAIN, fake_client(lambda p: next(replies)))
        tls = core.generate_timelines(DOMAIN, 0, 2, 5)
        df = pd.DataFrame(core.run(DOMAIN, tls, [pol]))
        log = pd.DataFrame(pol.call_log())
        self.assertEqual(len(log), 10)
        self.assertEqual(int((~log.valid).sum()), 5)
        self.assertEqual(df.n_questions.tolist(), [1, 0] * 5)  # invalid -> ask nothing
        self.assertEqual(sorted(set(log.user)), [0, 1])
        self.assertEqual(log.prompt_tokens.iloc[0], 300)

    def test_stateful_never_mixed(self):
        pol = LLMPolicy(DOMAIN, fake_client(ask_office_days))
        self.assertTrue(is_stateful(pol))
        with self.assertRaises(ValueError):
            Mixture(pol, TTL(DOMAIN, 3), 0.5)


class TestDriver(unittest.TestCase):
    def test_protocol_end_to_end_with_fake_model(self):
        a = argparse.Namespace(domain="commute", model="fake", sessions=10, llm_tune_seed=1000,
                               llm_tune_users=4, tune_seeds=[1000, 1001], tune_users=10,
                               eval_seed=0, eval_users=6, llm_impact=None, tol=0.01, out=None)
        res = m3.run(a, fake_client(ask_office_days))
        f = res["facts"]
        self.assertGreater(f["estimate"], 0)
        s = res["summary"].set_index("policy")
        self.assertIn(f"llm@{f['estimate']:g}", s.index)
        self.assertEqual(set(res["calls"].phase), {"tune", "held_out"})
        self.assertEqual((res["calls"].phase == "held_out").sum(), 6 * 10)
        c = res["comparisons"]
        llm_rows = c[c.policy_a.str.startswith("llm@") | c.policy_b.str.startswith("llm@")]
        self.assertGreater(len(llm_rows), 0)
        if f["flag_drift"]:
            self.assertTrue(llm_rows.flagged.all())
        self.assertEqual(set(res["sessions"].seed), {0})

    def test_overlapping_seeds_rejected(self):
        a = argparse.Namespace(domain="commute", model="fake", sessions=5, llm_tune_seed=0,
                               llm_tune_users=2, tune_seeds=[1000], tune_users=5,
                               eval_seed=0, eval_users=2, llm_impact=None, tol=0.01, out=None)
        with self.assertRaises(ValueError):
            m3.run(a, fake_client(ask_office_days))


class TestSpearman(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(spearman(pd.Series([1, 2, 3]), pd.Series([10, 20, 30])), 1.0)
        self.assertAlmostEqual(spearman(pd.Series([1, 2, 3]), pd.Series([3, 2, 1])), -1.0)
        self.assertTrue(pd.isna(spearman(pd.Series([1, 1, 1]), pd.Series([1, 2, 3]))))


if __name__ == "__main__":
    unittest.main()
