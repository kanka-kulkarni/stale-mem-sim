"""LLM impact source tests with a fake transport (no model needed)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulator import impact_llm
from simulator.domains.commute import DOMAIN
from simulator.llm import OllamaClient


class FakeTransport:
    """Replies with a rating derived from the prompt; counts calls."""

    def __init__(self, reply=lambda prompt: '{"rating": 2}'):
        self.reply, self.calls = reply, 0

    def __call__(self, url, payload, timeout):
        self.calls += 1
        return {"response": self.reply(payload["prompt"]), "prompt_eval_count": 50,
                "eval_count": 5}


class TestParse(unittest.TestCase):
    def test_valid_and_invalid(self):
        self.assertEqual(impact_llm.parse_rating('{"rating": 3}'), 3)
        self.assertEqual(impact_llm.parse_rating('{"rating": 0.0}'), 0)
        for bad in ("3", '{"rating": 4}', '{"rating": -1}', '{"rating": 1.5}',
                    '{"rating": "2"}', '{"score": 2}', "not json", '{"rating": true}', ""):
            self.assertIsNone(impact_llm.parse_rating(bad), bad)

    def test_prompt_hides_dependencies(self):
        task = DOMAIN.task("weekend_outing")
        prompt = impact_llm.build_prompt(task, DOMAIN.attribute("office_days"))
        self.assertIn("Plan a cheap weekend outing", prompt)
        self.assertIn("office days", prompt)
        self.assertNotIn("depends", prompt.lower())
        self.assertNotIn("weekly_budget", prompt)


class TestRateAndCache(unittest.TestCase):
    def test_table_normalised_invalid_counted_and_cached(self):
        def reply(prompt):
            return "garbage" if "language" in prompt else '{"rating": 3}'

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            fake = FakeTransport(reply)
            df = impact_llm.rate_impacts(DOMAIN, OllamaClient("m", cache, transport=fake))
            n = len(DOMAIN.tasks) * len(DOMAIN.attributes)
            self.assertEqual((len(df), fake.calls), (n, n))
            self.assertEqual(int((~df.valid).sum()), len(DOMAIN.tasks))  # language rows
            self.assertTrue((df[df.valid].impact == 1.0).all())
            self.assertTrue((df[~df.valid].impact == 0.0).all())
            # second client re-reads the cache file: zero new calls
            fake2 = FakeTransport(reply)
            df2 = impact_llm.rate_impacts(DOMAIN, OllamaClient("m", cache, transport=fake2))
            self.assertEqual(fake2.calls, 0)
            self.assertTrue(df2.cached.all())
            self.assertEqual(df.rating.tolist()[:5], df2.rating.tolist()[:5])
            # a different model is a different cache key
            fake3 = FakeTransport(reply)
            impact_llm.rate_impacts(DOMAIN, OllamaClient("other", cache, transport=fake3))
            self.assertEqual(fake3.calls, n)

            path = Path(tmp) / "t.csv"
            df.to_csv(path, index=False)
            table = impact_llm.load_impact_table(path, DOMAIN)
            self.assertEqual(len(table), n)
            df.iloc[:-1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                impact_llm.load_impact_table(path, DOMAIN)

    def test_request_payload_is_deterministic(self):
        seen = []

        def transport(url, payload, timeout):
            seen.append(payload)
            return {"response": json.dumps({"rating": 1})}

        OllamaClient("m", transport=transport).generate("hi")
        self.assertEqual(seen[0]["options"], {"temperature": 0.0, "seed": 0})
        self.assertEqual(seen[0]["format"], "json")



class TestThinkOption(unittest.TestCase):
    def test_unset_think_keeps_old_keys_and_payload(self):
        import hashlib
        seen = []

        def transport(url, payload, timeout):
            seen.append(payload)
            return {"response": '{"rating": 1}'}

        client = OllamaClient("m", transport=transport)
        old_blob = json.dumps({"model": "m", "prompt": "p", "options": client.options,
                               "format": "json"}, sort_keys=True)
        self.assertEqual(client.key("p"), hashlib.sha256(old_blob.encode("utf-8")).hexdigest())
        client.generate("p")
        self.assertNotIn("think", seen[0])

    def test_think_false_sent_and_keyed_separately(self):
        seen = []

        def transport(url, payload, timeout):
            seen.append(payload)
            return {"response": '{"rating": 1}'}

        off = OllamaClient("m", transport=transport, think=False)
        self.assertNotEqual(off.key("p"), OllamaClient("m").key("p"))
        off.generate("p")
        self.assertIs(seen[0]["think"], False)


class TestRetries(unittest.TestCase):
    def test_transient_failures_are_retried_and_not_cached(self):
        calls = {"n": 0}

        def flaky(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("server busy")
            return {"response": '{"rating": 1}'}

        client = OllamaClient("m", transport=flaky, retries=3, retry_wait_s=0)
        self.assertEqual(client.generate("p").text, '{"rating": 1}')
        self.assertEqual(calls["n"], 3)

    def test_gives_up_after_retries(self):
        def down(url, payload, timeout):
            raise ConnectionError("down")

        client = OllamaClient("m", transport=down, retries=2, retry_wait_s=0)
        with self.assertRaises(ConnectionError):
            client.generate("p")
        self.assertEqual(client.cache, {})


if __name__ == "__main__":
    unittest.main()
