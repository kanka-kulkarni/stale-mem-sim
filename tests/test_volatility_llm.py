"""LLM-rated volatility: the prompt reveals nothing about our classes or rates, and the
rating-to-rate mapping is the fixed one."""

import re
import unittest

from simulator.domains import get_domain
from simulator.volatility_llm import RATE_OF_RATING, build_prompt


class TestVolatilityPrompt(unittest.TestCase):
    def test_prompt_mentions_no_class_or_rate(self):
        for name in ("commute", "meals_calibrated"):
            for attr in get_domain(name).attributes:
                text = build_prompt(attr).lower()
                for word in (r"\bstatic\b", r"\bslow", r"\bfast", "volatil", "probab",
                             "per session", r"0\.03", r"0\.15", r"\brates?\b"):
                    self.assertIsNone(re.search(word, text), (attr.name, word))
                self.assertIn(attr.name.replace("_", " "), text)
                self.assertIn("0 = never, 1 = rarely, 2 = sometimes, 3 = often", text)

    def test_mapping_is_pre_committed(self):
        self.assertEqual(RATE_OF_RATING, {0: 0.0, 1: 0.03, 2: 0.09, 3: 0.15})


if __name__ == "__main__":
    unittest.main()
