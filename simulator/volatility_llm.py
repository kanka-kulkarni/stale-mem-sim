"""LLM-rated volatility: a local model rates, once per fact,
how often that kind of fact typically changes for a person, on a 0-3 scale. The prompt
never mentions our volatility classes or change rates.

The fixed mapping to a per-session change probability:
0 -> 0, 1 -> slow default, 2 -> midpoint of slow and fast defaults, 3 -> fast default.
Invalid replies are counted and mapped to rating 0.
"""

from __future__ import annotations

import pandas as pd

from simulator.core import DEFAULT_RATES, Attribute, Domain
from simulator.impact_llm import parse_rating
from simulator.llm import OllamaClient

RATE_OF_RATING = {0: 0.0, 1: DEFAULT_RATES["slow"],
                  2: (DEFAULT_RATES["slow"] + DEFAULT_RATES["fast"]) / 2,
                  3: DEFAULT_RATES["fast"]}
CLASS_ORDER = {"static": 0, "slow": 1, "fast": 2}
COLUMNS = ["domain", "attribute", "volatility", "rating", "valid", "rate", "latency_s",
           "prompt_tokens", "completion_tokens", "cached", "model"]


def build_prompt(attr: Attribute) -> str:
    """Volatility prompt for one fact (no classes or rates mentioned)."""
    values = ", ".join(v.replace("_", " ") for v in attr.values)
    return (
        "You help a personal assistant that remembers facts about its users from earlier "
        "conversations.\n\n"
        f"Remembered fact: {attr.name.replace('_', ' ')} (possible values: {values})\n\n"
        "In general, how often does this kind of fact change for a person?\n"
        "0 = never, 1 = rarely, 2 = sometimes, 3 = often.\n"
        'Reply with JSON only, in the form {"rating": <0, 1, 2 or 3>}.'
    )


def rate_volatility(domain: Domain, client: OllamaClient, label: str | None = None) -> pd.DataFrame:
    """Rate every fact of a domain once; one row per fact."""
    rows = []
    for attr in domain.attributes:
        resp = client.generate(build_prompt(attr))
        rating = parse_rating(resp.text)
        rows.append({"domain": domain.name, "attribute": attr.name,
                     "volatility": attr.volatility, "rating": rating,
                     "valid": rating is not None,
                     "rate": RATE_OF_RATING[rating if rating is not None else 0],
                     "latency_s": resp.latency_s, "prompt_tokens": resp.prompt_tokens,
                     "completion_tokens": resp.completion_tokens, "cached": resp.cached,
                     "model": label or client.model})
    return pd.DataFrame(rows, columns=COLUMNS)


def rate_table(df: pd.DataFrame) -> dict[str, float]:
    """{fact: mapped per-session change probability} from a rating table."""
    return {r.attribute: float(r.rate) for r in df.itertuples()}
