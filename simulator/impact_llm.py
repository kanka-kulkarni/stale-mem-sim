"""LLM impact source: a local model rates each (task type, attribute) pair once.

The model sees the user's request in plain words and one remembered fact (its name
and possible values), never the task's dependency list or decision rule, and rates
on a 0-3 scale how likely a change in that fact would change the best answer.
Ratings are normalised to [0, 1] (rating / 3) and stored as a table that the sxi and
budget policies use as their "llm" impact version.

Invalid replies (not JSON, or no integer rating in 0..3) get impact 0 and are counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simulator.core import Attribute, Domain, Task
from simulator.llm import OllamaClient

SCALE_MAX = 3
COLUMNS = ["domain", "task", "attribute", "rating", "valid", "impact", "latency_s",
           "prompt_tokens", "completion_tokens", "cached", "model"]


def build_prompt(task: Task, attr: Attribute) -> str:
    """Prompt for one (task, attribute) rating. Reveals no dependency information."""
    values = ", ".join(v.replace("_", " ") for v in attr.values)
    request = task.description or task.name.replace("_", " ")
    return (
        "You help a personal assistant decide which remembered facts about a user to "
        "double-check before answering a request.\n\n"
        f'User request: "{request}"\n'
        f"Remembered fact: {attr.name.replace('_', ' ')} (possible values: {values})\n\n"
        "If this fact had changed since the assistant last checked it, how likely is it "
        "that the best answer to the request would change?\n"
        "0 = not at all, 1 = slightly, 2 = moderately, 3 = very likely.\n"
        'Reply with JSON only, in the form {"rating": <0, 1, 2 or 3>}.'
    )


def parse_rating(text: str) -> int | None:
    """Integer rating in 0..3 from a JSON reply, or None if invalid."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or "rating" not in obj:
        return None
    r = obj["rating"]
    if isinstance(r, bool) or not isinstance(r, (int, float)) or r != int(r):
        return None
    return int(r) if 0 <= r <= SCALE_MAX else None


def rate_impacts(domain: Domain, client: OllamaClient) -> pd.DataFrame:
    """Rate every (task, attribute) pair once; one row per pair."""
    rows = []
    for task in domain.tasks:
        for attr in domain.attributes:
            resp = client.generate(build_prompt(task, attr))
            rating = parse_rating(resp.text)
            rows.append({
                "domain": domain.name, "task": task.name, "attribute": attr.name,
                "rating": rating, "valid": rating is not None,
                "impact": (rating or 0) / SCALE_MAX, "latency_s": resp.latency_s,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens, "cached": resp.cached,
                "model": client.model,
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def load_impact_table(path: Path, domain: Domain) -> dict[tuple[str, str], float]:
    """Read a ratings CSV into {(task, attribute): impact}; must cover every pair."""
    df = pd.read_csv(path)
    if (df.domain == domain.name).any():  # else e.g. meals ratings reused for meals_calibrated
        df = df[df.domain == domain.name]
    table = {(r.task, r.attribute): float(r.impact) for r in df.itertuples()}
    missing = [(k.name, a.name) for k in domain.tasks for a in domain.attributes
               if (k.name, a.name) not in table]
    if missing:
        raise ValueError(f"impact table {path} misses {len(missing)} pairs, e.g. {missing[:3]}")
    return table


def agreement(domain: Domain, table: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Compare llm impact with the dependency lists: per task, which declared
    dependencies got impact 0 (misses) and which non-dependencies got > 0 (extras)."""
    rows = []
    for task in domain.tasks:
        deps = set(task.depends_on)
        rated = {a.name for a in domain.attributes if table[(task.name, a.name)] > 0}
        rows.append({"task": task.name, "deps": len(deps),
                     "missed_deps": ";".join(sorted(deps - rated)),
                     "extra_attrs": ";".join(sorted(rated - deps))})
    return pd.DataFrame(rows)
