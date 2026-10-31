"""Decision-table diagnostics: how much each attribute can move each task's answer.

Used to catch dead or near-dead attributes (e.g. a fast attribute whose value changes
almost never flip any answer). Not used by the simulator or policies.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np
import pandas as pd

from simulator.core import Domain, State, impact_exact

MAX_ENUMERATED = 20_000  # above this many states, sample instead of enumerating


def reference_states(domain: Domain, n_samples: int = 5_000, seed: int = 0) -> list[State]:
    """All attribute-value combinations if small enough, else a uniform random sample."""
    names = [a.name for a in domain.attributes]
    total = int(np.prod([len(a.values) for a in domain.attributes]))
    if total <= MAX_ENUMERATED:
        return [dict(zip(names, c)) for c in itertools.product(*(a.values for a in domain.attributes))]
    rng = np.random.default_rng(seed)
    return [{a.name: a.values[int(rng.integers(len(a.values)))] for a in domain.attributes}
            for _ in range(n_samples)]


def impact_table(domain: Domain, states: list[State] | None = None) -> pd.DataFrame:
    """Mean exact impact of each attribute (rows) on each task (columns), averaged
    uniformly over reference memory states."""
    states = states or reference_states(domain)
    data = {
        task.name: [np.mean([impact_exact(domain, task, a.name, s) for s in states])
                    for a in domain.attributes]
        for task in domain.tasks
    }
    return pd.DataFrame(data, index=[a.name for a in domain.attributes])


def _pairs(values: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    return itertools.combinations(values, 2)


def pair_flip_table(domain: Domain, attr: str, states: list[State] | None = None) -> pd.DataFrame:
    """For each unordered value pair (v, w) of `attr` and each task: the fraction of
    reference states in which switching attr between v and w changes the answer."""
    states = states or reference_states(domain)
    values = domain.attribute(attr).values
    rows = {}
    for v, w in _pairs(values):
        rows[f"{v} <-> {w}"] = {
            task.name: np.mean([task.answer({**s, attr: v}) != task.answer({**s, attr: w})
                                for s in states])
            for task in domain.tasks
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["any_task"] = df.max(axis=1)
    return df


def dead_value_pairs(domain: Domain, states: list[State] | None = None) -> list[tuple[str, str, str]]:
    """(attribute, v, w) for every value pair whose switch never changes any task's answer."""
    states = states or reference_states(domain)
    dead = []
    for a in domain.attributes:
        for v, w in _pairs(a.values):
            if not any(task.answer({**s, a.name: v}) != task.answer({**s, a.name: w})
                       for task in domain.tasks for s in states):
                dead.append((a.name, v, w))
    return dead
