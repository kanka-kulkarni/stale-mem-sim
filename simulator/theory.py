"""Selection rules and bounds from the formulation (Proposition 1), stated on plain
per-fact value estimates so they can be checked numerically.

Notation: r[a] is the value of re-asking fact a, the chance that asking a fixes the
answer; under separability the expected number of corrected answers from asking a set
Q is sum(r[a] for a in Q). P[a] is the probability that a changed since it was last
confirmed.
"""

from __future__ import annotations

from typing import Iterable, Mapping


def top_k(r: Mapping[str, float], k: int) -> set[str]:
    """Budget selection: the k facts with the highest positive values (fewer if fewer
    are positive). Ties are broken by the mapping's order, as in `policies.Budget`."""
    order = {a: i for i, a in enumerate(r)}
    ranked = sorted((a for a in r if r[a] > 0), key=lambda a: (-r[a], order[a]))
    return set(ranked[:k])


def above_threshold(r: Mapping[str, float], lam: float) -> set[str]:
    """Threshold selection: every fact whose value exceeds the price lam of a question."""
    return {a for a in r if r[a] > lam}


def separable_gain(r: Mapping[str, float], q: Iterable[str]) -> float:
    """Expected corrected answers from asking q under separability."""
    return sum(r[a] for a in q)


def union_bound(p: Mapping[str, float], unasked: Iterable[str]) -> float:
    """Proposition 1(b): P(answer wrong) <= sum of P_a over unasked dependencies."""
    return sum(p[a] for a in unasked)


def independent_bound(p: Mapping[str, float], unasked: Iterable[str]) -> float:
    """Proposition 1(b) under independent staleness: 1 - prod(1 - P_a)."""
    out = 1.0
    for a in unasked:
        out *= 1.0 - p[a]
    return 1.0 - out
