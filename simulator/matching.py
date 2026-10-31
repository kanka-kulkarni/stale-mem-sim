"""Tune real policies to a target question cost, for paired tests at matched cost.

All tuning runs on tuning timelines (separate seeds); the resulting `Calibration` is
then rebuilt and evaluated on held-out timelines. Question cost is always measured
relative to `always` on the same timelines.

- Integer families: evaluate every value; if none is within `tol` of the target, mix
  the two neighbouring values that bracket it and bisect the mixture weight.
- Continuous families: bisect the parameter; if the cost is a step function and no
  explored value is within `tol`, mix the two closest bracketing values the same way.

Stateful families (see `policies.is_stateful`) are never mixed: calibration raises if
a mixture would be needed. A stateful policy such as the per-session LLM policy is instead used
as the reference, and other families are tuned to its realized cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from simulator import core
from simulator.core import Domain, Policy, Timeline
from simulator.policies import Always, Family, Matched, Mixture, is_stateful


def mean_questions(domain: Domain, timelines: Sequence[Timeline], policy: Policy) -> float:
    """Mean questions per session of `policy` on `timelines`."""
    return float(np.mean([r["n_questions"] for r in core.run(domain, timelines, [policy])]))


def question_cost(domain: Domain, timelines: Sequence[Timeline], policy: Policy,
                  always_q: float) -> float:
    """Mean questions relative to `always` on the same timelines."""
    return mean_questions(domain, timelines, policy) / always_q


@dataclass(frozen=True)
class Calibration:
    """Outcome of tuning one family to one target cost: a single setting, a mixture of
    two, or unreachable."""

    family: str
    target_cost: float
    kind: str
    param_a: float | None = None
    param_b: float | None = None
    weight: float | None = None
    tuned_cost: float | None = None

    def build(self, fam: Family) -> Matched | None:
        """Rebuild the tuned policy (fresh state) for evaluation; None if unreachable."""
        if self.kind == "unreachable":
            return None
        if self.kind == "single":
            inner: Policy = fam.make(self.param_a)
        else:
            inner = Mixture(fam.make(self.param_a), fam.make(self.param_b), self.weight)
        return Matched(inner, fam.name, self.target_cost)

    def to_dict(self) -> dict:
        """JSON-serialisable record for the run log."""
        return asdict(self)


def _explore(domain: Domain, tls: Sequence[Timeline], fam: Family, target: float,
             always_q: float, iters: int) -> list[tuple[float, float]]:
    """(param, cost) points: every value for integer families, a bisection trail for
    continuous ones. Sorted by param."""
    cost = lambda p: question_cost(domain, tls, fam.make(p), always_q)  # noqa: E731
    if fam.kind == "integer":
        return sorted((float(v), cost(v)) for v in fam.values)
    lo, hi = fam.bounds
    pts = {lo: cost(lo), hi: cost(hi)}
    increasing = pts[hi] >= pts[lo]
    for _ in range(iters):
        mid = (lo + hi) / 2
        pts[mid] = c = cost(mid)
        if (c < target) == increasing:
            lo = mid
        else:
            hi = mid
    return sorted(pts.items())


def calibrate(domain: Domain, tuning: Sequence[Timeline], fam: Family, target: float,
              tol: float = 0.01, iters: int = 20, weight_iters: int = 12) -> Calibration:
    """Find a policy in `fam` whose question cost on `tuning` is within `tol` of `target`,
    using a single setting if possible, else a mixture of two neighbouring settings."""
    always_q = mean_questions(domain, tuning, Always())
    pts = _explore(domain, tuning, fam, target, always_q, iters)
    best_p, best_c = min(pts, key=lambda pc: abs(pc[1] - target))
    if abs(best_c - target) <= tol:
        return Calibration(fam.name, target, "single", param_a=best_p, tuned_cost=best_c)
    bracket = next(((a, b) for a, b in zip(pts, pts[1:])
                    if min(a[1], b[1]) <= target <= max(a[1], b[1])), None)
    if bracket is None:
        return Calibration(fam.name, target, "unreachable")
    (pa, ca), (pb, cb) = bracket
    if is_stateful(fam.make(pa)):
        raise ValueError(
            f"family {fam.name!r} is stateful and no single setting is within tol of "
            f"{target}; stateful policies are never mixed (match other families to it)")
    # weight w = P(use pa). Expected cost runs from cb (w=0) to ca (w=1).
    lo, hi = 0.0, 1.0
    best = (abs(cb - target), 0.0, cb)
    for _ in range(weight_iters):
        w = (lo + hi) / 2
        c = question_cost(domain, tuning, Mixture(fam.make(pa), fam.make(pb), w), always_q)
        best = min(best, (abs(c - target), w, c))
        if abs(c - target) <= tol:
            break
        if (c < target) == (ca > cb):
            lo = w
        else:
            hi = w
    _, w, c = best
    return Calibration(fam.name, target, "mixture", param_a=pa, param_b=pb, weight=w,
                       tuned_cost=c)


def check_disjoint(tune_seeds: Sequence[int], eval_seeds: Sequence[int]) -> None:
    """Raise if any seed is used for both tuning and evaluation."""
    overlap = set(tune_seeds) & set(eval_seeds)
    if overlap:
        raise ValueError(f"tuning and evaluation seeds overlap: {sorted(overlap)}")
