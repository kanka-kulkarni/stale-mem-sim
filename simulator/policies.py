"""Re-ask policies. Interface: `decide(memory, task, t) -> list[str]`.

Each policy has a `name` (one setting, e.g. "ttl(3)") and a `family` (the curve it
belongs to). Families that differ in impact version are distinct: "budget[exact]" and
"budget[coarse]" are separate curves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from simulator.core import (Domain, Memory, Policy, Task, answer_changing_stale,
                            effective_rate, generate_timeline, impact_coarse, impact_exact,
                            p_changed)

MIXTURE_STREAM = 0x31  # tag separating mixture coins from timeline and event streams


def is_stateful(policy: Policy) -> bool:
    """A policy's `stateful` flag (False if it doesn't declare one)."""
    return bool(getattr(policy, "stateful", False))


class Never:
    """Never re-ask anything."""

    name = family = "never"
    stateful = False

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Return no attributes."""
        return []


class Always:
    """Re-ask every attribute the task depends on."""

    name = family = "always"
    stateful = False

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Return all of the task's declared dependencies."""
        return list(task.depends_on)


def family_name(base: str, impact: str | None = None) -> str:
    """Family label: 'budget' + 'exact' -> 'budget[exact]'; no impact -> 'budget'."""
    return f"{base}[{impact}]" if impact else base


@dataclass(frozen=True)
class Family:
    """A parametric policy family that can be tuned to a target question cost."""

    name: str
    kind: str
    make: Callable[[float], Policy]
    values: tuple[int, ...] = ()
    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.kind not in ("integer", "continuous"):
            raise ValueError(f"unknown family kind {self.kind!r}")
        if self.kind == "integer" and len(self.values) < 1:
            raise ValueError("integer family needs values")


class Mixture:
    """Each session, follow policy `a` with probability `w`, else `b`, using a coin
    stream separate from the timelines."""

    role = "sweep"
    stateful = True

    def __init__(self, a: Policy, b: Policy, w: float, name: str | None = None) -> None:
        for p in (a, b):
            if is_stateful(p):
                raise ValueError(f"cannot mix stateful policy {p.name!r}")
        fa, fb = getattr(a, "family", a.name), getattr(b, "family", b.name)
        if fa != fb:
            raise ValueError(f"mixture must stay within one family, got {fa!r} and {fb!r}")
        if not 0.0 <= w <= 1.0:
            raise ValueError("w must be in [0, 1]")
        self.a, self.b, self.w, self.family = a, b, w, fa
        self.name = name or f"mix({a.name},{b.name},w={w:.4f})"
        self._rng: np.random.Generator | None = None
        self._last_t = 0

    def start_user(self, seed: int, user: int) -> None:
        """Seed this user's coin stream and forward the hook to both components."""
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, user, MIXTURE_STREAM]))
        self._last_t = 0
        for p in (self.a, self.b):
            if hasattr(p, "start_user"):
                p.start_user(seed, user)

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Draw this session's coin and delegate to the chosen component."""
        if self._rng is None or t != self._last_t + 1:
            raise RuntimeError("Mixture needs start_user() and sessions in order 1..T")
        self._last_t = t
        chosen = self.a if self._rng.random() < self.w else self.b
        return chosen.decide(memory, task, t)


class Matched:
    """A policy tuned to a target question cost; logged with role 'matched'."""

    role = "matched"

    def __init__(self, inner: Policy, family: str, target_cost: float) -> None:
        self.inner, self.family, self.target_cost = inner, family, target_cost
        self.name = f"{family}@{target_cost:g}"
        self.stateful = is_stateful(inner)

    def start_user(self, seed: int, user: int) -> None:
        """Forward the per-user hook."""
        if hasattr(self.inner, "start_user"):
            self.inner.start_user(seed, user)

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Delegate to the tuned policy."""
        return self.inner.decide(memory, task, t)


ImpactTable = dict[tuple[str, str], float]  # (task name, attribute) -> impact in [0, 1]
IMPACT_KINDS = ("exact", "coarse", "llm")


class _RatePolicy:
    """Base for policies that use per-attribute change rates at an assumed rate
    multiplier (and optionally another domain's rates)."""

    stateful = False

    def __init__(self, domain: Domain, rate_multiplier: float = 1.0,
                 impact_table: ImpactTable | None = None,
                 rate_domain: Domain | None = None) -> None:
        self.domain = domain
        src = rate_domain or domain
        self.rates = {a.name: effective_rate(src, a.name, rate_multiplier)
                      for a in domain.attributes}
        self.order = {a.name: i for i, a in enumerate(domain.attributes)}
        self.all_attributes = tuple(a.name for a in domain.attributes)
        self.impact_table = impact_table

    def p_changed(self, memory: Memory, attr: str, t: int) -> float:
        """P(attr changed since last verified), from its assumed rate and age."""
        return p_changed(self.rates[attr], memory.age(attr, t))

    def impact(self, kind: str, task: Task, attr: str, memory: Memory) -> float:
        """Exact (decision rule + memory values), coarse (dependency list), or llm
        (a model's per-(task, attribute) rating, normalised to [0, 1])."""
        if kind == "exact":
            return impact_exact(self.domain, task, attr, memory.values)
        if kind == "coarse":
            return impact_coarse(task, attr)
        if kind == "llm":
            if self.impact_table is None:
                raise ValueError("llm impact needs an impact_table")
            return self.impact_table[(task.name, attr)]
        raise ValueError(f"unknown impact kind {kind!r}")

    def candidates(self, kind: str, task: Task) -> tuple[str, ...]:
        """Attributes that can get a nonzero score. Exact and coarse impact are 0
        outside the dependency list; llm ratings may cover any attribute."""
        return self.all_attributes if kind == "llm" else task.depends_on

    def scores(self, kind: str, memory: Memory, task: Task, t: int) -> dict[str, float]:
        """P(changed) x impact for every candidate attribute."""
        return {a: self.p_changed(memory, a, t) * self.impact(kind, task, a, memory)
                for a in self.candidates(kind, task)}


class TTL(_RatePolicy):
    """Re-ask a dependency older than N sessions; `blind=True` considers every stored
    fact regardless of the task."""

    def __init__(self, domain: Domain, n: float, rate_multiplier: float = 1.0,
                 blind: bool = False) -> None:
        super().__init__(domain, rate_multiplier)
        self.n, self.blind = int(n), blind
        self.family = family_name("ttl", "blind" if blind else None)
        self.name = f"ttl({self.n},blind)" if blind else f"ttl({self.n})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Attributes in scope with age > N."""
        scope = self.all_attributes if self.blind else task.depends_on
        return [a for a in scope if memory.age(a, t) > self.n]


class Volatility(_RatePolicy):
    """Re-ask fast attributes every session and slow ones every K sessions (age >= K);
    static never. Task-aware by default; `blind=True` considers every stored attribute."""

    def __init__(self, domain: Domain, k: float, rate_multiplier: float = 1.0,
                 blind: bool = False) -> None:
        super().__init__(domain, rate_multiplier)
        self.k, self.blind = int(k), blind
        self.family = family_name("volatility", "blind" if blind else None)
        self.name = f"volatility({self.k},blind)" if blind else f"volatility({self.k})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Fast attributes in scope always; slow ones once they are K sessions old."""
        scope = self.all_attributes if self.blind else task.depends_on
        out = []
        for a in scope:
            vol = self.domain.attribute(a).volatility
            if vol == "fast" or (vol == "slow" and memory.age(a, t) >= self.k):
                out.append(a)
        return out


class StalenessXImpact(_RatePolicy):
    """Re-ask attribute a if P(changed) x impact > threshold."""

    def __init__(self, domain: Domain, threshold: float, impact: str,
                 rate_multiplier: float = 1.0, impact_table: ImpactTable | None = None,
                 rate_domain: Domain | None = None) -> None:
        super().__init__(domain, rate_multiplier, impact_table, rate_domain)
        self.threshold, self.impact_kind = float(threshold), impact
        self.family = family_name("sxi", impact)
        self.name = f"sxi({self.threshold:.6g},{impact})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Attributes whose score exceeds the threshold."""
        return [a for a, s in self.scores(self.impact_kind, memory, task, t).items()
                if s > self.threshold]


class Budget(_RatePolicy):
    """Re-ask the top-k attributes by P(changed) x impact; fewer if scores are zero.
    Ties are broken by attribute declaration order."""

    def __init__(self, domain: Domain, k: float, impact: str,
                 rate_multiplier: float = 1.0, impact_table: ImpactTable | None = None,
                 rate_domain: Domain | None = None) -> None:
        super().__init__(domain, rate_multiplier, impact_table, rate_domain)
        self.k, self.impact_kind = int(k), impact
        self.family = family_name("budget", impact)
        self.name = f"budget({self.k},{impact})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Up to k attributes with the highest positive scores."""
        scored = [(s, a) for a, s in self.scores(self.impact_kind, memory, task, t).items()
                  if s > 0]
        scored.sort(key=lambda sa: (-sa[0], self.order[sa[1]]))
        return [a for _, a in scored[: self.k]]


class InfoGain(_RatePolicy):
    """Information-gain-style baseline: re-ask any stored fact whose P(changed) exceeds
    the threshold, ignoring the task (uncertainty reduction without task impact)."""

    family = "info_gain"

    def __init__(self, domain: Domain, threshold: float, rate_multiplier: float = 1.0,
                 rate_domain: Domain | None = None) -> None:
        super().__init__(domain, rate_multiplier, rate_domain=rate_domain)
        self.threshold = float(threshold)
        self.name = f"info_gain({self.threshold:.6g})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Every stored fact with P(changed) above the threshold."""
        return [a for a in self.all_attributes if self.p_changed(memory, a, t) > self.threshold]


class AlwaysAll:
    """Reference point: re-ask every stored fact, relevant or not (memory = truth)."""

    name = family = "always_all"
    stateful = False

    def __init__(self, domain: Domain) -> None:
        self.attributes = [a.name for a in domain.attributes]

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """All stored facts."""
        return list(self.attributes)


class Oracle:
    """Perfect-information bound, not deployable: re-asks exactly the stale,
    answer-changing facts."""

    name = family = "oracle"
    stateful = True  # holds the current user's timeline; never mixed

    def __init__(self, domain: Domain, n_sessions: int, rate_multiplier: float = 1.0) -> None:
        self.domain, self.n_sessions, self.mult = domain, n_sessions, rate_multiplier
        self._timeline = None

    def start_user(self, seed: int, user: int) -> None:
        """Regenerate this user's true timeline (paired design: identical to the run's)."""
        self._timeline = generate_timeline(self.domain, seed, user, self.n_sessions, self.mult)

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Stale facts whose true value alone would change this task's answer."""
        if self._timeline is None or self._timeline.tasks[t] != task.name:
            raise RuntimeError("Oracle timeline does not match the served session")
        return answer_changing_stale(task, memory.values, self._timeline.states[t])


# Default sweep grids (the curves in the main figure).
TTL_GRID = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20)
VOL_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21)
SXI_GRID = (0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
BUDGET_GRID = (0, 1, 2, 3, 4)


def _impacts(impact_table: ImpactTable | None) -> tuple[str, ...]:
    return IMPACT_KINDS if impact_table is not None else ("exact", "coarse")


def build_families(domain: Domain, rate_multiplier: float = 1.0, max_sessions: int = 20,
                   impact_table: ImpactTable | None = None,
                   rate_domain: Domain | None = None,
                   include_info_gain: bool = False) -> dict[str, Family]:
    """Parametric families for matched-cost tuning, bound to a domain and an assumed
    rate multiplier."""
    m, tab, rd = rate_multiplier, impact_table, rate_domain
    n_attrs = len(domain.attributes)
    fams = []
    for blind in (False, True):
        tag = "blind" if blind else None
        fams.append(Family(family_name("ttl", tag), "integer",
                           lambda n, b=blind: TTL(domain, n, m, blind=b),
                           values=tuple(range(0, max_sessions + 1))))
        fams.append(Family(family_name("volatility", tag), "integer",
                           lambda k, b=blind: Volatility(domain, k, m, blind=b),
                           values=tuple(range(1, max_sessions + 2))))
    for impact in _impacts(tab):
        fams.append(Family(family_name("sxi", impact), "continuous",
                           lambda x, i=impact: StalenessXImpact(domain, x, i, m, tab, rd),
                           bounds=(0.0, 1.0)))
        fams.append(Family(family_name("budget", impact), "integer",
                           lambda k, i=impact: Budget(domain, k, i, m, tab, rd),
                           values=tuple(range(0, n_attrs + 1))))
    if include_info_gain:
        fams.append(Family("info_gain", "continuous",
                           lambda x: InfoGain(domain, x, m, rd), bounds=(0.0, 1.0)))
    return {f.name: f for f in fams}


def info_gain_specs() -> list[str]:
    """Sweep grid for the information-gain baseline (same thresholds as sxi)."""
    return [f"info_gain({x:g})" for x in SXI_GRID]


def sweep_specs(include_llm: bool = False) -> list[str]:
    """Policy specs for the default sweep: every family across its grid."""
    specs = []
    for suffix in ("", ",blind"):
        specs += [f"ttl({n}{suffix})" for n in TTL_GRID]
        specs += [f"volatility({k}{suffix})" for k in VOL_GRID]
    for impact in IMPACT_KINDS if include_llm else ("exact", "coarse"):
        specs += [f"sxi({x:g},{impact})" for x in SXI_GRID]
        specs += [f"budget({k},{impact})" for k in BUDGET_GRID]
    return specs


def make_policy(spec: str, domain: Domain | None = None, rate_multiplier: float = 1.0,
                impact_table: ImpactTable | None = None,
                rate_domain: Domain | None = None, n_sessions: int | None = None,
                true_multiplier: float | None = None) -> Policy:
    """Build a policy from a spec such as 'ttl(3)', 'sxi(0.05,coarse)' or
    'budget(2,exact)'."""
    if spec == "never":
        return Never()
    if spec == "always":
        return Always()
    if spec == "always_all" and domain is not None:
        return AlwaysAll(domain)
    if spec == "oracle" and domain is not None:
        if n_sessions is None:
            raise KeyError("oracle needs n_sessions (and the true rate multiplier)")
        return Oracle(domain, n_sessions,
                      rate_multiplier if true_multiplier is None else true_multiplier)
    m = re.fullmatch(r"(\w+)\(([^,()]+)(?:,(exact|coarse|llm|blind))?\)", spec)
    if m is None or domain is None:
        raise KeyError(f"unknown policy {spec!r} (parametric policies need a domain)")
    base, arg, var = m.group(1), float(m.group(2)), m.group(3)
    if base in ("ttl", "volatility") and var in (None, "blind"):
        cls = TTL if base == "ttl" else Volatility
        return cls(domain, arg, rate_multiplier, blind=var == "blind")
    if base == "info_gain" and var is None:
        return InfoGain(domain, arg, rate_multiplier, rate_domain)
    if base in ("sxi", "budget") and var in IMPACT_KINDS:
        cls = StalenessXImpact if base == "sxi" else Budget
        return cls(domain, arg, var, rate_multiplier, impact_table, rate_domain)
    raise KeyError(f"unknown policy {spec!r}")
