"""Core simulator: attributes, life events, tasks, timelines, memory, session loop.

No language model is used anywhere in this module. The simulated user and the
answerer are deterministic, so ground truth is exact.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

DEFAULT_RATES: dict[str, float] = {"static": 0.0, "slow": 0.03, "fast": 0.15}

State = dict[str, str]
EVENT_STREAM = 0xE7  # tag separating life-event randomness from the user stream
ANSWER_STREAM = 0xA5  # wrong user answers (realism study S2)
_WRONG_ANSWER_RATE = 0.0  # set only through `wrong_answers(e)`; 0 = truthful user


@dataclass(frozen=True)
class Attribute:
    """A user fact with a fixed value set and a per-session change probability."""

    name: str
    values: tuple[str, ...]
    volatility: str
    rate: float | None = None  # None -> DEFAULT_RATES[volatility]

    def __post_init__(self) -> None:
        if self.volatility not in DEFAULT_RATES:
            raise ValueError(f"unknown volatility class {self.volatility!r}")
        if len(self.values) < 2:
            raise ValueError(f"attribute {self.name!r} needs at least two values")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"attribute {self.name!r} has duplicate values")

    @property
    def base_rate(self) -> float:
        """Per-session independent change probability before any multiplier."""
        return DEFAULT_RATES[self.volatility] if self.rate is None else self.rate


@dataclass(frozen=True)
class LifeEvent:
    """An event that changes several attributes together; `apply(state, rng)` returns
    the new values."""

    name: str
    prob: float
    touches: tuple[str, ...]
    apply: Callable[[State, np.random.Generator], State]


@dataclass(frozen=True)
class Task:
    """A request type with its declared dependencies and a deterministic answer rule."""

    name: str
    depends_on: tuple[str, ...]
    options: tuple[str, ...]
    rule: Callable[[State], str]
    description: str = ""  # the user's request in plain words (shown to LLMs only)

    def answer(self, values: State) -> str:
        """Return the option chosen by the decision rule for the given values."""
        out = self.rule(values)
        if out not in self.options:
            raise ValueError(f"task {self.name!r} returned unknown option {out!r}")
        return out


@dataclass(frozen=True)
class Domain:
    """A set of attributes, tasks and (optionally) life events."""

    name: str
    attributes: tuple[Attribute, ...]
    tasks: tuple[Task, ...]
    events: tuple[LifeEvent, ...] = ()

    def __post_init__(self) -> None:
        names = {a.name for a in self.attributes}
        for task in self.tasks:
            missing = set(task.depends_on) - names
            if missing:
                raise ValueError(f"task {task.name!r} depends on unknown {missing}")
        for ev in self.events:
            missing = set(ev.touches) - names
            if missing:
                raise ValueError(f"event {ev.name!r} touches unknown {missing}")

    def attribute(self, name: str) -> Attribute:
        """Look up an attribute by name."""
        for a in self.attributes:
            if a.name == name:
                return a
        raise KeyError(name)

    def task(self, name: str) -> Task:
        """Look up a task by name."""
        for k in self.tasks:
            if k.name == name:
                return k
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Rates and P(changed)
# ---------------------------------------------------------------------------


def scaled(p: float, multiplier: float) -> float:
    """Scale a probability by a multiplier, clipped to [0, 1]."""
    return float(min(1.0, max(0.0, p * multiplier)))


def effective_rate(domain: Domain, attr: str, multiplier: float = 1.0) -> float:
    """p_a: the attribute's own rate plus summed probabilities of events touching it."""
    p = scaled(domain.attribute(attr).base_rate, multiplier)
    p += sum(scaled(ev.prob, multiplier) for ev in domain.events if attr in ev.touches)
    return min(1.0, p)


def p_changed(p_a: float, age: int) -> float:
    """Probability an attribute changed at least once in `age` sessions: 1-(1-p)^d."""
    if age < 0:
        raise ValueError("age must be non-negative")
    return 1.0 - (1.0 - p_a) ** age


# ---------------------------------------------------------------------------
# Timelines (paired design)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Timeline:
    """One user's pre-generated ground truth: states[t] for t=0..T and tasks[t] for
    t=1..T (session 0 is onboarding)."""

    user: int
    seed: int
    states: tuple[State, ...]
    tasks: tuple[str | None, ...]

    @property
    def n_sessions(self) -> int:
        """Number of task sessions T (excluding onboarding)."""
        return len(self.states) - 1


def user_rng(seed: int, user: int) -> np.random.Generator:
    """Independent, reproducible random stream for one (seed, user) pair."""
    return np.random.default_rng(np.random.SeedSequence([seed, user]))


def generate_timeline(
    domain: Domain, seed: int, user: int, n_sessions: int, rate_multiplier: float = 1.0,
    own_rates: dict[str, float] | None = None,
) -> Timeline:
    """Generate a user's full timeline upfront; it never depends on any policy, and
    timelines at different rate multipliers share their random numbers."""
    rng = user_rng(seed, user)
    state: State = {a.name: a.values[int(rng.integers(len(a.values)))] for a in domain.attributes}
    states = [dict(state)]
    tasks: list[str | None] = [None]
    for t in range(1, n_sessions + 1):
        state = _drift(domain, state, rng, rate_multiplier, (seed, user, t), own_rates)
        states.append(dict(state))
        tasks.append(domain.tasks[int(rng.integers(len(domain.tasks)))].name)
    return Timeline(user=user, seed=seed, states=tuple(states), tasks=tuple(tasks))


def _drift(
    domain: Domain, state: State, rng: np.random.Generator, mult: float,
    key: tuple[int, int, int], own_rates: dict[str, float] | None = None,
) -> State:
    """One session of independent drift followed by life events."""
    new = dict(state)
    for a in domain.attributes:
        u = rng.random()
        pick = int(rng.integers(len(a.values) - 1))  # drawn always, for stream alignment
        rate = a.base_rate if own_rates is None else own_rates[a.name]
        if u < scaled(rate, mult):
            others = [v for v in a.values if v != new[a.name]]
            new[a.name] = others[pick]
    for i, ev in enumerate(domain.events):
        u = rng.random()
        if u < scaled(ev.prob, mult):
            ev_rng = np.random.default_rng(np.random.SeedSequence([*key, i, EVENT_STREAM]))
            new.update(ev.apply(dict(new), ev_rng))
    return new


def generate_timelines(
    domain: Domain, seed: int, n_users: int, n_sessions: int, rate_multiplier: float = 1.0
) -> list[Timeline]:
    """Generate timelines for users 0..n_users-1 under one seed."""
    return [
        generate_timeline(domain, seed, u, n_sessions, rate_multiplier) for u in range(n_users)
    ]


def state_hash(state: State) -> str:
    """Short stable hash of a state, logged so paired runs can be checked from CSVs."""
    blob = "|".join(f"{k}={state[k]}" for k in sorted(state))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class Memory:
    """The assistant's stored value and last-verified session for each attribute."""

    values: State
    verified_at: dict[str, int] = field(default_factory=dict)

    @classmethod
    def onboard(cls, true_state: State) -> "Memory":
        """Session-0 memory: equals the true state, everything verified at 0."""
        return cls(values=dict(true_state), verified_at={k: 0 for k in true_state})

    def age(self, attr: str, t: int) -> int:
        """Sessions since `attr` was last verified."""
        return t - self.verified_at[attr]

    def update(self, attr: str, value: str, t: int) -> None:
        """Store a freshly verified value."""
        self.values[attr] = value
        self.verified_at[attr] = t


# ---------------------------------------------------------------------------
# Impact
# ---------------------------------------------------------------------------


def impact_exact(domain: Domain, task: Task, attr: str, memory_values: State) -> float:
    """Fraction of alternative values of `attr` that change the task's answer,
    holding all other attributes at their memory values. Never reads true state."""
    base = task.answer(memory_values)
    alts = [v for v in domain.attribute(attr).values if v != memory_values[attr]]
    flips = sum(task.answer({**memory_values, attr: v}) != base for v in alts)
    return flips / len(alts)


def impact_coarse(task: Task, attr: str) -> float:
    """1 if `attr` is in the task's declared dependency list, else 0."""
    return 1.0 if attr in task.depends_on else 0.0


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


def answer_changing_stale(task: Task, memory_values: State, true_state: State) -> list[str]:
    """Stale dependencies whose true value alone would change the task's answer,
    holding the other attributes at their memory values (scoring only)."""
    base = task.answer(memory_values)
    return [
        a for a in task.depends_on
        if memory_values[a] != true_state[a]
        and task.answer({**memory_values, a: true_state[a]}) != base
    ]


class Policy(Protocol):
    """Common policy interface: `decide(memory, task, t)` returns the facts to re-ask."""

    name: str

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Return the attributes to re-ask before answering `task` at session t."""
        ...


@contextmanager
def wrong_answers(rate: float):
    """Realism study S2: within this block a re-asked user gives a random other value
    with probability `rate`, identically under every policy."""
    global _WRONG_ANSWER_RATE
    old, _WRONG_ANSWER_RATE = _WRONG_ANSWER_RATE, float(rate)
    try:
        yield
    finally:
        _WRONG_ANSWER_RATE = old


def _answer_table(domain: Domain, timeline: Timeline) -> tuple[np.ndarray, np.ndarray]:
    """Per-user uniforms and replacement indices, shape (sessions + 1, facts)."""
    rng = np.random.default_rng(np.random.SeedSequence([timeline.seed, timeline.user,
                                                        ANSWER_STREAM]))
    shape = (timeline.n_sessions + 1, len(domain.attributes))
    return rng.random(shape), rng.integers(0, 2**31 - 1, size=shape)


def _reported(domain: Domain, attr: str, truth: str, t: int, noise) -> str:
    """What the user says when re-asked: the truth, or (S2) a random other value."""
    if noise is None:
        return truth
    i = next(j for j, a in enumerate(domain.attributes) if a.name == attr)
    u, pick = noise[0][t, i], noise[1][t, i]
    if u >= _WRONG_ANSWER_RATE:
        return truth
    others = [v for v in domain.attributes[i].values if v != truth]
    return others[int(pick) % len(others)]


def run_user(domain: Domain, timeline: Timeline, policy: Policy,
             record_memory: bool = False) -> list[dict]:
    """Replay one user's timeline under one policy; return one record per session
    (with memory and true state if `record_memory`)."""
    memory = Memory.onboard(timeline.states[0])
    noise = _answer_table(domain, timeline) if _WRONG_ANSWER_RATE > 0 else None
    if hasattr(policy, "start_user"):
        policy.start_user(timeline.seed, timeline.user)
    family = getattr(policy, "family", policy.name)
    role = getattr(policy, "role", "sweep")
    target_cost = getattr(policy, "target_cost", float("nan"))
    records = []
    for t in range(1, timeline.n_sessions + 1):
        true_state = timeline.states[t]  # step 1: memory is not told
        task = domain.task(timeline.tasks[t])  # step 2
        pre = dict(memory.values)  # pre-ask snapshot, used only for scoring
        asked = list(dict.fromkeys(policy.decide(memory, task, t)))  # step 3, dedup
        for a in asked:  # step 4: simulated user answers (truthfully unless S2 noise)
            memory.update(a, _reported(domain, a, true_state[a], t, noise), t)
        given = task.answer(memory.values)  # step 5
        truth = task.answer(true_state)
        stale_deps = [a for a in task.depends_on if memory.values[a] != true_state[a]]
        changing = answer_changing_stale(task, pre, true_state)
        pre_ask_correct = task.answer(pre) == truth
        correct = given == truth
        records.append(
            {
                "domain": domain.name,
                "seed": timeline.seed,
                "user": timeline.user,
                "session": t,
                "policy": policy.name,
                "family": family,
                "role": role,
                "target_cost": target_cost,
                "task": task.name,
                "state_hash": state_hash(true_state),
                "asked": ";".join(asked),
                "n_questions": len(asked),
                "n_asked_stale": sum(pre[a] != true_state[a] for a in asked),
                "n_answer_changing": len(changing),
                "n_answer_changing_asked": sum(a in changing for a in asked),
                "pre_ask_correct": pre_ask_correct,
                "answer": given,
                "true_answer": truth,
                "correct": correct,
                "n_stale_deps": len(stale_deps),
                "stale_error": (not correct) and bool(stale_deps),
            }
        )
        if record_memory:
            records[-1]["memory"] = dict(memory.values)
            records[-1]["true_state"] = dict(true_state)
    return records


def run(domain: Domain, timelines: Sequence[Timeline], policies: Sequence[Policy],
        record_memory: bool = False) -> list[dict]:
    """Run every policy on every timeline (paired: identical timelines per policy)."""
    out: list[dict] = []
    for policy in policies:
        for tl in timelines:
            out.extend(run_user(domain, tl, policy, record_memory))
    return out
