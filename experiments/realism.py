"""Realism sensitivity analyses. No LLM calls.

S1: people differ in how often their facts change (per-user gamma rates, CV grid), with
    a policy that learns each user's rates from re-ask outcomes and an oracle-rates
    reference.
S2: users sometimes answer wrongly.
S3: strength of correlated changes on meals.

Example:
    python -m experiments.realism s1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from simulator import core, metrics, stats
from simulator.core import Attribute, Domain, LifeEvent, Memory, Task, Timeline
from simulator.domains import get_domain
from simulator.matching import _explore, calibrate, mean_questions
from simulator.policies import (Always, Family, InfoGain, Matched, Never, StalenessXImpact,
                                TTL, family_name)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "realism"
TUNE_SEEDS, EVAL_SEEDS = (1000, 1001, 1002, 1003, 1004), (0, 1, 2, 3, 4)
USERS, SESSIONS, TARGETS, TOL = 50, 20, (0.25, 0.5), 0.01
RATE_STREAM = 0x52  # per-user individual rates (S1)
CVS = (0.0, 0.5, 1.0, 2.0)
PRIOR_STRENGTH = 5.0
DOMAINS = ("commute", "meals_calibrated")
MATCHED = ("ttl", "sxi[exact]", "sxi[coarse]", "info_gain", "oracle_rates", "learned_rates")


# --------------------------------------------------------------------------- environment


def individual_rates(domain: Domain, seed: int, user: int, cv: float) -> dict[str, float]:
    """One user's own drift rate per fact: gamma with the population mean and coefficient
    of variation cv, capped at 1."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, user, RATE_STREAM]))
    out = {}
    for a in domain.attributes:
        mu = a.base_rate
        draw = rng.gamma(1.0 / cv**2, mu * cv**2) if cv > 0 and mu > 0 else mu
        if cv > 0 and mu == 0:
            rng.gamma(1.0, 1.0)  # keep one draw per fact for stream alignment
        out[a.name] = float(min(1.0, draw))
    return out


def event_rate(domain: Domain, attr: str) -> float:
    """Summed probability of the life events that touch a fact."""
    return sum(ev.prob for ev in domain.events if attr in ev.touches)


@dataclass
class Env:
    domain: Domain
    cv: float
    tuning: list[Timeline]
    held: list[Timeline]
    rates: dict  # (seed, user) -> individual own rates


def make_env(name: str, cv: float, domain: Domain | None = None) -> Env:
    """Tuning and held-out timelines with individual rates drawn at this cv."""
    d = domain or get_domain(name)
    rates, tl = {}, {}
    for seeds, key in ((TUNE_SEEDS, "tuning"), (EVAL_SEEDS, "held")):
        out = []
        for s in seeds:
            for u in range(USERS):
                r = individual_rates(d, s, u, cv)
                rates[(s, u)] = r
                out.append(core.generate_timeline(d, s, u, SESSIONS, own_rates=r))
        tl[key] = out
    return Env(d, cv, tl["tuning"], tl["held"], rates)


# --------------------------------------------------------------------------- policies


class OracleRates(StalenessXImpact):
    """Staleness x impact (dependency list) with the user's true individual rates; not
    deployable."""

    stateful = False

    def __init__(self, domain: Domain, threshold: float, rates: dict) -> None:
        super().__init__(domain, threshold, "coarse")
        self.user_rates, self.family = rates, "oracle_rates"
        self.name = f"oracle_rates({threshold:.6g})"

    def start_user(self, seed: int, user: int) -> None:
        """Load this user's true individual rates."""
        own = self.user_rates[(seed, user)]
        self.rates = {a: min(1.0, own[a] + event_rate(self.domain, a)) for a in own}


class LearnedRates(StalenessXImpact):
    """Staleness x impact (dependency list) with per-user rates learned online from re-ask
    outcomes (Beta prior at the population rate). Stateful."""

    stateful = True

    def __init__(self, domain: Domain, threshold: float, strength: float = PRIOR_STRENGTH) -> None:
        super().__init__(domain, threshold, "coarse")
        self.population = dict(self.rates)
        self.strength, self.family = strength, "learned_rates"
        self.name = f"learned_rates({threshold:.6g})"
        self._ab: dict[str, list[float]] = {}
        self._snap: tuple[int, dict, dict] | None = None

    def start_user(self, seed: int, user: int) -> None:
        """Reset the posterior to the population prior."""
        self._ab = {a: [self.strength * p, self.strength * (1 - p)]
                    for a, p in self.population.items()}
        self._snap = None
        self.rates = dict(self.population)

    def _learn(self, memory: Memory) -> None:
        """Update from the facts re-asked at the previous session."""
        if self._snap is None:
            return
        t_prev, values, verified = self._snap
        for a, tau in verified.items():
            if memory.verified_at[a] == t_prev and tau < t_prev:
                d = t_prev - tau
                if memory.values[a] != values[a]:
                    self._ab[a][0] += 1.0
                    self._ab[a][1] += d - 1
                else:
                    self._ab[a][1] += d
        self.rates = {a: ab[0] / (ab[0] + ab[1]) if ab[0] + ab[1] > 0 else 0.0
                      for a, ab in self._ab.items()}

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Learn from last session's answers, then ask by staleness x impact."""
        self._learn(memory)
        self._snap = (t, dict(memory.values), dict(memory.verified_at))
        return super().decide(memory, task, t)


class TimelineOracle:
    """The oracle, reading this condition's own timelines (individual rates included)."""

    name = family = "oracle"
    stateful = True

    def __init__(self, timelines: list[Timeline]) -> None:
        self.by_user = {(tl.seed, tl.user): tl for tl in timelines}
        self._tl = None

    def start_user(self, seed: int, user: int) -> None:
        self._tl = self.by_user[(seed, user)]

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        assert self._tl.tasks[t] == task.name
        return core.answer_changing_stale(task, memory.values, self._tl.states[t])


# --------------------------------------------------------------------------- tuning


def families(env: Env) -> dict[str, Family]:
    """Matched families of S1 (learned_rates is tuned separately, without mixtures)."""
    d = env.domain
    return {
        "ttl": Family("ttl", "integer", lambda n: TTL(d, n), values=tuple(range(SESSIONS + 1))),
        "sxi[exact]": Family(family_name("sxi", "exact"), "continuous",
                             lambda x: StalenessXImpact(d, x, "exact"), bounds=(0.0, 1.0)),
        "sxi[coarse]": Family(family_name("sxi", "coarse"), "continuous",
                              lambda x: StalenessXImpact(d, x, "coarse"), bounds=(0.0, 1.0)),
        "info_gain": Family("info_gain", "continuous", lambda x: InfoGain(d, x), bounds=(0.0, 1.0)),
        "oracle_rates": Family("oracle_rates", "continuous",
                               lambda x: OracleRates(d, x, env.rates), bounds=(0.0, 1.0)),
    }


def tune_threshold_only(env: Env, make: Callable[[float], object], target: float) -> tuple[float, float, list]:
    """Threshold search without mixtures (stateful policy): bisection trail on the
    tuning users, keep the setting whose cost is closest to the target."""
    fam = Family("learned_rates", "continuous", make, bounds=(0.0, 1.0))
    always_q = mean_questions(env.domain, env.tuning, Always())
    pts = _explore(env.domain, env.tuning, fam, target, always_q, 20)
    best_p, best_c = min(pts, key=lambda pc: abs(pc[1] - target))
    return best_p, best_c, pts


def run_condition(env: Env, keep: tuple = ("ttl", "sxi[exact]", "sxi[coarse]", "info_gain",
                                           "oracle_rates"),
                  learned: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Run one condition's matched families, references and bracketing sweep points;
    callers set the wrong-answer rate (S2)."""
    d, records, cals = env.domain, [], []
    refs = [Never(), Always(), TimelineOracle(env.tuning + env.held)]
    records += core.run(d, env.held, refs)
    for key, fam in families(env).items():
        if key not in keep:
            continue
        params = set()
        for target in TARGETS:
            cal = calibrate(d, env.tuning, fam, target, TOL)
            cals.append({**cal.to_dict(), "cv": env.cv})
            pol = cal.build(fam)
            if pol is None:
                continue
            records += core.run(d, env.held, [pol])
            params |= {p for p in (cal.param_a, cal.param_b) if p is not None}
        records += core.run(d, env.held, [fam.make(p) for p in sorted(params)])
    if not learned:
        return pd.DataFrame(records), cals
    make = lambda x: LearnedRates(d, x)  # noqa: E731
    sweep_params = set()
    for target in TARGETS:
        p, c, pts = tune_threshold_only(env, make, target)
        cals.append({"family": "learned_rates", "target_cost": target, "kind": "single",
                     "param_a": p, "tuned_cost": c, "cv": env.cv})
        records += core.run(d, env.held, [Matched(make(p), "learned_rates", target)])
        near = sorted(pts, key=lambda pc: abs(pc[1] - target))[1:3]  # neighbours for slope
        sweep_params |= {q for q, _ in near}
    records += core.run(d, env.held, [make(q) for q in sorted(sweep_params)])
    return pd.DataFrame(records), cals


def pairwise(summary: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """All pairs of matched policies (different families) per domain and cost, with
    exact McNemar and the cost check (Holm applied by the caller)."""
    comp = metrics.matched_comparisons(summary)
    return stats.add_mcnemar(comp, sessions)


# --------------------------------------------------------------------------- S1


def run_s1() -> None:
    """S1 for both domains and every CV."""
    summaries, comps = [], []
    for name in DOMAINS:
        for cv in CVS:
            env = make_env(name, cv)
            df, cals = run_condition(env)
            out = OUT / f"s1_{name}" / f"cv{cv:g}"
            out.mkdir(parents=True, exist_ok=True)
            s = metrics.summarize(df)
            s["cv"] = cv
            s.to_csv(out / "summary.csv", index=False)
            matched = df[df.role == "matched"]
            matched.to_csv(out / "sessions_matched.csv.gz", index=False)
            (out / "calibration.json").write_text(json.dumps(cals, indent=1))
            c = pairwise(s, matched)
            c["cv"] = cv
            summaries.append(s)
            comps.append(c)
            print(f"S1 {name} cv={cv:g}: {(s.role == 'matched').sum()} matched", flush=True)
    comp = pd.concat(comps, ignore_index=True)
    comp["p_holm"] = stats.holm(comp.p_mcnemar.tolist())
    comp.to_csv(OUT / "s1_comparisons.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(OUT / "s1_summary.csv", index=False)
    (OUT / "s1_config.json").write_text(json.dumps({
        "cvs": CVS, "prior_strength": PRIOR_STRENGTH, "rate_stream": RATE_STREAM,
        "targets": TARGETS, "tol": TOL, "tune_seeds": TUNE_SEEDS, "eval_seeds": EVAL_SEEDS,
        "users": USERS, "sessions": SESSIONS}, indent=1))
    print(f"S1: {len(comp)} comparisons, {(comp.p_holm < 0.05).sum()} significant (Holm)")


LINES = {  # family -> (label, colour, marker, linestyle); hues as in the main figures
    "sxi[exact]": ("Staleness×impact, exact (population rates)", "#1baf7a", "^", "-"),
    "sxi[coarse]": ("Staleness×impact, dep. list (population rates)", "#1baf7a", "^", (0, (4, 2))),
    "learned_rates": ("Learned per-user rates, dep. list", "#eb6834", "o", "-"),
    "oracle_rates": ("True per-user rates, dep. list (not deployable)", "#e34948", "s", (0, (1, 1.2))),
    "ttl": ("TTL, dep. list", "#2a78d6", "D", "-"),
    "info_gain": ("Information gain (task-blind)", "#6250d6", "v", (0, (3, 1, 1, 1, 1, 1))),
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def plot_s1() -> None:
    """S1 figure: accuracy at 25% and 50% cost against CV, one line per policy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.read_csv(OUT / "s1_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True)
    for i, dom in enumerate(DOMAINS):
        for j, t in enumerate(TARGETS):
            ax = axes[i, j]
            for fam, (label, colour, marker, ls) in LINES.items():
                g = s[(s.domain == dom) & (s.family == fam) & (s.target_cost == t)
                      & (s.role == "matched")].sort_values("cv")
                ax.plot(g.cv, g.accuracy, color=colour, marker=marker, linestyle=ls,
                        markersize=4.5, linewidth=1.5, label=label, zorder=3,
                        markerfacecolor="white" if fam == "sxi[coarse]" else colour)
            for ref, ls in (("never", ":"), ("oracle", "--")):
                g = s[(s.domain == dom) & (s.policy == ref)].sort_values("cv")
                ax.plot(g.cv, g.accuracy, color=INK_2, linestyle=ls, linewidth=1.0, zorder=2,
                        label="Never" if ref == "never" else "Oracle (perfect information, own cost)")
            name = "Commute" if dom == "commute" else "Meal planning (calibrated)"
            ax.set_title(f"{name}, {t:.0%} of Always's questions", fontsize=8, color=INK, loc="left")
            ax.set_xticks(list(CVS))
            ax.tick_params(labelsize=7, colors=INK_2)
            ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 1:
                ax.set_xlabel("spread of per-user change rates (CV)", fontsize=7.5, color=INK)
            if j == 0:
                ax.set_ylabel("accuracy at matched cost", fontsize=7.5, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.11))
    fig.tight_layout()
    fig_dir = ROOT / "paper" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"realism_s1.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- S2, S3

ERRORS = (0.0, 0.05, 0.1, 0.2)
S2_CV = 1.0
FACTORS = (0.0, 1.0, 2.0, 4.0)


def run_family_of_conditions(tag: str, conditions: list[tuple[str, float, Env]],
                             keep: tuple, learned: bool, error_of=lambda level: 0.0) -> None:
    """Run each (domain, level, env), write per-condition outputs and one Holm family."""
    summaries, comps = [], []
    for name, level, env in conditions:
        with core.wrong_answers(error_of(level)):
            df, cals = run_condition(env, keep, learned)
        out = OUT / f"{tag}_{name}" / f"level{level:g}"
        out.mkdir(parents=True, exist_ok=True)
        s = metrics.summarize(df)
        s["level"] = level
        s.to_csv(out / "summary.csv", index=False)
        matched = df[df.role == "matched"]
        matched.to_csv(out / "sessions_matched.csv.gz", index=False)
        (out / "calibration.json").write_text(json.dumps(cals, indent=1))
        c = pairwise(s, matched)
        c["level"] = level
        summaries.append(s)
        comps.append(c)
        print(f"{tag.upper()} {name} level={level:g}: {(s.role == 'matched').sum()} matched", flush=True)
    comp = pd.concat(comps, ignore_index=True)
    comp["p_holm"] = stats.holm(comp.p_mcnemar.tolist())
    comp.to_csv(OUT / f"{tag}_comparisons.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(OUT / f"{tag}_summary.csv", index=False)
    print(f"{tag.upper()}: {len(comp)} comparisons, {(comp.p_holm < 0.05).sum()} significant (Holm)")


def run_s2() -> None:
    """S2: wrong answers with probability e, all at CV = 1 (heterogeneous setting)."""
    conds = []
    for name in DOMAINS:
        env = make_env(name, S2_CV)
        conds += [(name, e, env) for e in ERRORS]
    run_family_of_conditions("s2", conds, ("ttl", "sxi[exact]", "sxi[coarse]", "info_gain"),
                             learned=True, error_of=lambda e: e)
    (OUT / "s2_config.json").write_text(json.dumps({"errors": ERRORS, "cv": S2_CV,
                                                    "answer_stream": core.ANSWER_STREAM}, indent=1))


ERRORS_ADDED = (0.01, 0.02, 0.15)  # extra S2 error rates, own Holm family


def run_s2_added() -> None:
    """S2 added grid points, same setting as S2 (CV = 1)."""
    conds = []
    for name in DOMAINS:
        env = make_env(name, S2_CV)
        conds += [(name, e, env) for e in ERRORS_ADDED]
    run_family_of_conditions("s2added", conds, ("ttl", "sxi[exact]", "sxi[coarse]", "info_gain"),
                             learned=True, error_of=lambda e: e)


S2_LINES = {  # as in the main figures; Always and never in grey
    "sxi[exact]": ("Staleness×impact, exact", "#1baf7a", "^", "-", True),
    "sxi[coarse]": ("Staleness×impact, dep. list", "#1baf7a", "^", (0, (4, 2)), False),
    "learned_rates": ("Learned per-user rates, dep. list", "#eb6834", "o", "-", True),
    "ttl": ("TTL, dep. list", "#2a78d6", "D", "-", True),
    "info_gain": ("Information gain (task-blind)", "#6250d6", "v", (0, (3, 1, 1, 1, 1, 1)), True),
}
TARGETED = ("ttl", "sxi[coarse]", "sxi[exact]", "learned_rates", "info_gain")


def s2_all() -> pd.DataFrame:
    """S2 summaries for every grid point."""
    a = pd.read_csv(OUT / "s2_summary.csv").assign(added=False)
    b = pd.read_csv(OUT / "s2added_summary.csv").assign(added=True)
    return pd.concat([a, b], ignore_index=True)


def plot_s2() -> None:
    """Accuracy at 50% cost against e, one line per policy, both domains."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = s2_all()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=False)
    for ax, dom in zip(axes, DOMAINS):
        d = s[s.domain == dom]
        for fam, (label, colour, marker, ls, filled) in S2_LINES.items():
            g = d[d.policy == f"{fam}@0.5"].sort_values("level")
            ax.plot(g.level, g.accuracy, color=colour, marker=marker, linestyle=ls, markersize=4,
                    linewidth=1.5, label=label, zorder=3,
                    markerfacecolor=colour if filled else "white")
        for ref, ls, mk, label in (("always", "-", "P", "Always (every dependency, cost 1)"),
                                   ("never", ":", None, "Never")):
            g = d[d.policy == ref].sort_values("level")
            ax.plot(g.level, g.accuracy, color=INK_2, linestyle=ls, marker=mk, markersize=4,
                    linewidth=1.3, label=label, zorder=2)
        added = d[d.added].level.unique()
        for e in added:
            ax.axvline(e, color=GRID, linewidth=0.8, zorder=0)
        ax.set_title("Commute" if dom == "commute" else "Meal planning (calibrated)",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("probability of a wrong answer e", fontsize=7.5, color=INK)
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
        ax.tick_params(labelsize=7, colors=INK_2)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("accuracy at 50% of Always's questions", fontsize=7.5, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "paper" / "figures" / f"realism_s2.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def s2_crossing() -> pd.DataFrame:
    """Smallest e at which Always falls below the best targeted policy at 50% cost, with an
    exact McNemar test between Always and that policy at that e."""
    s = s2_all()
    rows = []
    for dom in DOMAINS:
        d = s[s.domain == dom]
        for e in sorted(d.level.unique()):
            x = d[d.level == e].set_index("policy").accuracy
            best = max(TARGETED, key=lambda f: x[f"{f}@0.5"])
            rows.append({"domain": dom, "e": e, "always": x["always"], "best": best,
                         "best_acc": x[f"{best}@0.5"], "gap": x[f"{best}@0.5"] - x["always"]})
    out = pd.DataFrame(rows)
    tests = []
    for dom in DOMAINS:
        below = out[(out.domain == dom) & (out.gap > 0)]
        if below.empty:
            tests.append({"domain": dom, "crossing_e": None})
            continue
        r = below.sort_values("e").iloc[0]
        env = make_env(dom, S2_CV)
        tag = "s2added" if r.e in ERRORS_ADDED else "s2"
        with core.wrong_answers(r.e):
            alw = pd.DataFrame(core.run(env.domain, env.held, [Always()]))
        sess = pd.read_csv(OUT / f"{tag}_{dom}" / f"level{r.e:g}" / "sessions_matched.csv.gz")
        both = pd.concat([alw, sess[sess.policy == f"{r.best}@0.5"]])
        b, c, n = stats.discordant(both, f"{r.best}@0.5", "always")
        tests.append({"domain": dom, "crossing_e": r.e, "best": r.best, "best_acc": r.best_acc,
                      "always": r.always, "gap": r.gap, "b": b, "c": c, "n": n,
                      "p_mcnemar": stats.mcnemar_exact(b, c)})
    out.to_csv(OUT / "s2_crossing_by_e.csv", index=False)
    t = pd.DataFrame(tests)
    t.to_csv(OUT / "s2_crossing.csv", index=False)
    return t


def scaled_events_domain(f: float) -> Domain:
    """Calibrated meals with life-event probabilities scaled by f and own rates adjusted
    so each fact's total rate stays the same."""
    base = get_domain("meals_calibrated")
    attrs = tuple(Attribute(a.name, a.values, a.volatility,
                            rate=max(0.0, a.base_rate + (1.0 - f) * event_rate(base, a.name)))
                  for a in base.attributes)
    events = tuple(LifeEvent(e.name, e.prob * f, e.touches, e.apply) for e in base.events)
    return Domain(name=base.name, attributes=attrs, tasks=base.tasks, events=events)


def run_s3() -> None:
    """S3: meals only, life events x f, CV = 0, without learned_rates."""
    conds = []
    for f in FACTORS:
        d = scaled_events_domain(f)
        conds.append(("meals_calibrated", f, make_env("meals_calibrated", 0.0, domain=d)))
    run_family_of_conditions("s3", conds, ("ttl", "sxi[exact]", "sxi[coarse]", "info_gain"),
                             learned=False)
    adj = {f: {a.name: round(a.base_rate, 4) for a in scaled_events_domain(f).attributes}
           for f in FACTORS}
    (OUT / "s3_config.json").write_text(json.dumps({"factors": FACTORS, "own_rates": adj}, indent=1))


def table_s2_s3() -> None:
    """paper/tables/realism_s2_s3.tex: accuracy at 25% / 50% per condition."""
    s2 = pd.read_csv(OUT / "s2_summary.csv")
    s3 = pd.read_csv(OUT / "s3_summary.csv")

    def cell(s, dom, lv, fam):
        g = s[(s.domain == dom) & (s.level == lv)]
        if f"{fam}@0.25" not in set(g.policy):
            return "--"
        a = [g[g.policy == f"{fam}@{t:g}"].accuracy.iloc[0] for t in TARGETS]
        return "/".join(f"{x:.3f}".lstrip("0") for x in a)

    fams = ("ttl", "sxi[coarse]", "sxi[exact]", "learned_rates", "info_gain")
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Condition & Level & TTL & S$\times$I (dep.) & S$\times$I (exact) & Learned & Info gain \\",
             r"\midrule"]
    for dom, label in (("commute", "S2 commute"), ("meals_calibrated", "S2 meals")):
        for e in ERRORS:
            lines.append(f"{label} & $e={e:g}$ & " + " & ".join(cell(s2, dom, e, f) for f in fams) + r" \\")
        lines.append(r"\midrule")
    for f in FACTORS:
        lines.append(f"S3 meals & events $\\times{f:g}$ & " +
                     " & ".join(cell(s3, "meals_calibrated", f, x) for x in fams) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "realism_s2_s3.tex").write_text("\n".join(lines) + "\n",
                                                                 encoding="utf-8")


def table_s3() -> None:
    """paper/tables/realism_s3.tex: S3 accuracy at 25% / 50% (S2 is a figure)."""
    s3 = pd.read_csv(OUT / "s3_summary.csv")
    fams = ("ttl", "sxi[coarse]", "sxi[exact]", "info_gain")
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Life events & TTL & S$\times$I (dep.) & S$\times$I (exact) & Info gain \\", r"\midrule"]
    for f in FACTORS:
        g = s3[(s3.domain == "meals_calibrated") & (s3.level == f)].set_index("policy").accuracy
        cells = ["/".join(f"{g[f'{x}@{t:g}']:.3f}".lstrip("0") for t in TARGETS) for x in fams]
        lines.append(f"$\\times{f:g}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "realism_s3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("experiments", nargs="+", choices=["s1", "plot_s1", "s2", "s3", "tables", "s2added", "plot_s2", "crossing"])
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if "s1" in args.experiments:
        run_s1()
    if "plot_s1" in args.experiments:
        plot_s1()
    if "s2" in args.experiments:
        run_s2()
    if "s3" in args.experiments:
        run_s3()
    if "tables" in args.experiments:
        table_s2_s3()
        table_s3()
    if "s2added" in args.experiments:
        run_s2_added()
    if "plot_s2" in args.experiments:
        plot_s2()
    if "crossing" in args.experiments:
        print(s2_crossing().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
