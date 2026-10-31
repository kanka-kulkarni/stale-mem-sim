"""Value-estimation study: E1 ladder and E2 value-estimate quality. No LLM calls: LLM-rated impact comes from the cached rating
tables, and per-session LLM decisions are replayed from the per-session LLM runs.

E1 crosses the "has it changed?" source (none, age, rate and age) with the "does it
matter?" source (none, coarse, five LLM raters, exact), under Threshold and Budget
selection, each tuned to 25% and 50% of Always's questions. E2 scores every
(session, stored fact) of each tuned policy's own held-out trajectory against the
oracle's label (stale and answer-changing).

Example:
    python -m experiments.value_study --domain commute
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import core, metrics, stats
from simulator.core import Domain, Memory, Task, Timeline
from simulator.domains import get_domain
from simulator.impact_llm import load_impact_table
from simulator.matching import calibrate
from simulator.policies import (AlwaysAll, Always, Family, Never, Oracle,
                                _RatePolicy)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "value_study"
TUNE_SEEDS, EVAL_SEEDS = (1000, 1001, 1002, 1003, 1004), (0, 1, 2, 3, 4)
USERS, SESSIONS, TARGETS, TOL = 50, 20, (0.25, 0.5), 0.01
RANDOM_STREAM = 0x5A  # coins of the random reference policy
CHANGED = ("none", "age", "rate")
RATERS = {"3b": "qwen2.5-3b", "7b": "qwen2.5-7b", "14b": "qwen2.5-14b",
          "llama8b": "llama3.1-8b", "qwen35": "qwen3.5-9b-nothink"}
MATTERS = ("none", "coarse", *(f"llm-{r}" for r in RATERS), "exact")
SELECTS = ("threshold", "budget")
# existing families that are exactly an E1 cell (results reproduced, not copied)
REUSED = {("threshold", "rate", "none"): "info_gain",
          ("threshold", "rate", "coarse"): "sxi[coarse]", ("budget", "rate", "coarse"): "budget[coarse]",
          ("threshold", "rate", "exact"): "sxi[exact]", ("budget", "rate", "exact"): "budget[exact]",
          ("threshold", "rate", "llm-7b"): "sxi[llm]", ("budget", "rate", "llm-7b"): "budget[llm]",
          ("threshold", "age", "coarse"): "ttl"}


def cell_family(select: str, changed: str, matters: str) -> str:
    """Family label of one E1 cell, e.g. 'threshold[rate x exact]'."""
    return f"{select}[{changed} x {matters}]"


def rating_table(domain: Domain, matters: str) -> dict | None:
    """Cached LLM impact table for an 'llm-<rater>' source, else None."""
    if not matters.startswith("llm-"):
        return None
    dom = "commute" if domain.name == "commute" else "meals"
    tag = RATERS[matters.removeprefix("llm-")]
    return load_impact_table(ROOT / "calibration" / f"llm_impact_{dom}_{tag}.csv", domain)


class ValuePolicy(_RatePolicy):
    """One E1 cell: value s_a = C_a x M_k(a), selected by Threshold (s_a > theta) or
    Budget (top k with s_a > 0, ties by declaration order)."""

    def __init__(self, domain: Domain, changed: str, matters: str, select: str,
                 param: float, table: dict | None = None) -> None:
        super().__init__(domain, 1.0, table)
        self.changed, self.matters, self.select, self.param = changed, matters, select, param
        self.family = cell_family(select, changed, matters)
        self.name = f"{self.family}({param:.6g})"

    def changed_part(self, memory: Memory, a: str, t: int) -> float:
        """Chance-it-changed part: 1, age d_a, or P_a."""
        if self.changed == "none":
            return 1.0
        if self.changed == "age":
            return float(memory.age(a, t))
        return self.p_changed(memory, a, t)

    def matters_part(self, task: Task, a: str, memory: Memory) -> float:
        """Matters-for-this-task part: 1, coarse, LLM rating / 3, or exact impact."""
        if self.matters == "none":
            return 1.0
        if self.matters.startswith("llm-"):
            return self.impact("llm", task, a, memory)
        return self.impact(self.matters, task, a, memory)

    def value(self, memory: Memory, task: Task, t: int, a: str) -> float:
        """Estimated value of re-asking a (0 for facts outside the candidates)."""
        if self.matters in ("coarse", "exact") and a not in task.depends_on:
            return 0.0
        return self.changed_part(memory, a, t) * self.matters_part(task, a, memory)

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Threshold or Budget selection on the cell's values."""
        cands = (task.depends_on if self.matters in ("coarse", "exact")
                 else self.all_attributes)
        vals = {a: self.value(memory, task, t, a) for a in cands}
        if self.select == "threshold":
            return [a for a, v in vals.items() if v > self.param]
        ranked = sorted((a for a in vals if vals[a] > 0), key=lambda a: (-vals[a], self.order[a]))
        return ranked[: int(self.param)]


class RandomPolicy:
    """Reference: ask each stored fact independently with probability q (dedicated
    coin stream keyed by (seed, user, RANDOM_STREAM); one coin per fact per session)."""

    family = "random"
    stateful = True  # holds a coin stream; tuned by bisecting q, never mixed

    def __init__(self, domain: Domain, q: float) -> None:
        self.attrs = [a.name for a in domain.attributes]
        self.q, self.name, self._rng = q, f"random({q:.6g})", None

    def start_user(self, seed: int, user: int) -> None:
        """Seed this user's coin stream."""
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, user, RANDOM_STREAM]))

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Each fact with probability q."""
        u = self._rng.random(len(self.attrs))
        return [a for a, x in zip(self.attrs, u) if x < self.q]


class Replay:
    """Replays logged decisions (the per-session LLM runs) to recover their memory
    trajectory without calling the model."""

    stateful = True

    def __init__(self, log: pd.DataFrame, name: str) -> None:
        self.asked = {(int(r.seed), int(r.user), int(r.session)):
                      ([] if pd.isna(r.asked) else str(r.asked).split(";"))
                      for r in log.itertuples()}
        self.name = self.family = name
        self._key = None

    def start_user(self, seed: int, user: int) -> None:
        """Select this user's log."""
        self._key = (seed, user)

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """The logged re-asks of this session."""
        return self.asked[(*self._key, t)]


class Recorder:
    """Wraps a policy and keeps each session's pre-ask memory and decision (for E2)."""

    def __init__(self, inner) -> None:
        self.inner, self.rows, self._user = inner, [], None
        for k in ("name", "family", "role", "target_cost", "stateful"):
            if hasattr(inner, k):
                setattr(self, k, getattr(inner, k))

    def start_user(self, seed: int, user: int) -> None:
        """Remember the user and forward the hook."""
        self._user = (seed, user)
        if hasattr(self.inner, "start_user"):
            self.inner.start_user(seed, user)

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Snapshot memory, then delegate."""
        asked = self.inner.decide(memory, task, t)
        self.rows.append((*self._user, t, task.name, dict(memory.values),
                          dict(memory.verified_at), list(asked)))
        return asked


# --------------------------------------------------------------------------- E2 metrics


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """AP with ties handled at distinct thresholds (precision at each, weighted by the
    recall gained there)."""
    pos = label.sum()
    if pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, y = score[order], label[order]
    tp = np.cumsum(y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # end of each tie block
    prec, rec = tp[last] / (last + 1), tp[last] / pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC AUC via average ranks (Mann-Whitney)."""
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def e2_rows(domain: Domain, rec: Recorder, states: dict, scorer: ValuePolicy | None) -> dict:
    """Score/label arrays of one recorded trajectory -> AP, AUC (if scored), precision
    and recall of the asked sets."""
    scores, labels, asked_flags = [], [], []
    for seed, user, t, task_name, values, verified, asked in rec.rows:
        task = domain.task(task_name)
        truth = states[(seed, user)][t]
        positive = set(core.answer_changing_stale(task, values, truth))
        mem = Memory(values=values, verified_at=verified)
        for a in domain.attributes:
            labels.append(a.name in positive)
            asked_flags.append(a.name in asked)
            if scorer is not None:
                scores.append(scorer.value(mem, task, t, a.name))
    y, q = np.array(labels), np.array(asked_flags)
    out = {"n_pairs": len(y), "n_positive": int(y.sum()),
           "precision": float((y & q).sum() / q.sum()) if q.sum() else float("nan"),
           "recall": float((y & q).sum() / y.sum()) if y.sum() else float("nan")}
    if scorer is not None:
        s = np.array(scores)
        out.update(average_precision=average_precision(s, y), roc_auc=roc_auc(s, y))
    return out


# --------------------------------------------------------------------------- runs


@dataclass
class Setup:
    domain: Domain
    tuning: list[Timeline]
    held: list[Timeline]
    states: dict


def setup(name: str) -> Setup:
    """Tuning and held-out timelines (existing split, 1x)."""
    d = get_domain(name)
    tl = lambda seeds: [x for s in seeds for x in core.generate_timelines(d, s, USERS, SESSIONS)]  # noqa: E731
    held = tl(EVAL_SEEDS)
    return Setup(d, tl(TUNE_SEEDS), held, {(x.seed, x.user): x.states for x in held})


def cell_family_obj(d: Domain, select: str, changed: str, matters: str) -> Family:
    """Tunable family of one cell: continuous theta or integer k."""
    table = rating_table(d, matters)
    make = lambda x: ValuePolicy(d, changed, matters, select, x, table)  # noqa: E731
    name = cell_family(select, changed, matters)
    if select == "budget":
        return Family(name, "integer", make, values=tuple(range(len(d.attributes) + 1)))
    hi = float(SESSIONS) if changed == "age" else 1.0
    return Family(name, "continuous", make, bounds=(0.0, hi))


def families(d: Domain) -> list[Family]:
    """Every E1 cell under both selections, plus the random reference."""
    fams = [cell_family_obj(d, s, c, m) for s in SELECTS for c in CHANGED for m in MATTERS]
    fams.append(Family("random", "continuous", lambda q: RandomPolicy(d, q), bounds=(0.0, 1.0)))
    return fams


def run_domain(name: str) -> None:
    """E1 and E2 for one domain; writes results/value_study/e1_<domain>/."""
    S = setup(name)
    d, out = S.domain, OUT / f"e1_{name}"
    out.mkdir(parents=True, exist_ok=True)
    refs = [Never(), Always(), AlwaysAll(d), Oracle(d, SESSIONS)]
    records = core.run(d, S.held, refs)
    cals, e2 = [], []
    for fam in families(d):
        params = set()
        for target in TARGETS:
            cal = calibrate(d, S.tuning, fam, target, TOL)
            cals.append(cal.to_dict())
            pol = cal.build(fam)
            if pol is None:
                continue
            params |= {p for p in (cal.param_a, cal.param_b) if p is not None}
            rec = Recorder(pol)
            records += core.run(d, S.held, [rec])
            scorer = None if fam.name == "random" else fam.make(0.0)
            e2.append({"family": fam.name, "policy": pol.name, "target_cost": target,
                       **e2_rows(d, rec, S.states, scorer)})
        # held-out sweep points at the bracketing settings (curve slope for the cost check)
        records += core.run(d, S.held, [fam.make(p) for p in sorted(params)])
    df = pd.DataFrame(records)
    summary = metrics.summarize(df)
    df[df.role == "matched"].to_csv(out / "sessions_matched.csv.gz", index=False)
    df[df.policy.isin(["never", "always", "always_all", "oracle"])].to_csv(
        out / "sessions_reference.csv.gz", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(e2).to_csv(out / "e2_scores.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(cals, indent=1))
    (out / "config.json").write_text(json.dumps({
        "domain": name, "tune_seeds": TUNE_SEEDS, "eval_seeds": EVAL_SEEDS, "users": USERS,
        "sessions": SESSIONS, "rate_multiplier": 1.0, "targets": TARGETS, "tol": TOL,
        "changed": CHANGED, "matters": MATTERS, "raters": RATERS, "selects": SELECTS,
        "random_stream": RANDOM_STREAM}, indent=1))
    e2_llm(name, S)
    print(f"{name}: {summary[summary.role == 'matched'].shape[0]} matched policies")


LLM_RUNS = {"commute": {"LLM 7B": "m3_commute_tune20", "LLM 7B guided": "m3_commute_guided",
                        "LLM 14B": "m3_commute_14b"},
            "meals_calibrated": {"LLM 7B": "m3_meals_calibrated",
                                 "LLM 7B guided": "m3_meals_calibrated_guided",
                                 "LLM 14B": "m3_meals_calibrated_14b"}}


def e2_llm(name: str, S: Setup) -> None:
    """Precision and recall of the per-session LLMs, replayed from their 20-user logs;
    checks that the replay reproduces the logged accuracy."""
    rows = []
    held20 = [x for x in S.held if x.seed == 0 and x.user < 20]
    for label, run_dir in LLM_RUNS[name].items():
        log = pd.read_csv(ROOT / "results" / run_dir / "sessions.csv")
        log = log[log.family == "llm"]
        rec = Recorder(Replay(log, label))
        replay = pd.DataFrame(core.run(S.domain, held20, [rec]))
        if not np.isclose(replay.correct.mean(), log.correct.mean()):
            raise RuntimeError(f"{label}: replay does not reproduce the logged accuracy")
        rows.append({"policy": label, "source": run_dir, "accuracy": float(log.correct.mean()),
                     "users": 20, **e2_rows(S.domain, rec, S.states, None)})
    pd.DataFrame(rows).to_csv(OUT / f"e1_{name}" / "e2_llm.csv", index=False)


def comparisons() -> pd.DataFrame:
    """Paired tests between cells sharing a row or a column (same domain, selection and
    cost), one Holm family over all E1 tests; plus the cost check."""
    comps = []
    for name in ("commute", "meals_calibrated"):
        sub = OUT / f"e1_{name}"
        summ = pd.read_csv(sub / "summary.csv")
        summ = summ[summ.family != "random"]
        comp = metrics.matched_comparisons(summ)
        fa = comp.policy_a.str.extract(r"^(\w+)\[(\w+) x ([\w-]+)\]")
        fb = comp.policy_b.str.extract(r"^(\w+)\[(\w+) x ([\w-]+)\]")
        keep = (fa[0] == fb[0]) & ((fa[1] == fb[1]) | (fa[2] == fb[2]))
        comp = comp[keep.fillna(False)]
        df = pd.read_csv(sub / "sessions_matched.csv.gz")
        comps.append(stats.add_mcnemar(comp, df))
    comp = pd.concat(comps, ignore_index=True)
    comp["p_holm"] = stats.holm(comp.p_mcnemar.tolist())
    comp.to_csv(OUT / "e1_comparisons.csv", index=False)
    return comp


def main(argv: list[str] | None = None) -> int:
    """Run E1/E2 for the requested domains, then the E1 paired tests."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", nargs="+", default=["commute", "meals_calibrated"])
    args = p.parse_args(argv)
    for name in args.domain:
        run_domain(name)
    if all((OUT / f"e1_{n}" / "summary.csv").exists() for n in ("commute", "meals_calibrated")):
        comp = comparisons()
        print(f"{len(comp)} E1 comparisons, {(comp.p_holm < 0.05).sum()} significant (Holm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
