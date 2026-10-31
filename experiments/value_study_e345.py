"""Value-estimation study, E3-E5. No LLM calls; the LLM raters enter only through their cached rating tables (E3 marks).

E3: start from exact r_a = P_a x exact impact and perturb it (random noise, or
over-inclusion of facts outside D_k), retune to matched cost, report accuracy.
E4: union bound sum_{a in D_k \\ Q} P_a against observed error for every E1 cell and the
reference policies; plus the error_target(eps) policy.
E5: how often re-asking is not additive (joint-only, cancelling sessions), and realized
corrections against the sum of exact r_a over asked facts.

Example:
    python -m experiments.value_study_e345
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from experiments.value_study import (CHANGED, MATTERS, OUT, RATERS, SELECTS, SESSIONS,
                                     TARGETS, TOL, Recorder, cell_family_obj, rating_table,
                                     setup)
from simulator import core, metrics, stats
from simulator.core import Domain, Memory, Task
from simulator.matching import Calibration, calibrate
from simulator.policies import Always, Family, Never, Oracle, _RatePolicy

NOISE_STREAM = 0x4E  # E3 noise draws
DELTAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
OVER = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = (0.02, 0.05, 0.1, 0.2)


class Perturbed(_RatePolicy):
    """Exact value P_a x exact impact, perturbed by uniform noise ("noise") or by
    giving irrelevant facts P_a x level ("over")."""

    stateful = False

    def __init__(self, domain: Domain, kind: str, level: float, select: str, param: float) -> None:
        super().__init__(domain)
        self.kind, self.level, self.select, self.param = kind, level, select, param
        self.family = f"{select}[{kind}={level:g}]"
        self.name = f"{self.family}({param:.6g})"
        self.u = None

    def start_user(self, seed: int, user: int) -> None:
        """Draw this user's noise table (sessions x facts)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, user, NOISE_STREAM]))
        self.u = rng.uniform(-1.0, 1.0, size=(SESSIONS + 1, len(self.all_attributes)))

    def value(self, memory: Memory, task: Task, t: int, a: str) -> float:
        """Perturbed value of re-asking a."""
        p = self.p_changed(memory, a, t)
        if a in task.depends_on:
            r = p * self.impact("exact", task, a, memory)
        else:
            r = p * self.level if self.kind == "over" else 0.0
        if self.kind == "noise":
            r = min(1.0, max(0.0, r + self.level * self.u[t, self.order[a]]))
        return r

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Threshold or Budget on the perturbed values."""
        vals = {a: self.value(memory, task, t, a) for a in self.all_attributes}
        if self.select == "threshold":
            return [a for a, v in vals.items() if v > self.param]
        ranked = sorted((a for a in vals if vals[a] > 0), key=lambda a: (-vals[a], self.order[a]))
        return ranked[: int(self.param)]


class ErrorTarget(_RatePolicy):
    """Error-targeted policy: re-ask facts in D_k in
    decreasing P_a until the remaining sum of P_a is at most eps."""

    stateful = False

    def __init__(self, domain: Domain, eps: float) -> None:
        super().__init__(domain)
        self.eps, self.family = eps, "error_target"
        self.name = f"error_target({eps:g})"

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Largest P_a first until the union bound is <= eps."""
        p = {a: self.p_changed(memory, a, t) for a in task.depends_on}
        remaining, asked = sum(p.values()), []
        for a in sorted(p, key=lambda a: (-p[a], self.order[a])):
            if remaining <= self.eps:
                break
            asked.append(a)
            remaining -= p[a]
        return asked


def perturbed_family(d: Domain, kind: str, level: float, select: str) -> Family:
    """Tunable family of one perturbation level."""
    make = lambda x: Perturbed(d, kind, level, select, x)  # noqa: E731
    name = f"{select}[{kind}={level:g}]"
    if select == "budget":
        return Family(name, "integer", make, values=tuple(range(len(d.attributes) + 1)))
    return Family(name, "continuous", make, bounds=(0.0, 1.0))


def rater_overinclusion(d: Domain) -> dict[str, float]:
    """Each rater's c: mean normalized rating over its irrelevant (task, fact) pairs."""
    out = {}
    for key in RATERS:
        table = rating_table(d, f"llm-{key}")
        irrelevant = [v for (k, a), v in table.items() if a not in d.task(k).depends_on]
        out[key] = float(np.mean(irrelevant))
    return out


def run_e3(name: str) -> None:
    """E3 for one domain: every level x selection, tuned to 25% and 50%."""
    S = setup(name)
    d, out = S.domain, OUT / f"e3_{name}"
    out.mkdir(parents=True, exist_ok=True)
    records = core.run(d, S.held, [Never(), Always()])
    cals = []
    grid = [("noise", x) for x in DELTAS] + [("over", c) for c in OVER]
    for kind, level in grid:
        for sel in SELECTS:
            fam, params = perturbed_family(d, kind, level, sel), set()
            for target in TARGETS:
                cal = calibrate(d, S.tuning, fam, target, TOL)
                cals.append({**cal.to_dict(), "kind": kind, "level": level, "select": sel})
                pol = cal.build(fam)
                if pol is not None:
                    records += core.run(d, S.held, [pol])
                    params |= {p for p in (cal.param_a, cal.param_b) if p is not None}
            records += core.run(d, S.held, [fam.make(p) for p in sorted(params)])
    df = pd.DataFrame(records)
    summary = metrics.summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    df[df.role == "matched"].to_csv(out / "sessions_matched.csv.gz", index=False)
    (out / "calibration.json").write_text(json.dumps(cals, indent=1))
    (out / "config.json").write_text(json.dumps({
        "domain": name, "deltas": DELTAS, "over": OVER, "targets": TARGETS, "tol": TOL,
        "noise_stream": NOISE_STREAM, "rater_overinclusion": rater_overinclusion(d)}, indent=1))
    print(f"E3 {name}: {(summary.role == 'matched').sum()} matched policies")


def e3_comparisons() -> pd.DataFrame:
    """Each perturbed level vs the unperturbed policy (same domain, kind, selection,
    cost); one Holm family over E3."""
    rows = []
    for name in ("commute", "meals_calibrated"):
        sub = OUT / f"e3_{name}"
        s = pd.read_csv(sub / "summary.csv")
        m = s[s.role == "matched"].set_index("policy")
        df = pd.read_csv(sub / "sessions_matched.csv.gz")
        for kind, levels in (("noise", DELTAS), ("over", OVER)):
            for sel in SELECTS:
                for t in TARGETS:
                    base = f"{sel}[{kind}=0]@{t:g}"
                    for lv in levels[1:]:
                        pol = f"{sel}[{kind}={lv:g}]@{t:g}"
                        if base not in m.index or pol not in m.index:
                            continue
                        b, c, n = stats.discordant(df, base, pol)
                        rows.append({"domain": name, "kind": kind, "select": sel, "target_cost": t,
                                     "level": lv, "acc_base": m.loc[base, "accuracy"],
                                     "acc": m.loc[pol, "accuracy"], "cost_base": m.loc[base, "question_cost"],
                                     "cost": m.loc[pol, "question_cost"], "b": b, "c": c, "n_paired": n,
                                     "p_mcnemar": stats.mcnemar_exact(b, c)})
    out = pd.DataFrame(rows)
    out["p_holm"] = stats.holm(out.p_mcnemar.tolist())
    out.to_csv(OUT / "e3_comparisons.csv", index=False)
    return out


def rebuild_e1(d: Domain, name: str) -> list:
    """The E1 matched policies, rebuilt from their stored calibrations."""
    cals = json.loads((OUT / f"e1_{name}" / "calibration.json").read_text())
    fams = {f.name: f for f in (cell_family_obj(d, s, c, m) for s in SELECTS
                                for c in CHANGED for m in MATTERS)}
    out = []
    for c in cals:
        if c["family"] in fams:
            pol = Calibration(**c).build(fams[c["family"]])
            if pol is not None:
                out.append(pol)
    return out


def run_e45(name: str) -> None:
    """E4 and E5 for one domain, by replaying every E1 matched policy (and never,
    always, oracle) with a recorder."""
    S = setup(name)
    d, out = S.domain, OUT / f"e45_{name}"
    out.mkdir(parents=True, exist_ok=True)
    rates = _RatePolicy(d)  # true rates (1x; calibrated for meals)
    pols = [Never(), Always(), Oracle(d, SESSIONS)] + rebuild_e1(d, name)
    pols += [ErrorTarget(d, e) for e in EPS]
    e1 = pd.read_csv(OUT / f"e1_{name}" / "summary.csv").set_index("policy")
    rows, records = [], []
    for pol in pols:
        rec = Recorder(pol)
        recs = core.run(d, S.held, [rec])
        records += recs
        bounds, rsum = [], []
        for (seed, user, t, task_name, values, verified, asked) in rec.rows:
            task, mem = d.task(task_name), Memory(values=values, verified_at=verified)
            unasked = [a for a in task.depends_on if a not in asked]
            bounds.append(min(1.0, sum(rates.p_changed(mem, a, t) for a in unasked)))
            rsum.append(sum(rates.p_changed(mem, a, t) * core.impact_exact(d, task, a, values)
                            for a in asked if a in task.depends_on))
        r = pd.DataFrame(recs)
        corrections = r.correct.astype(int) - r.pre_ask_correct.astype(int)
        rows.append({"policy": rec.name, "family": getattr(pol, "family", pol.name),
                     "target_cost": getattr(pol, "target_cost", np.nan),
                     "accuracy": r.correct.mean(), "observed_error": 1 - r.correct.mean(),
                     "mean_bound": float(np.mean(bounds)),
                     "sessions_bound_violated": None,
                     "net_corrections": int(corrections.sum()),
                     "fixed": int((corrections == 1).sum()), "broken": int((corrections == -1).sum()),
                     "sum_r_asked": float(np.sum(rsum)),
                     "joint_only_frac": float((~r.pre_ask_correct & (r.n_answer_changing == 0)).mean()),
                     "cancelling_frac": float((r.pre_ask_correct & (r.n_answer_changing > 0)).mean()),
                     "mean_questions": r.n_questions.mean()})
        if getattr(pol, "family", "") in e1.family.values and rec.name in e1.index:
            if not np.isclose(rows[-1]["accuracy"], e1.loc[rec.name, "accuracy"]):
                raise RuntimeError(f"{rec.name}: rebuilt policy does not reproduce E1")
    res = pd.DataFrame(rows).drop(columns="sessions_bound_violated")
    always_q = res.set_index("policy").loc["always", "mean_questions"]
    res["question_cost"] = res.mean_questions / always_q
    res.to_csv(out / "e45.csv", index=False)
    print(f"E4/E5 {name}: {len(res)} policies")


def main(argv: list[str] | None = None) -> int:
    """Run E3, then E4/E5, for both domains."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", nargs="+", default=["commute", "meals_calibrated"])
    p.add_argument("--only", choices=["e3", "e45"], default=None)
    args = p.parse_args(argv)
    for name in args.domain:
        if args.only in (None, "e3"):
            run_e3(name)
        if args.only in (None, "e45"):
            run_e45(name)
    if all((OUT / f"e3_{n}" / "summary.csv").exists() for n in ("commute", "meals_calibrated")):
        c = e3_comparisons()
        print(f"{len(c)} E3 comparisons, {(c.p_holm < 0.05).sum()} significant (Holm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
