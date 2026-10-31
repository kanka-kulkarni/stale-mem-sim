"""Add-on baselines for every condition of the main comparison: oracle (upper bound, uses the true
state), always_all (reference) and the information-gain baseline info_gain(threshold)
(swept and tuned to the matched-cost targets).

Results go to results/baselines_<domain>/<condition>/ so the main results and their
Holm families are untouched. info_gain's matched policies are compared with the main
matched policies on the same held-out sessions (paired design); those comparisons form
their own Holm family per domain.

Example:
    python -m experiments.baselines --domain commute
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import pandas as pd

from experiments import run
from experiments.m2 import Condition
from simulator import metrics, stats
from simulator.domains import DEFAULT_RATE_DOMAIN
from simulator.policies import info_gain_specs

ROOT = Path(__file__).resolve().parents[1]
BASE_POLICIES = ["never", "always", "always_all", "oracle"]


def conditions(domain: str, out: Path) -> list[Condition]:
    """The same conditions as the main comparison, with frozen settings taken from this
    add-on's own 1x runs (so info_gain is frozen like the other families)."""
    ms = (0.5, 1.0, 2.0)
    conds = [Condition("correct", m, m) for m in ms]
    conds += [Condition("misspecified", m, 1.0) for m in ms if m != 1.0]
    conds += [Condition("frozen", m, 1.0, frozen_from=out / "m1" / "calibration.json")
              for m in ms if m != 1.0]
    counterpart = DEFAULT_RATE_DOMAIN.get(domain)
    if counterpart:
        conds += [Condition("default_rates", m, m, "default") for m in ms]
        ref = ROOT / "results" / f"baselines_{counterpart}" / "m1" / "calibration.json"
        conds.append(Condition("default_rates_frozen", 1.0, 1.0, "default", ref))
    return conds


def run_condition(domain: str, cond: Condition, sub: Path) -> int:
    """One run.py invocation for the baselines; stdout goes to <sub>.log."""
    cli = ["--domain", domain, "--policies", *BASE_POLICIES, *info_gain_specs(),
           "--match-families", "info_gain", "--users", "50", "--sessions", "20",
           "--seeds", "0", "1", "2", "3", "4", "--tune-seeds", "1000", "1001", "1002",
           "1003", "1004", "--rate-multiplier", str(cond.multiplier),
           "--assumed-multiplier", str(cond.assumed), "--assumed-rates", cond.assumed_rates,
           "--cost-levels", "0.25", "0.5", "0.75", "--tol", "0.01", "--out", str(sub)]
    if cond.frozen_from:
        cli += ["--frozen-calibration", str(cond.frozen_from)]
    sub.mkdir(parents=True, exist_ok=True)
    with open(sub.parent / f"{sub.name}.log", "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log):
        return run.main(cli)


def compare_with_m2(domain: str, cond: Condition, base_sub: Path) -> pd.DataFrame:
    """info_gain matched policies vs every main matched policy at the same target, on the
    shared held-out sessions."""
    m2_sub = ROOT / "results" / f"m2_{domain}" / cond.dirname
    s_base = pd.read_csv(base_sub / "summary.csv")
    s_m2 = pd.read_csv(m2_sub / "summary.csv")
    # paired-design check: never/always must be identical in both runs
    for p in ("never", "always"):
        a = s_base.set_index("policy").loc[p, ["accuracy", "mean_questions"]]
        b = s_m2.set_index("policy").loc[p, ["accuracy", "mean_questions"]]
        if not a.equals(b):
            raise RuntimeError(f"{domain}/{cond.dirname}: {p} differs from the main run (pairing broken)")
    summary = pd.concat([s_base[s_base.role == "matched"], s_m2[s_m2.role == "matched"]])
    comp = metrics.matched_comparisons(summary)
    comp = comp[comp.policy_a.str.startswith("info_gain@") | comp.policy_b.str.startswith("info_gain@")]
    if comp.empty:
        return comp
    keep = set(comp.policy_a) | set(comp.policy_b)
    df = pd.concat([pd.read_csv(base_sub / "sessions.csv"), pd.read_csv(m2_sub / "sessions.csv")])
    df = df[df.policy.isin(keep)]
    return stats.add_mcnemar(comp, df)


def main(argv: list[str] | None = None) -> int:
    """Run every condition (skipping finished ones), then compare and summarise."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", required=True)
    args = p.parse_args(argv)
    out = ROOT / "results" / f"baselines_{args.domain}"
    out.mkdir(parents=True, exist_ok=True)
    statuses, summaries, comps = [], [], []
    for cond in conditions(args.domain, out):
        sub = out / cond.dirname
        if not (sub / "summary.csv").exists():
            statuses.append(run_condition(args.domain, cond, sub))
        s = pd.read_csv(sub / "summary.csv")
        c = compare_with_m2(args.domain, cond, sub)
        for d in (s, c):
            d["rate_multiplier"], d["assumed_multiplier"] = cond.multiplier, cond.assumed
            d["assumed_rates"], d["condition"] = cond.assumed_rates, cond.label
        summaries.append(s)
        comps.append(c)
    summary = pd.concat(summaries, ignore_index=True)
    comp = pd.concat(comps, ignore_index=True)
    comp["p_holm"] = stats.holm(comp.p_mcnemar.tolist()) if len(comp) else []
    summary.to_csv(out / "summary_all.csv", index=False)
    comp.to_csv(out / "info_gain_comparisons_all.csv", index=False)
    ok = all(x == 0 for x in statuses)
    print(f"wrote {out}; sanity checks {'passed' if ok else 'FAILED'}; "
          f"{len(comp)} info_gain comparisons")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
