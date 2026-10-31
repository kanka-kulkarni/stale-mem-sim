"""CLI: run policies on paired timelines and write per-session results.

Example:
    python -m experiments.run --domain commute --policies never always \
        --users 50 --sessions 20 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import core, matching, metrics, stats
from simulator.domains import DEFAULT_RATE_DOMAIN, get_domain
from simulator.matching import Calibration
from simulator.impact_llm import load_impact_table
from simulator.policies import Family, build_families, make_policy, sweep_specs

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", default="commute")
    p.add_argument("--policies", nargs="+", default=["never", "always"])
    p.add_argument("--users", type=int, default=50)
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="held-out evaluation seeds (all reported results)")
    p.add_argument("--tune-seeds", type=int, nargs="+", default=[1000],
                   help="seeds used only to tune matched-cost policies")
    p.add_argument("--rate-multiplier", type=float, default=1.0,
                   help="true drift-rate multiplier of the environment")
    p.add_argument("--assumed-multiplier", type=float, default=None,
                   help="multiplier the policies assume when computing P(changed) "
                        "(default: the true one; 1.0 at 0.5x/2x = misspecified rates)")
    p.add_argument("--assumed-rates", choices=["true", "default"], default="true",
                   help="per-attribute rates the policies use: the domain's true rates, or "
                        "the default class rates of its uncalibrated counterpart")
    p.add_argument("--frozen-calibration", type=Path, default=None,
                   help="calibration.json of another run: reuse its tuned settings for the "
                        "matched policies instead of re-tuning (frozen thresholds)")
    p.add_argument("--llm-impact", type=Path, default=None,
                   help="CSV of LLM impact ratings (adds the llm impact version)")
    p.add_argument("--cost-levels", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="question-cost levels for interpolation and matched-cost targets")
    p.add_argument("--sweep", action="store_true",
                   help="add every family's default sweep grid to --policies")
    p.add_argument("--match-families", nargs="*", default=[],
                   help="families to tune to each cost level ('all' = every family in "
                        "policies.build_families)")
    p.add_argument("--tol", type=float, default=0.01,
                   help="allowed |question cost - target| on tuning seeds")
    p.add_argument("--out", type=Path, default=None, help="output dir (default results/<run name>)")
    return p.parse_args(argv)


def run_experiment(
    args: argparse.Namespace, families: dict[str, Family] | None = None
) -> tuple[pd.DataFrame, list[Calibration]]:
    """Tune matched-cost policies on tuning seeds, then replay every policy on the
    held-out seeds' timelines (generated once per seed, shared by all policies)."""
    domain = get_domain(args.domain)
    assumed = assumed_multiplier(args)
    rate_domain = None
    if getattr(args, "assumed_rates", "true") == "default":
        rate_domain = get_domain(DEFAULT_RATE_DOMAIN[domain.name])
    llm_path = getattr(args, "llm_impact", None)
    table = load_impact_table(llm_path, domain) if llm_path else None
    if families is None:
        families = build_families(domain, assumed, args.sessions, table, rate_domain,
                                  include_info_gain="info_gain" in args.match_families)
    sweep = sweep_specs(include_llm=table is not None) if args.sweep else []
    specs = list(dict.fromkeys(args.policies + sweep))
    policies = [make_policy(s, domain, assumed, table, rate_domain, n_sessions=args.sessions,
                            true_multiplier=args.rate_multiplier) for s in specs]
    match = list(families) if args.match_families == ["all"] else args.match_families
    calibrations: list[Calibration] = []
    frozen = getattr(args, "frozen_calibration", None)
    if frozen:
        for rec in json.loads(Path(frozen).read_text()):
            cal = Calibration(**rec)
            if cal.family in families and (not match or cal.family in match):
                calibrations.append(cal)
                if (policy := cal.build(families[cal.family])) is not None:
                    policies.append(policy)
    elif match:
        matching.check_disjoint(args.tune_seeds, args.seeds)
        tuning = [tl for seed in args.tune_seeds for tl in core.generate_timelines(
            domain, seed, args.users, args.sessions, args.rate_multiplier)]
        for fam_name in match:
            fam = families[fam_name]
            for level in args.cost_levels:
                cal = matching.calibrate(domain, tuning, fam, level, tol=args.tol)
                calibrations.append(cal)
                if (policy := cal.build(fam)) is not None:
                    policies.append(policy)
    rows: list[dict] = []
    for seed in args.seeds:
        timelines = core.generate_timelines(
            domain, seed, args.users, args.sessions, args.rate_multiplier
        )
        rows.extend(core.run(domain, timelines, policies))
    df = pd.DataFrame(rows)
    df["rate_multiplier"] = args.rate_multiplier
    df["assumed_multiplier"] = assumed
    return df, calibrations


def assumed_multiplier(args: argparse.Namespace) -> float:
    """The multiplier policies assume: --assumed-multiplier if given, else the true one."""
    assumed = getattr(args, "assumed_multiplier", None)
    return args.rate_multiplier if assumed is None else assumed


def matched_report(summary: pd.DataFrame, calibrations: list[Calibration]) -> pd.DataFrame:
    """Per tuned policy: how it was built, cost on tuning seeds, and held-out results."""
    held = summary[summary.role == "matched"].set_index(["family", "target_cost"])
    rows = []
    for cal in calibrations:
        row = {"family": cal.family, "target_cost": cal.target_cost, "kind": cal.kind,
               "param_a": cal.param_a, "param_b": cal.param_b, "weight": cal.weight,
               "tune_cost": cal.tuned_cost}
        if (cal.family, cal.target_cost) in held.index:
            h = held.loc[(cal.family, cal.target_cost)]
            row.update(realized_cost=h.question_cost, accuracy=h.accuracy,
                       gap_closed=h.gap_closed)
        rows.append(row)
    return pd.DataFrame(rows)


def sanity_checks(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Sanity checks that can be read off a results table: (name, passed, detail)."""
    checks: list[tuple[str, bool, str]] = []
    pols = set(df.policy)
    if "always" in pols:
        acc = df[df.policy == "always"].correct.mean()
        checks.append(("always accuracy == 100%", acc == 1.0, f"{acc:.4f}"))
    if "never" in pols:
        s = df[df.policy == "never"].groupby("session").correct.mean()
        slope = float(np.polyfit(s.index.to_numpy(), s.to_numpy(), 1)[0])
        first, last = s.iloc[:5].mean(), s.iloc[-5:].mean()
        checks.append(("never accuracy declines over sessions", slope < 0 and last < first,
                       f"slope={slope:+.4f}/session, first5={first:.3f}, last5={last:.3f}"))
    if len(pols) >= 2:
        key = ["seed", "user", "session"]
        views = [g.sort_values(key).set_index(key)[["task", "state_hash", "true_answer"]]
                 for _, g in df.groupby("policy")]
        same = all(v.equals(views[0]) for v in views[1:])
        checks.append(("all policies see identical true states and tasks", same,
                       f"{len(views[0])} sessions x {len(views)} policies"))
    return checks


def main(argv: list[str] | None = None) -> int:
    """Run, write CSVs + config, print summary and sanity checks."""
    args = parse_args(argv)
    tag = "-".join(args.policies + [f"match:{f}" for f in args.match_families])
    name = (f"{args.domain}_{tag}_u{args.users}_t{args.sessions}"
            f"_s{'-'.join(map(str, args.seeds))}_m{args.rate_multiplier:g}"
            f"{f'_assume{assumed_multiplier(args):g}' if args.assumed_multiplier else ''}"
            f"{'_defaultrates' if args.assumed_rates == 'default' else ''}"
            f"{'_frozen' if args.frozen_calibration else ''}")
    out = args.out or ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)

    if args.match_families:
        for base in ("never", "always"):  # needed for question cost and gap closed
            if base not in args.policies:
                args.policies.append(base)
    df, calibrations = run_experiment(args)
    summary = metrics.summarize(df)
    per_session = metrics.by_session(df)
    interpolated = metrics.at_matched_cost(summary, args.cost_levels)
    report = matched_report(summary, calibrations)
    comparisons = stats.add_mcnemar(metrics.matched_comparisons(summary), df)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                  platform=platform.platform())
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "calibration.json").write_text(
        json.dumps([c.to_dict() for c in calibrations], indent=2))
    df.to_csv(out / "sessions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    per_session.to_csv(out / "by_session.csv", index=False)
    interpolated.to_csv(out / "interpolated_cost.csv", index=False)
    report.to_csv(out / "matched_policies.csv", index=False)
    comparisons.to_csv(out / "matched_comparisons.csv", index=False)

    pd.set_option("display.width", 140)
    fmt = {"float_format": lambda x: f"{x:.3f}", "na_rep": "N/A", "index": False}
    print(f"config: {json.dumps(config)}\n")
    head = summary[["domain", "policy", "target_cost", *metrics.HEADLINE]].rename(
        columns={"question_cost": "realized_cost"})
    print("headline (held-out seeds; realized_cost = question cost on these seeds):\n"
          + head.to_string(**fmt))
    print("\nsecondary:\n" + summary[["domain", "policy", *metrics.SECONDARY]].to_string(**fmt))
    print("\nsingle-fact test edge cases (pre-ask memory; fraction of all sessions):\n"
          + summary[["domain", "policy", *metrics.EDGE, "n_sessions"]].to_string(**fmt))
    print("\ninterpolated at question cost (figures only; sweep settings within a family):\n"
          + interpolated.to_string(**fmt))
    if len(report):
        print("\nmatched-cost policies (tuned on tune seeds, evaluated on held-out seeds):\n"
              + report.to_string(**fmt))
    if len(comparisons):
        print("\nmatched comparisons (realized held-out costs; flagged = leader's lead does "
              "not exceed its local slope x extra cost):\n" + comparisons.to_string(**fmt))
    print()
    wide = per_session.pivot(index="session", columns="policy", values="accuracy")
    print("accuracy by session:\n" + wide.to_string(float_format=lambda x: f"{x:.3f}"), "\n")
    checks = sanity_checks(df)
    for label, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    print(f"\nwrote {out}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
