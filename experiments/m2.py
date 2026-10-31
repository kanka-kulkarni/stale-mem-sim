"""Main comparison: sweep + matched-cost runs at several rate multipliers, under correctly
specified rates and (at multipliers != 1) misspecified rates where policies assume 1x.
Then pooled paired statistics (Holm across every matched comparison reported for the
domain) and the figures.

Example:
    python -m experiments.m2 --domain commute --llm-impact calibration/llm_impact_commute_qwen2.5-7b.csv
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments import plots, run
from simulator import stats
from simulator.domains import DEFAULT_RATE_DOMAIN

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", default="commute")
    p.add_argument("--multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--no-misspecified", action="store_true",
                   help="skip the conditions where policies assume 1x rates (re-tuned/frozen)")
    p.add_argument("--llm-impact", type=Path, default=None)
    p.add_argument("--users", type=int, default=50)
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--tune-seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    p.add_argument("--cost-levels", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--force", action="store_true", help="re-run conditions that have results")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


@dataclass(frozen=True)
class Condition:
    """One run: true multiplier, what the policies assume, and whether tuned settings
    are frozen from another run instead of re-tuned."""

    label: str
    multiplier: float
    assumed: float
    assumed_rates: str = "true"       # "true" | "default"
    frozen_from: Path | None = None   # calibration.json to reuse

    @property
    def dirname(self) -> str:
        """Output subdirectory of this condition."""
        if self.label == "correct":
            return f"m{self.multiplier:g}"
        if self.label == "misspecified":
            return f"m{self.multiplier:g}_assume{self.assumed:g}"
        return f"m{self.multiplier:g}_{self.label}"


def conditions(args: argparse.Namespace, out: Path) -> list[Condition]:
    """Every run for this domain, in dependency order (1x correct before frozen runs)."""
    ms = args.multipliers
    conds = [Condition("correct", m, m) for m in ms]
    if not args.no_misspecified:
        conds += [Condition("misspecified", m, 1.0) for m in ms if m != 1.0]
        if 1.0 in ms:
            base = out / "m1" / "calibration.json"
            conds += [Condition("frozen", m, 1.0, frozen_from=base) for m in ms if m != 1.0]
    counterpart = DEFAULT_RATE_DOMAIN.get(args.domain)
    if counterpart:
        conds += [Condition("default_rates", m, m, "default") for m in ms]
        ref = ROOT / "results" / f"m2_{counterpart}" / "m1" / "calibration.json"
        if 1.0 in ms and ref.exists():
            conds.append(Condition("default_rates_frozen", 1.0, 1.0, "default", ref))
    return conds


def run_condition(args: argparse.Namespace, cond: Condition, sub: Path) -> int:
    """One run.py invocation; stdout goes to <sub>.log."""
    cli = ["--domain", args.domain, "--sweep", "--match-families", "all",
           "--users", str(args.users), "--sessions", str(args.sessions),
           "--seeds", *map(str, args.seeds), "--tune-seeds", *map(str, args.tune_seeds),
           "--rate-multiplier", str(cond.multiplier), "--assumed-multiplier", str(cond.assumed),
           "--assumed-rates", cond.assumed_rates,
           "--cost-levels", *map(str, args.cost_levels), "--tol", str(args.tol),
           "--out", str(sub)]
    if cond.frozen_from:
        cli += ["--frozen-calibration", str(cond.frozen_from)]
    if args.llm_impact:
        cli += ["--llm-impact", str(args.llm_impact)]
    sub.mkdir(parents=True, exist_ok=True)
    log_path = sub.parent / f"{sub.name}.log"  # not with_suffix: "m0.5" has a ".5" suffix
    with open(log_path, "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log):
        return run.main(cli)


def main(argv: list[str] | None = None) -> int:
    """Run every condition (reusing finished ones unless --force), pool comparisons,
    apply Holm, write tables and figures."""
    args = parse_args(argv)
    out = args.out or ROOT / "results" / f"m2_{args.domain}"
    out.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, statuses = [], [], []
    for cond in conditions(args, out):
        sub = out / cond.dirname
        if args.force or not (sub / "summary.csv").exists():
            statuses.append(run_condition(args, cond, sub))
        for frame, name in ((summaries, "summary.csv"), (comparisons, "matched_comparisons.csv")):
            d = pd.read_csv(sub / name)
            d["rate_multiplier"], d["assumed_multiplier"] = cond.multiplier, cond.assumed
            d["assumed_rates"], d["condition"] = cond.assumed_rates, cond.label
            frame.append(d)
    summary = pd.concat(summaries, ignore_index=True)
    comp = pd.concat(comparisons, ignore_index=True)
    comp["p_holm"] = stats.holm(comp.p_mcnemar.tolist()) if len(comp) else []
    summary.to_csv(out / "summary_all.csv", index=False)
    comp.to_csv(out / "matched_comparisons_all.csv", index=False)

    n_users = len(args.seeds) * args.users
    plots.main_figure(summary[summary.condition == "correct"],
                      out / "accuracy_vs_questions.png",
                      f"{args.domain}: accuracy vs. questions asked "
                      f"({n_users} held-out users × {args.sessions} sessions)")
    plots.main_figure(summary[summary.condition == "correct"],
                      out / "accuracy_vs_questions_variants.png",
                      f"{args.domain}: task-blind and LLM-impact variants vs. their task-aware "
                      f"counterparts ({n_users} held-out users × {args.sessions} sessions)",
                      families=plots.VARIANTS)
    mis = summary[summary.condition == "misspecified"]
    if len(mis):
        plots.main_figure(mis, out / "accuracy_vs_questions_misspecified.png",
                          f"{args.domain}: policies assume 1× rates while true rates differ "
                          f"({n_users} held-out users × {args.sessions} sessions)",
                          panel_title="true rates {m:g}×, policies assume 1×")
    (out / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2))
    print(f"wrote {out}; sanity checks {'passed' if all(x == 0 for x in statuses) else 'FAILED'}")
    return max(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
