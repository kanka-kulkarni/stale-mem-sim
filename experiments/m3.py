"""Per-session LLM study: the LLM policy on a subset, with other families matched to
its cost.

Protocol:
1. Run the LLM policy on a tuning subset (default 10 users x 20 sessions of tuning seed
   1000) and estimate its question cost relative to `always` on that same subset.
2. Tune every other family to that cost on the full tuning seeds (as in the main comparison).
3. Run never, always, the LLM policy and the tuned policies on a held-out subset
   (default 20 users x 20 sessions of seed 0).
4. If |LLM realized held-out cost - tuning estimate| > tol, flag every comparison
   involving the LLM and report the gap.

Example:
    python -m experiments.m3 --domain commute --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from simulator import core, matching, metrics, stats
from simulator.domains import get_domain
from simulator.impact_llm import load_impact_table
from simulator.llm import OllamaClient
from simulator.llm_policy import LLMPolicy
from simulator.policies import Always, Matched, Never, build_families

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", default="commute")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--prompt", choices=["plain", "guided"], default="plain",
                   help="per-session instruction (guided = step-by-step text)")
    p.add_argument("--with-info-gain", action="store_true",
                   help="also tune the info_gain baseline as an opponent")
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--llm-tune-seed", type=int, default=1000)
    p.add_argument("--llm-tune-users", type=int, default=10)
    p.add_argument("--tune-seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    p.add_argument("--tune-users", type=int, default=50)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--eval-users", type=int, default=20)
    p.add_argument("--llm-impact", type=Path, default=None)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def run(args: argparse.Namespace, client: OllamaClient | None = None) -> dict:
    """Execute the protocol; returns the tables written to disk."""
    domain = get_domain(args.domain)
    matching.check_disjoint(args.tune_seeds + [args.llm_tune_seed], [args.eval_seed])
    client = client or OllamaClient(args.model, cache_path=ROOT / "calibration" / "llm_cache.jsonl")
    llm = LLMPolicy(domain, client, getattr(args, "prompt", "plain"))

    # 1. LLM cost estimate on the tuning subset
    llm_tune = core.generate_timelines(domain, args.llm_tune_seed, args.llm_tune_users,
                                       args.sessions)
    estimate = matching.question_cost(domain, llm_tune, llm,
                                      matching.mean_questions(domain, llm_tune, Always()))

    # 2. tune the other families to that cost
    table = load_impact_table(args.llm_impact, domain) if args.llm_impact else None
    families = build_families(domain, 1.0, args.sessions, table,
                              include_info_gain=getattr(args, "with_info_gain", False))
    tuning = [tl for s in args.tune_seeds
              for tl in core.generate_timelines(domain, s, args.tune_users, args.sessions)]
    cals = [matching.calibrate(domain, tuning, fam, estimate, tol=args.tol)
            for fam in families.values()]

    # 3. held-out subset
    held = core.generate_timelines(domain, args.eval_seed, args.eval_users, args.sessions)
    policies = [Never(), Always(), Matched(llm, "llm", estimate)]
    policies += [p for c in cals if (p := c.build(families[c.family])) is not None]
    n_tune_calls = len(llm.calls)
    df = pd.DataFrame(core.run(domain, held, policies))
    summary = metrics.summarize(df)
    comps = stats.add_mcnemar(metrics.matched_comparisons(summary), df)

    # 4. cost drift check for the LLM
    realized = float(summary.set_index("policy").loc[f"llm@{estimate:g}", "question_cost"])
    drift = realized - estimate
    involves = comps.policy_a.str.startswith("llm@") | comps.policy_b.str.startswith("llm@")
    comps["llm_cost_drift"] = drift
    if abs(drift) > args.tol:
        comps.loc[involves, "flagged"] = True
        comps.loc[involves, "note"] = comps.loc[involves, "note"] + "; LLM cost drift > tol"

    calls = pd.DataFrame(llm.call_log())
    calls["phase"] = ["tune"] * n_tune_calls + ["held_out"] * (len(calls) - n_tune_calls)
    return {"summary": summary, "comparisons": comps, "calls": calls, "sessions": df,
            "calibration": pd.DataFrame([c.to_dict() for c in cals]),
            "facts": {"estimate": estimate, "realized": realized, "drift": drift,
                      "flag_drift": abs(drift) > args.tol, "model": client.model,
                      "llm_tune_size": f"{args.llm_tune_users} users x {args.sessions}",
                      "eval_size": f"{args.eval_users} users x {args.sessions}"}}


def main(argv: list[str] | None = None) -> int:
    """Run the per-session LLM study for one domain and write results/m3_<domain>/."""
    args = parse_args(argv)
    out = args.out or ROOT / "results" / f"m3_{args.domain}"
    out.mkdir(parents=True, exist_ok=True)
    res = run(args)
    for name in ("summary", "comparisons", "calls", "sessions", "calibration"):
        res[name].to_csv(out / f"{name}.csv", index=False)
    calls = res["calls"]
    fresh = calls[~calls.cached]
    facts = {**res["facts"], "calls": len(calls), "invalid": int((~calls.valid).sum()),
             "latency_mean_s": float(fresh.latency_s.mean()) if len(fresh) else 0.0,
             "prompt_tokens": int(calls.prompt_tokens.sum()),
             "completion_tokens": int(calls.completion_tokens.sum())}
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out / "config.json").write_text(json.dumps({**cfg, **facts}, indent=2))
    pd.set_option("display.width", 200)
    fmt = {"float_format": lambda x: f"{x:.3f}", "na_rep": "-", "index": False}
    print(json.dumps(facts, indent=2))
    print(res["summary"][["policy", "accuracy", "gap_closed", "question_cost",
                          "decision_precision", "reask_recall"]].to_string(**fmt))
    c = res["comparisons"]
    print(c[c.policy_a.str.startswith("llm@") | c.policy_b.str.startswith("llm@")][
        ["policy_a", "policy_b", "acc_a", "acc_b", "cost_a", "cost_b", "b", "c", "p_holm",
         "flagged", "note"]].to_string(**fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
