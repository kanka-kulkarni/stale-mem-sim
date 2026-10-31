"""End-to-end study: an LLM writes the answer from each policy's memory.

The answerer prompt is fixed; per-session LLM policies are replayed from the
cache only (a policy needing a new per-session call raises).

Subcommands:
    python -m experiments.e2e determinism            # repeat agreement
    python -m experiments.e2e estimate --domain commute
    python -m experiments.e2e run --domain commute    # answer and score
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import core, stats
from simulator.core import Domain, State, Task
from simulator.domains import get_domain
from simulator.llm import OllamaClient
from simulator.llm_policy import LLMPolicy
from simulator.matching import Calibration
from simulator.policies import AlwaysAll, Always, Never, Oracle, build_families

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "calibration" / "llm_cache.jsonl"
MODEL = "qwen2.5:7b"
EVAL_SEED, EVAL_USERS, SESSIONS = 0, 20, 20
MATCHED = (("sxi[exact]", 0.25), ("sxi[exact]", 0.5), ("sxi[coarse]", 0.25),
           ("sxi[coarse]", 0.5), ("ttl", 0.25), ("ttl", 0.5))
ANSWER_HEAD = ("You are a personal assistant. The user has made a request. "
               "Here is what you know about the user:")


# --------------------------------------------------------------------------- prompt


def answer_prompt(domain: Domain, values: State, task: Task) -> str:
    """The answerer prompt: facts without ages, request, options."""
    facts = "\n".join(f"{a.name}: {values[a.name].replace('_', ' ')}"
                      for a in domain.attributes)
    request = task.description or task.name.replace("_", " ")
    return (f"{ANSWER_HEAD}\n{facts}\nRequest: {request}\n"
            f"Choose the single best option for this user from this list: "
            f"{', '.join(task.options)}\n"
            'Reply with JSON only: {"answer": "<one option exactly as written>"}')


def parse_answer(text: str, options: tuple[str, ...]) -> str | None:
    """The chosen option, or None if the reply is invalid (scored as wrong)."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("answer"), str):
        return None
    return obj["answer"] if obj["answer"] in options else None


# --------------------------------------------------------------------------- policies


def _blocked(url, payload, timeout):
    raise RuntimeError("per-session LLM decision not in cache (would need a new call)")


def build_policies(domain_name: str, drop_50: bool = False,
                   only: list[str] | None = None) -> list:
    """never, always, always_all, oracle, the tuned matched settings from the main 1x run,
    and the plain and guided per-session LLMs replayed from the cache only."""
    domain = get_domain(domain_name)
    pols = [Never(), Always(), AlwaysAll(domain), Oracle(domain, SESSIONS, 1.0)]
    fams = build_families(domain, 1.0, SESSIONS)
    cal_path = ROOT / "results" / f"m2_{domain_name}" / "m1" / "calibration.json"
    cals = {(c["family"], c["target_cost"]): Calibration(**c)
            for c in json.loads(cal_path.read_text())}
    for fam, target in MATCHED:
        if drop_50 and target == 0.5:
            continue
        cal = cals.get((fam, target))
        if cal is not None and cal.kind != "unreachable":
            pols.append(cal.build(fams[fam]))
    cached_only = OllamaClient(MODEL, cache_path=CACHE, transport=_blocked)  # 7B decisions
    for style in ("plain", "guided"):
        pols.append(LLMPolicy(domain, cached_only, style))
    if only is not None:
        missing = set(only) - {p.name for p in pols}
        if missing:
            raise KeyError(f"unknown policies: {sorted(missing)}")
        pols = [p for p in pols if p.name in only]
    return pols


def records(domain_name: str, drop_50: bool = False,
            only: list[str] | None = None) -> pd.DataFrame:
    """Per-(policy, user, session) records with memory and true state."""
    domain = get_domain(domain_name)
    tls = core.generate_timelines(domain, EVAL_SEED, EVAL_USERS, SESSIONS)
    rows = core.run(domain, tls, build_policies(domain_name, drop_50, only),
                    record_memory=True)
    df = pd.DataFrame(rows)
    df["policy_prompt"] = [answer_prompt(domain, m, domain.task(k))
                           for m, k in zip(df["memory"], df.task)]
    df["reference_prompt"] = [answer_prompt(domain, s, domain.task(k))
                              for s, k in zip(df["true_state"], df.task)]
    return df


# --------------------------------------------------------------------------- metrics


def responsiveness(ref: pd.DataFrame, domain: Domain) -> dict:
    """Over consecutive same-task sessions where a dependency changed: how often the
    reference answer and the rule's answer change."""
    ref = ref.sort_values(["user", "session"])
    n = changed = rule_changed = 0
    for _, g in ref.groupby("user"):
        rows = g.to_dict("records")
        for a, b in zip(rows, rows[1:]):
            if a["task"] != b["task"]:
                continue
            deps = domain.task(a["task"]).depends_on
            if not any(a["true_state"][d] != b["true_state"][d] for d in deps):
                continue
            n += 1
            changed += a["ref_answer"] != b["ref_answer"]
            rule_changed += a["true_answer"] != b["true_answer"]
    return {"pairs": n, "reference_changes": changed / n if n else float("nan"),
            "rule_changes": rule_changed / n if n else float("nan")}


def evaluate(df: pd.DataFrame, domain: Domain) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Self-agreement, rule agreement, costs, McNemar/Holm, responsiveness, sway."""
    df = df.copy()
    df["self_agree"] = df.policy_answer.notna() & (df.policy_answer == df.ref_answer)
    df["rule_agree"] = df.policy_answer.notna() & (df.policy_answer == df.true_answer)
    always_q = df[df.policy == "always"].n_questions.mean()
    g = df.groupby("policy", sort=False)
    summ = g.agg(self_agreement=("self_agree", "mean"), rule_agreement=("rule_agree", "mean"),
                 sim_accuracy=("correct", "mean"), mean_questions=("n_questions", "mean"),
                 invalid=("policy_answer", lambda s: int(s.isna().sum())),
                 sessions=("self_agree", "size")).reset_index()
    summ["question_cost"] = summ.mean_questions / always_q
    base = summ.set_index("policy").self_agreement
    lo, hi = base["never"], base["always_all"]
    summ["gap_closed"] = (summ.self_agreement - lo) / (hi - lo) if hi != lo else float("nan")
    # paired exact McNemar on self-agreement, every pair of policies, Holm across all
    key = ["seed", "user", "session"]
    wide = df.pivot_table(index=key, columns="policy", values="self_agree", aggfunc="first")
    pols = list(summ.policy)
    comps = []
    for i, a in enumerate(pols):
        for b in pols[i + 1:]:
            x, y = wide[a].astype(bool), wide[b].astype(bool)
            nb, nc = int((x & ~y).sum()), int((~x & y).sum())
            comps.append({"policy_a": a, "policy_b": b, "acc_a": x.mean(), "acc_b": y.mean(),
                          "b": nb, "c": nc, "p_mcnemar": stats.mcnemar_exact(nb, nc)})
    comp = pd.DataFrame(comps)
    comp["p_holm"] = stats.holm(comp.p_mcnemar.tolist())
    ref = df[df.policy == "never"][["user", "session", "task", "true_state", "ref_answer",
                                    "true_answer"]]
    facts = {
        "reference_rule_agreement": float((ref.ref_answer == ref.true_answer).mean()),
        "reference_invalid": int(ref.ref_answer.isna().sum()),
        "irrelevant_fact_sway": float(base["always_all"] - base["always"]),
        **{f"responsiveness_{k}": v for k, v in responsiveness(ref, domain).items()},
    }
    return summ, comp, facts


# --------------------------------------------------------------------------- commands


def determinism(n_per_domain: int = 50) -> dict:
    """Repeat agreement: reference prompts sent twice each at temperature 0, cache bypassed."""
    prompts = []
    rng = random.Random(0)
    for dom in ("commute", "meals_calibrated"):
        domain = get_domain(dom)
        tls = core.generate_timelines(domain, EVAL_SEED, EVAL_USERS, SESSIONS)
        cells = [(tl, t) for tl in tls for t in range(1, SESSIONS + 1)]
        for tl, t in rng.sample(cells, n_per_domain):
            task = domain.task(tl.tasks[t])
            prompts.append((dom, task.options, answer_prompt(domain, tl.states[t], task)))
    first, second = OllamaClient(MODEL), OllamaClient(MODEL)  # no cache: real repeats
    same_parsed = same_raw = invalid = 0
    lat = []
    for _, options, prompt in prompts:
        r1, r2 = first.generate(prompt), second.generate(prompt)
        lat += [r1.latency_s, r2.latency_s]
        a1, a2 = parse_answer(r1.text, options), parse_answer(r2.text, options)
        same_parsed += a1 == a2
        same_raw += r1.text == r2.text
        invalid += (a1 is None) + (a2 is None)
    out = {"prompts": len(prompts), "repeat_agreement_parsed": same_parsed / len(prompts),
           "repeat_agreement_raw_text": same_raw / len(prompts), "invalid_replies": invalid,
           "latency_median_s": float(np.median(lat)), "model": MODEL}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "e2e_determinism.json").write_text(json.dumps(out, indent=2))
    return out


def estimate(domain_name: str, seconds_per_call: float, drop_50: bool = False,
             model: str = MODEL, only: list[str] | None = None) -> dict:
    """Distinct answerer prompts not yet cached, and the hours they would take."""
    df = records(domain_name, drop_50, only)
    distinct = set(df.policy_prompt) | set(df.reference_prompt)
    client = OllamaClient(model, cache_path=CACHE)
    new = [p for p in distinct if client.key(p) not in client.cache]
    return {"domain": domain_name, "records": len(df), "distinct_prompts": len(distinct),
            "uncached": len(new), "hours": len(new) * seconds_per_call / 3600,
            "drop_50": drop_50, "model": model, "policies": sorted(set(df.policy))}


def run(domain_name: str, drop_50: bool = False, model: str = MODEL,
        only: list[str] | None = None, out_name: str | None = None) -> dict:
    """Answer every distinct prompt (cached), score, write results/e2e_<domain>."""
    domain = get_domain(domain_name)
    df = records(domain_name, drop_50, only)
    client = OllamaClient(model, cache_path=CACHE)
    distinct = sorted(set(df.policy_prompt) | set(df.reference_prompt))
    replies, start = {}, time.time()
    for i, p in enumerate(distinct, 1):
        replies[p] = client.generate(p).text
        if i % 100 == 0:
            print(f"{i}/{len(distinct)} answerer prompts, {(time.time() - start) / 60:.0f} min",
                  flush=True)
    opts = {k.name: k.options for k in domain.tasks}
    df["policy_answer"] = [parse_answer(replies[p], opts[k])
                           for p, k in zip(df.policy_prompt, df.task)]
    df["ref_answer"] = [parse_answer(replies[p], opts[k])
                        for p, k in zip(df.reference_prompt, df.task)]
    summ, comp, facts = evaluate(df, domain)
    out = ROOT / "results" / (out_name or f"e2e_{domain_name}")
    out.mkdir(parents=True, exist_ok=True)
    keep = ["policy", "family", "seed", "user", "session", "task", "asked", "n_questions",
            "correct", "true_answer", "policy_answer", "ref_answer"]
    df[keep].to_csv(out / "answers.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    comp.to_csv(out / "comparisons.csv", index=False)
    facts.update(distinct_prompts=len(distinct), drop_50=drop_50, model=model,
                 policies=sorted(set(df.policy)))
    (out / "facts.json").write_text(json.dumps(facts, indent=2))
    return {"summary": summ, "comparisons": comp, "facts": facts}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["determinism", "estimate", "run"])
    p.add_argument("--domain", default="commute")
    p.add_argument("--seconds-per-call", type=float, default=6.0)
    p.add_argument("--drop-50", action="store_true")
    p.add_argument("--model", default=MODEL, help="answerer model")
    p.add_argument("--policies", nargs="*", default=None,
                   help="run only these policies (e.g. a subset for 14B)")
    p.add_argument("--out-name", default=None, help="results/<out-name> instead of e2e_<domain>")
    a = p.parse_args(argv)
    pd.set_option("display.width", 200)
    if a.command == "determinism":
        print(json.dumps(determinism(), indent=2))
    elif a.command == "estimate":
        print(json.dumps(estimate(a.domain, a.seconds_per_call, a.drop_50, a.model,
                                  a.policies), indent=2))
    else:
        res = run(a.domain, a.drop_50, a.model, a.policies, a.out_name)
        print(json.dumps(res["facts"], indent=2))
        print(res["summary"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
