"""Descriptive, per-task analysis of the end-to-end answerer's reference answers (the
answers given the TRUE state). Uses existing outputs only: results/e2e_<domain>/
answers.csv plus true states regenerated deterministically from the seeds. Makes no
model calls.

For each task:
- rule agreement: reference answer == rule's answer on the true state;
- responsiveness (consecutive): over consecutive sessions of a user with the same task
  where at least one dependency fact changed, how often the reference answer changes
  (and, for context, how often the rule's answer changes);
- responsiveness (all pairs): over all pairs of sessions with that task (any users)
  whose dependency values differ, how often the reference answers differ, vs the rule;
- answer distribution: reference answers and rule answers per option.

Example:
    python -m experiments.e2e_descriptive --domain commute
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import pandas as pd

from experiments.e2e import EVAL_SEED, EVAL_USERS, SESSIONS
from simulator import core
from simulator.domains import get_domain

ROOT = Path(__file__).resolve().parents[1]


def reference_table(domain_name: str) -> pd.DataFrame:
    """One row per (user, session): task, reference answer, rule answer, dependency values."""
    domain = get_domain(domain_name)
    ans = pd.read_csv(ROOT / "results" / f"e2e_{domain_name}" / "answers.csv")
    ref = ans[ans.policy == "never"][["user", "session", "task", "ref_answer", "true_answer"]]
    tls = {tl.user: tl for tl in core.generate_timelines(domain, EVAL_SEED, EVAL_USERS, SESSIONS)}
    for r in ref.itertuples():  # pairing check: regenerated timeline matches the outputs
        if tls[r.user].tasks[r.session] != r.task:
            raise RuntimeError("regenerated timeline does not match the recorded answers")
    ref = ref.copy()
    ref["deps"] = [tuple(tls[u].states[s][d] for d in domain.task(k).depends_on)
                   for u, s, k in zip(ref.user, ref.session, ref.task)]
    return ref.sort_values(["user", "session"]).reset_index(drop=True)


def entropy(counts: pd.Series) -> float:
    """Shannon entropy in bits of an answer distribution."""
    p = counts / counts.sum()
    return float(-(p[p > 0] * p[p > 0].map(math.log2)).sum())


def per_task(ref: pd.DataFrame, domain_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-task summary and per-task answer distributions."""
    domain = get_domain(domain_name)
    rows, dist = [], []
    for task in domain.tasks:
        g = ref[ref.task == task.name]
        if g.empty:
            continue
        cons = cons_ref = cons_rule = 0
        for _, u in g.groupby("user"):
            recs = u.sort_values("session").to_dict("records")
            for a, b in zip(recs, recs[1:]):
                if b["session"] != a["session"] + 1 or a["deps"] == b["deps"]:
                    continue
                cons += 1
                cons_ref += a["ref_answer"] != b["ref_answer"]
                cons_rule += a["true_answer"] != b["true_answer"]
        allp = all_ref = all_rule = 0
        for a, b in combinations(g.to_dict("records"), 2):
            if a["deps"] == b["deps"]:
                continue
            allp += 1
            all_ref += a["ref_answer"] != b["ref_answer"]
            all_rule += a["true_answer"] != b["true_answer"]
        ref_counts = g.ref_answer.fillna("<invalid>").value_counts()
        rule_counts = g.true_answer.value_counts()
        rows.append({
            "domain": domain_name, "task": task.name, "sessions": len(g),
            "n_options": len(task.options), "deps": len(task.depends_on),
            "rule_agreement": float((g.ref_answer == g.true_answer).mean()),
            "consecutive_pairs": cons,
            "consecutive_ref_changes": cons_ref / cons if cons else float("nan"),
            "consecutive_rule_changes": cons_rule / cons if cons else float("nan"),
            "all_pairs": allp,
            "allpairs_ref_differs": all_ref / allp if allp else float("nan"),
            "allpairs_rule_differs": all_rule / allp if allp else float("nan"),
            "ref_distinct_answers": int(ref_counts.size),
            "ref_top_answer_share": float(ref_counts.iloc[0] / len(g)),
            "ref_entropy_bits": entropy(ref_counts),
            "rule_distinct_answers": int(rule_counts.size),
            "rule_entropy_bits": entropy(rule_counts),
        })
        for opt in task.options:
            dist.append({"domain": domain_name, "task": task.name, "option": opt,
                         "reference": int(ref_counts.get(opt, 0)),
                         "rule": int(rule_counts.get(opt, 0))})
        if "<invalid>" in ref_counts:
            dist.append({"domain": domain_name, "task": task.name, "option": "<invalid>",
                         "reference": int(ref_counts["<invalid>"]), "rule": 0})
    return pd.DataFrame(rows), pd.DataFrame(dist)


def main(argv: list[str] | None = None) -> int:
    """Write per_task.csv and answer_distribution.csv for one domain and print them."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domain", required=True)
    a = p.parse_args(argv)
    out = ROOT / "results" / f"e2e_{a.domain}"
    summ, dist = per_task(reference_table(a.domain), a.domain)
    summ.to_csv(out / "per_task.csv", index=False)
    dist.to_csv(out / "answer_distribution.csv", index=False)
    pd.set_option("display.width", 220)
    fmt = {"index": False, "float_format": lambda x: f"{x:.2f}"}
    print(summ.drop(columns=["domain"]).to_string(**fmt))
    nz = dist[(dist.reference > 0) | (dist.rule > 0)]
    print("\nanswer distribution (options used by either):\n" + nz.drop(columns="domain").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
