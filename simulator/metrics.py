"""Aggregate metrics over per-session records.

Headline metrics (per domain, per policy):
- accuracy: per-session exact match with the true-state answer.
- gap_closed = (acc - acc_never) / (acc_always - acc_never).
- question_cost = mean questions / mean questions of `always`.
- staleness_precision: fraction of asked facts whose memory value was outdated.
- decision_precision: fraction of asked facts that were answer-changing stale facts.
- reask_recall: fraction of answer-changing stale facts that were re-asked.

"Answer-changing" uses the single-fact test in `core.answer_changing_stale`. Its two
blind spots are counted per (domain, policy) in `edge_cases`.

Secondary: mean_questions, stale_error_rate, and marginal accuracy per question
= (acc - acc_never) / mean questions (N/A for `never`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HEADLINE = ["accuracy", "gap_closed", "question_cost", "staleness_precision",
            "decision_precision", "reask_recall"]
SECONDARY = ["mean_questions", "stale_error_rate", "marginal_acc_per_question"]
EDGE = ["joint_only_n", "joint_only_frac", "cancelling_n", "cancelling_frac"]
ID = ["domain", "policy", "family", "role", "target_cost"]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise num/den with NaN where den is 0 or missing."""
    return num.astype(float) / den.replace(0, np.nan)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (domain, policy) of headline and secondary metrics, pooled over
    sessions."""
    df = df.assign(
        joint_only=~df.pre_ask_correct & (df.n_answer_changing == 0),
        cancelling=df.pre_ask_correct & (df.n_answer_changing > 0),
    )
    g = df.groupby(["domain", "policy"], sort=False)
    out = g.agg(
        family=("family", "first"),
        role=("role", "first"),
        target_cost=("target_cost", "first"),
        accuracy=("correct", "mean"),
        mean_questions=("n_questions", "mean"),
        stale_error_rate=("stale_error", "mean"),
        asked=("n_questions", "sum"),
        asked_stale=("n_asked_stale", "sum"),
        changing=("n_answer_changing", "sum"),
        changing_asked=("n_answer_changing_asked", "sum"),
        joint_only_n=("joint_only", "sum"),
        joint_only_frac=("joint_only", "mean"),
        cancelling_n=("cancelling", "sum"),
        cancelling_frac=("cancelling", "mean"),
        n_sessions=("correct", "size"),
    ).reset_index()

    def baseline(policy: str, col: str) -> pd.Series:
        ref = out[out.policy == policy].set_index("domain")[col]
        return out["domain"].map(ref).astype(float)

    acc_never, acc_always = baseline("never", "accuracy"), baseline("always", "accuracy")
    out["gap_closed"] = _ratio(out.accuracy - acc_never, acc_always - acc_never)
    out["question_cost"] = _ratio(out.mean_questions, baseline("always", "mean_questions"))
    out["staleness_precision"] = _ratio(out.asked_stale, out.asked)
    out["decision_precision"] = _ratio(out.changing_asked, out.asked)
    out["reask_recall"] = _ratio(out.changing_asked, out.changing)
    out["marginal_acc_per_question"] = _ratio(out.accuracy - acc_never, out.mean_questions)
    return out[[*ID, *HEADLINE, *SECONDARY, *EDGE, "n_sessions"]]


def by_session(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and mean questions per session index, one row per (domain, policy, session)."""
    return (
        df.groupby(["domain", "policy", "session"], sort=False)
        .agg(accuracy=("correct", "mean"), mean_questions=("n_questions", "mean"))
        .reset_index()
    )


def macro_average(summary: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean of per-domain metrics for each policy."""
    return summary.groupby("policy", sort=False)[HEADLINE + SECONDARY].mean().reset_index()


def at_matched_cost(summary: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    """Accuracy and gap closed of each family at the given costs, interpolated along its
    sweep (figures only)."""
    rows = []
    s = summary[summary.role == "sweep"].dropna(subset=["question_cost"])
    for (domain, fam), g in s.groupby(["domain", "family"], sort=False):
        g = g.groupby("question_cost", as_index=False)[["accuracy", "gap_closed"]].mean()
        x = g.question_cost.to_numpy()
        for lvl in levels:
            inside = x.min() - 1e-9 <= lvl <= x.max() + 1e-9
            row = {"domain": domain, "family": fam, "question_cost": lvl}
            for col in ("accuracy", "gap_closed"):
                row[col] = float(np.interp(lvl, x, g[col].to_numpy())) if inside else np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=["domain", "family", "question_cost", "accuracy",
                                       "gap_closed"])


def local_slope(points: list[tuple[float, float]], cost: float) -> float:
    """Slope d(accuracy)/d(cost) of a family's piecewise-linear curve at `cost`."""
    pts = pd.DataFrame(points, columns=["c", "a"]).groupby("c", as_index=False).a.mean()
    if len(pts) < 2:
        return float("nan")
    x, y = pts.c.to_numpy(), pts.a.to_numpy()
    i = int(np.clip(np.searchsorted(x, cost) - 1, 0, len(x) - 2))
    return float((y[i + 1] - y[i]) / (x[i + 1] - x[i]))


COMPARISON_COLS = [
    "domain", "target_cost", "policy_a", "policy_b", "cost_a", "cost_b", "acc_a", "acc_b",
    "leader", "lead", "leader_costlier", "cost_diff", "leader_slope", "lead_from_cost",
    "flagged", "note",
]


def matched_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    """Paired tests between matched policies at the same target cost, flagging leads
    that the winner's extra questions could explain."""
    held = summary.dropna(subset=["question_cost"])
    matched = held[held.role == "matched"]
    rows = []
    for (domain, target), g in matched.groupby(["domain", "target_cost"], sort=False):
        g = g.sort_values("family").to_dict("records")
        for i, a in enumerate(g):
            for b in g[i + 1:]:
                if a["family"] == b["family"]:
                    continue
                row = {"domain": domain, "target_cost": target, "policy_a": a["policy"],
                       "policy_b": b["policy"], "cost_a": a["question_cost"],
                       "cost_b": b["question_cost"], "acc_a": a["accuracy"],
                       "acc_b": b["accuracy"], "leader": None, "lead": 0.0,
                       "leader_costlier": False, "cost_diff": b["question_cost"] - a["question_cost"],
                       "leader_slope": np.nan, "lead_from_cost": np.nan,
                       "flagged": False, "note": "tie"}
                if a["accuracy"] != b["accuracy"]:
                    lead_p, other = (a, b) if a["accuracy"] > b["accuracy"] else (b, a)
                    diff = lead_p["question_cost"] - other["question_cost"]
                    row.update(leader=lead_p["policy"], lead=lead_p["accuracy"] - other["accuracy"],
                               leader_costlier=diff > 0, cost_diff=diff, note="leader not costlier")
                    if diff > 0:
                        fam = held[(held.domain == domain) & (held.family == lead_p["family"])
                                   & (held.policy != lead_p["policy"])]
                        slope = local_slope(list(zip(fam.question_cost, fam.accuracy)),
                                            lead_p["question_cost"])
                        row["leader_slope"] = slope
                        if np.isnan(slope):
                            row.update(flagged=True, note="no slope available")
                        else:
                            row["lead_from_cost"] = slope * diff
                            ok = row["lead"] > row["lead_from_cost"]
                            row.update(flagged=not ok,
                                       note="lead exceeds cost effect" if ok
                                       else "lead within cost effect")
                rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLS)
