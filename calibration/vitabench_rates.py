"""Derive per-category preference change rates from VitaBench 2.0.

Data (not committed, ~131 MB; no license is stated on the dataset card, so only
derived counts are kept here):
    curl -L -o vb2_tasks.json \
      https://huggingface.co/datasets/meituan-longcat/VitaBench-2.0/resolve/main/tasks.json

Usage:
    python -m calibration.vitabench_rates --tasks path/to/vb2_tasks.json

Each user has a sequence of subtasks. Every subtask's
user_scenario.personalized_preference_memory holds `current` (preference tags by
category) and `preference_tag_change_history`, which lists each tag with its status
relative to the previous subtask: unchanged / create / update / delete. A change
entry is assigned to the category whose `current` list contains it (this subtask's
for create/update, the previous subtask's for delete). One subtask transition is
treated as one simulator session.

Rates per category (transitions = subtask pairs where the category exists):
- per_tag_rate = (update + delete) / (tags held at the start of each transition),
  i.e. P(a given stored tag is revised or dropped per session). This is the analogue
  of the simulator's per-attribute change probability (one attribute ~ one tag).
- p_category_changed = P(at least one update/delete in the category per transition).
- per_tag_rate_with_creates adds new tags (create) as a sensitivity check.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {  # VitaBench category -> English label
    "饮食偏好": "food",
    "消费购物偏好": "shopping_spending",
    "出行住宿偏好": "travel_lodging",
    "休闲娱乐偏好": "leisure",
    "其他特殊偏好": "other",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(k: int, n: int) -> float:
    return k / n if n else math.nan


def _tags(cur: dict, cat: str) -> list[str]:
    v = cur.get(cat, [])
    return v if isinstance(v, list) else ([v] if v else [])


def _category(tag: str, *currents: dict) -> str | None:
    for cur in currents:
        for cat in cur:
            if tag in _tags(cur, cat):
                return cat
    return None


def derive(tasks: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Per-category counts and rates, plus dataset-level facts."""
    ev: collections.Counter = collections.Counter()
    touched: collections.Counter = collections.Counter()
    transitions: collections.Counter = collections.Counter()
    tag_exposure: collections.Counter = collections.Counter()
    unmapped, gaps = 0, []
    for user in tasks:
        subs = user["subtasks"]
        for prev, cur in zip(subs, subs[1:]):
            pc = prev["user_scenario"]["personalized_preference_memory"]["current"]
            cc = cur["user_scenario"]["personalized_preference_memory"]["current"]
            gaps.append((dt.date.fromisoformat(cur["current_time"][:10])
                         - dt.date.fromisoformat(prev["current_time"][:10])).days)
            for cat in set(pc) | set(cc):
                transitions[cat] += 1
                tag_exposure[cat] += len(_tags(pc, cat))
            hit = set()
            hist = cur["user_scenario"]["personalized_preference_memory"].get(
                "preference_tag_change_history", [])
            for e in hist:
                if e["type"] not in ("create", "update", "delete"):
                    continue
                cat = (_category(e["content"], pc) if e["type"] == "delete"
                       else _category(e["content"], cc)) or _category(e["content"], cc, pc)
                if cat is None:
                    unmapped += 1
                    continue
                ev[(cat, e["type"])] += 1
                if e["type"] != "create":
                    hit.add(cat)
            for cat in hit:
                touched[cat] += 1
    rows = []
    for cat, label in CATEGORIES.items():
        up, de, cr = ev[(cat, "update")], ev[(cat, "delete")], ev[(cat, "create")]
        n_tags, n_tr = tag_exposure[cat], transitions[cat]
        lo, hi = wilson(up + de, n_tags)
        rows.append({
            "category": label, "category_zh": cat, "transitions": n_tr,
            "tag_exposure": n_tags, "update": up, "delete": de, "create": cr,
            "per_tag_rate": _ratio(up + de, n_tags), "per_tag_ci_low": lo,
            "per_tag_ci_high": hi, "p_category_changed": _ratio(touched[cat], n_tr),
            "per_tag_rate_with_creates": _ratio(up + de + cr, n_tags),
        })
    gaps.sort()
    facts = {"users": len(tasks), "subtasks": sum(len(u["subtasks"]) for u in tasks),
             "transitions": len(gaps), "gap_days_median": gaps[len(gaps) // 2],
             "gap_days_mean": sum(gaps) / len(gaps), "unmapped_entries": unmapped}
    return pd.DataFrame(rows), facts


def main(argv: list[str] | None = None) -> int:
    """Write calibration/vitabench_rates.csv and print the table."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "calibration" / "vitabench_rates.csv")
    args = p.parse_args(argv)
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    df, facts = derive(tasks)
    df.to_csv(args.out, index=False)
    print(json.dumps(facts))
    print(df.drop(columns="category_zh").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
