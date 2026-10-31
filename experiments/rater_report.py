"""Compare LLM impact raters: agreement with dependency lists and Spearman correlation
with exact impact, per task and per model.

Example:
    python -m experiments.rater_report --domains commute meals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import diagnostics
from simulator.domains import get_domain
from simulator.impact_llm import load_impact_table

ROOT = Path(__file__).resolve().parents[1]


def exact_reference(domain_name: str) -> pd.DataFrame:
    """Mean exact impact per (attribute, task) over the diagnostic reference states."""
    domain = get_domain(domain_name)
    return diagnostics.impact_table(domain, diagnostics.reference_states(domain))


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation (average ranks for ties); NaN if either is constant."""
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def rater_rows(domain_name: str, path: Path, exact: pd.DataFrame) -> list[dict]:
    """Per-task metrics for one ratings file."""
    domain = get_domain(domain_name)
    raw = pd.read_csv(path)
    model = raw.model.iloc[0]
    table = load_impact_table(path, domain)
    rows = []
    for task in domain.tasks:
        attrs = [a.name for a in domain.attributes]
        llm = pd.Series([table[(task.name, a)] for a in attrs], index=attrs)
        ex = exact[task.name].reindex(attrs)
        deps = set(task.depends_on)
        rated = {a for a in attrs if llm[a] > 0}
        rows.append({
            "domain": domain_name, "model": model, "task": task.name,
            "spearman_vs_exact": spearman(llm, ex),
            "dep_recall": len(deps & rated) / len(deps),
            "dep_precision": len(deps & rated) / len(rated) if rated else float("nan"),
            "missed_deps": ";".join(sorted(deps - rated)),
            "extra_attrs": len(rated - deps),
            "invalid": int((~raw[raw.task == task.name].valid).sum()),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    """Print per-task and per-model summaries; write calibration/rater_report.csv."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--domains", nargs="+", default=["commute", "meals"])
    args = p.parse_args(argv)
    rows = []
    for dom in args.domains:
        exact = exact_reference(dom)
        for path in sorted((ROOT / "calibration").glob(f"llm_impact_{dom}_*.csv")):
            rows.extend(rater_rows(dom, path, exact))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "calibration" / "rater_report.csv", index=False)
    pd.set_option("display.width", 200)
    fmt = {"float_format": lambda x: f"{x:.2f}", "na_rep": "n/a", "index": False}
    print("per task:\n" + df.to_string(**fmt))
    summary = df.groupby(["domain", "model"]).agg(
        spearman_mean=("spearman_vs_exact", "mean"),
        spearman_min=("spearman_vs_exact", "min"),
        dep_recall=("dep_recall", "mean"), dep_precision=("dep_precision", "mean"),
        missed_deps=("missed_deps", lambda s: int(sum(bool(x) for x in s))),
        invalid=("invalid", "sum")).reset_index()
    print("\nper model (means over tasks; missed_deps = tasks with at least one missed "
          "dependency):\n" + summary.to_string(**fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
