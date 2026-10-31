"""Paired statistics: exact McNemar test and Holm correction."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np
import pandas as pd

KEY = ["seed", "user", "session"]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B wrong)
    and c (the reverse)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = Fraction(sum(comb(n, i) for i in range(k + 1)), 2 ** n)
    return float(min(Fraction(1), 2 * tail))


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def discordant(df: pd.DataFrame, policy_a: str, policy_b: str) -> tuple[int, int, int]:
    """(b, c, n_paired) for two policies on their shared (seed, user, session) rows."""
    a = df[df.policy == policy_a].set_index(KEY).correct
    b = df[df.policy == policy_b].set_index(KEY).correct
    a, b = a.align(b, join="inner")
    return int((a & ~b).sum()), int((~a & b).sum()), len(a)


def add_mcnemar(comparisons: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add b, c, n_paired, p_mcnemar and Holm-adjusted p_holm (across all rows given)."""
    out = comparisons.copy()
    rows = [discordant(df[df.domain == r.domain], r.policy_a, r.policy_b)
            for r in out.itertuples()]
    out["b"] = [r[0] for r in rows]
    out["c"] = [r[1] for r in rows]
    out["n_paired"] = [r[2] for r in rows]
    out["p_mcnemar"] = [mcnemar_exact(b, c) for b, c, _ in rows]
    out["p_holm"] = holm(out.p_mcnemar.tolist()) if len(out) else []
    return out
