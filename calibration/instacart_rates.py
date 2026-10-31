"""Instacart: per-order change rates of order-history facts, on real and
shuffled order sequences.

Memory: only needed columns are loaded, with compact dtypes; orders.csv first;
order contents are read in chunks and reduced to per-order aggregates. Product names
are used only to derive the organic flag and are never written out. Outputs are
aggregate statistics only.

Usage:
    python -m calibration.instacart_rates --data path/to/instacart_csvs [--window 5]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_SESSIONS = 30
N_SHUFFLES = 10
FACTS = ("order_day", "hour_band", "order_gap_band", "basket_size_band", "top_department",
         "buys_meat", "buys_organic")
ORGANIC_SHARE = 0.20


@dataclass
class OrderData:
    """Per-order features for eligible users, sorted by (user, order_number)."""

    user: np.ndarray        # int32, one entry per order
    dow: np.ndarray         # int8 0..6
    hour_band: np.ndarray   # int8 0..3
    gap: np.ndarray         # float32 days since prior order (NaN for first order)
    n_items: np.ndarray     # int16
    has_meat: np.ndarray    # bool
    n_organic: np.ndarray   # int16
    dept: np.ndarray        # int16 [n_orders, n_departments] product counts
    n_users_all: int
    n_users_eligible: int


def hour_band(hour: np.ndarray) -> np.ndarray:
    """morning 6-11 -> 0, afternoon 12-16 -> 1, evening 17-21 -> 2, night 22-5 -> 3."""
    out = np.full(hour.shape, 3, dtype=np.int8)
    out[(hour >= 6) & (hour <= 11)] = 0
    out[(hour >= 12) & (hour <= 16)] = 1
    out[(hour >= 17) & (hour <= 21)] = 2
    return out


def csv_path(data_dir: Path, name: str) -> Path:
    """`name`.csv, or its zipped form `name`.csv.zip (read directly, never unzipped)."""
    for cand in (data_dir / f"{name}.csv", data_dir / f"{name}.csv.zip"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name}.csv[.zip] not found in {data_dir}")


def load(data_dir: Path, chunksize: int = 4_000_000) -> OrderData:
    """Load and reduce the Instacart CSVs to per-order features for eligible users."""
    orders = pd.read_csv(
        csv_path(data_dir, "orders"),
        usecols=["order_id", "user_id", "eval_set", "order_number", "order_dow",
                 "order_hour_of_day", "days_since_prior_order"],
        dtype={"order_id": np.int32, "user_id": np.int32, "eval_set": "category",
               "order_number": np.int16, "order_dow": np.int8,
               "order_hour_of_day": np.int8, "days_since_prior_order": np.float32})
    n_users_all = int(orders.user_id.nunique())
    orders = orders[orders.eval_set.isin(["prior", "train"])]
    counts = orders.groupby("user_id").size()
    eligible = counts.index[counts >= MIN_SESSIONS]
    orders = orders[orders.user_id.isin(eligible)].sort_values(["user_id", "order_number"])
    orders = orders.reset_index(drop=True)
    row_of = pd.Series(np.arange(len(orders), dtype=np.int32), index=orders.order_id.values)

    products = pd.read_csv(csv_path(data_dir, "products"),
                           usecols=["product_id", "product_name", "department_id"],
                           dtype={"product_id": np.int32, "department_id": np.int16})
    departments = pd.read_csv(csv_path(data_dir, "departments"))
    dept_ids = sorted(departments.department_id.tolist())
    col_of = {d: i for i, d in enumerate(dept_ids)}
    name_of = dict(zip(departments.department_id, departments.department))
    meat_id = next(d for d, n in name_of.items() if n == "meat seafood")
    max_pid = int(products.product_id.max())
    p_dept = np.full(max_pid + 1, -1, dtype=np.int16)
    p_dept[products.product_id.values] = [col_of[d] for d in products.department_id.values]
    p_meat = np.zeros(max_pid + 1, dtype=bool)
    p_meat[products.product_id.values] = products.department_id.values == meat_id
    p_organic = np.zeros(max_pid + 1, dtype=bool)
    p_organic[products.product_id.values] = products.product_name.str.contains(
        "organic", case=False, na=False).values
    missing_col = next((col_of[d] for d, n in name_of.items() if n == "missing"), None)
    del products

    n, k = len(orders), len(dept_ids)
    n_items = np.zeros(n, dtype=np.int32)
    has_meat = np.zeros(n, dtype=bool)
    n_organic = np.zeros(n, dtype=np.int32)
    dept = np.zeros((n, k), dtype=np.int32)
    for fname in ("order_products__prior", "order_products__train"):
        for chunk in pd.read_csv(csv_path(data_dir, fname), usecols=["order_id", "product_id"],
                                 dtype={"order_id": np.int32, "product_id": np.int32},
                                 chunksize=chunksize):
            rows = row_of.reindex(chunk.order_id.values).values
            keep = ~np.isnan(rows)
            r = rows[keep].astype(np.int64)
            pid = chunk.product_id.values[keep]
            n_items += np.bincount(r, minlength=n).astype(np.int32)
            n_organic += np.bincount(r, weights=p_organic[pid], minlength=n).astype(np.int32)
            meat_rows = r[p_meat[pid]]
            has_meat[meat_rows] = True
            d = p_dept[pid].astype(np.int64)
            known = d >= 0
            dept += np.bincount(r[known] * k + d[known],
                                minlength=n * k).reshape(n, k).astype(np.int32)
    if missing_col is not None:
        dept[:, missing_col] = 0
    return OrderData(
        user=orders.user_id.values, dow=orders.order_dow.values,
        hour_band=hour_band(orders.order_hour_of_day.values),
        gap=orders.days_since_prior_order.values.astype(np.float32),
        n_items=n_items.astype(np.int16), has_meat=has_meat,
        n_organic=n_organic.astype(np.int16), dept=dept.astype(np.int16),
        n_users_all=n_users_all, n_users_eligible=len(eligible))


# --------------------------------------------------------------------------- facts


def window_sums(x: np.ndarray, w: int) -> np.ndarray:
    """Sliding sums over the last w rows, for rows w-1..end (works on 1-D or 2-D)."""
    cs = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:], dtype=np.int64),
                                   x.astype(np.int64)]), axis=0)
    return cs[w:] - cs[:-w]


def window_mode(counts: np.ndarray, present: np.ndarray, w: int) -> np.ndarray:
    """Mode per window from per-row value counts [n, K]; ties go to the value that
    occurred most recently in the window, then to the lowest code."""
    n = counts.shape[0]
    wc = window_sums(counts, w)                                   # [n-w+1, K]
    idx = np.arange(n)[:, None]
    last = np.maximum.accumulate(np.where(present, idx, -1), axis=0)[w - 1:]
    t = np.arange(w - 1, n)[:, None]
    recency = np.where(wc > 0, last - (t - w), 0)                 # 1..w for present
    score = wc * (w + 1) + recency
    return score.argmax(axis=1)


def one_hot(values: np.ndarray, k: int) -> np.ndarray:
    """One-hot encode integer codes 0..k-1."""
    out = np.zeros((len(values), k), dtype=np.int8)
    out[np.arange(len(values)), values] = 1
    return out


def band(x: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    """0 if x <= edges[0], 1 if <= edges[1], else 2."""
    return np.where(x <= edges[0], 0, np.where(x <= edges[1], 1, 2))


def user_facts(f: dict[str, np.ndarray], w: int) -> dict[str, np.ndarray]:
    """Fact values for windows ending at each order from index w-1 onwards."""
    sw = np.lib.stride_tricks.sliding_window_view
    dow_oh = one_hot(f["dow"], 7)
    hb_oh = one_hot(f["hour_band"], 4)
    gaps = sw(f["gap"], w)
    with np.errstate(all="ignore"):
        gap_med = np.nanmedian(gaps, axis=1) if np.isfinite(gaps).any() else np.full(len(gaps), np.nan)
    size_med = np.median(sw(f["n_items"].astype(np.float32), w), axis=1)
    meat_count = window_sums(f["has_meat"].astype(np.int8), w)
    organic_share = window_sums(f["n_organic"], w) / np.maximum(window_sums(f["n_items"], w), 1)
    return {
        "order_day": window_mode(dow_oh, dow_oh > 0, w),
        "hour_band": window_mode(hb_oh, hb_oh > 0, w),
        "order_gap_band": np.where(np.isnan(gap_med), -1, band(gap_med, (8, 20))),
        "basket_size_band": band(size_med, (5, 15)),
        "top_department": window_mode(f["dept"], f["dept"] > 0, w),
        "buys_meat": (meat_count >= math.ceil(w / 2)).astype(np.int8),
        "buys_organic": (organic_share >= ORGANIC_SHARE).astype(np.int8),
    }


def change_counts(data: OrderData, w: int, shuffle_seed: int | None = None
                  ) -> tuple[dict[str, int], dict[str, list[float]], int]:
    """Pooled changes per fact, per-user change rates, and number of pairs."""
    changes = {k: 0 for k in FACTS}
    per_user = {k: [] for k in FACTS}
    pairs = 0
    bounds = np.flatnonzero(np.diff(data.user)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(data.user)]])
    for s, e in zip(starts, ends):
        f = {"dow": data.dow[s:e], "hour_band": data.hour_band[s:e], "gap": data.gap[s:e],
             "n_items": data.n_items[s:e], "has_meat": data.has_meat[s:e],
             "n_organic": data.n_organic[s:e], "dept": data.dept[s:e]}
        if shuffle_seed is not None:
            perm = np.random.default_rng([shuffle_seed, int(data.user[s])]).permutation(e - s)
            f = {k: v[perm] for k, v in f.items()}
        vals = user_facts(f, w)
        n_pairs = len(vals["order_day"]) - 1
        pairs += n_pairs
        for k in FACTS:
            c = int((vals[k][1:] != vals[k][:-1]).sum())
            changes[k] += c
            per_user[k].append(c / n_pairs)
    return changes, per_user, pairs


def analyse(data: OrderData, w: int, n_shuffles: int = N_SHUFFLES) -> pd.DataFrame:
    """Real vs shuffled change rates per fact, with the test for temporal structure."""
    real, per_user, pairs = change_counts(data, w)
    shuf = [change_counts(data, w, seed)[0] for seed in range(n_shuffles)]
    rows = []
    for k in FACTS:
        r = real[k] / pairs
        s = np.array([x[k] / pairs for x in shuf])
        outside = r < s.min() or r > s.max()
        rel = abs(r - s.mean()) / s.mean() if s.mean() > 0 else math.inf
        rows.append({"fact": k, "window": w, "real_rate": r,
                     "real_user_mean": float(np.mean(per_user[k])),
                     "real_user_median": float(np.median(per_user[k])),
                     "shuffled_mean": float(s.mean()), "shuffled_min": float(s.min()),
                     "shuffled_max": float(s.max()), "relative_diff": float(rel),
                     "direction": "below" if r < s.mean() else "above",
                     "temporal_structure": bool(outside and rel >= 0.10),
                     "pairs": pairs})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the analysis and write calibration/instacart_phase1_w<W>.csv plus a JSON summary."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--window", type=int, default=5)
    args = p.parse_args(argv)
    data = load(args.data)
    table = analyse(data, args.window)
    out = ROOT / "calibration" / f"instacart_phase1_w{args.window}.csv"
    table.to_csv(out, index=False)
    n_struct = int(table.temporal_structure.sum())
    facts = {"users_all": data.n_users_all, "users_30plus": data.n_users_eligible,
             "orders_eligible": int(len(data.user)), "window": args.window,
             "facts_with_temporal_structure": n_struct,
             "go": bool(n_struct >= 4) if args.window == 5 else None}
    (ROOT / "calibration" / f"instacart_phase1_w{args.window}.json").write_text(
        json.dumps(facts, indent=2))
    print(json.dumps(facts))
    print(table.drop(columns=["window", "pairs"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
