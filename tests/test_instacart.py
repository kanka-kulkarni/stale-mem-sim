"""Instacart analysis tests on a small synthetic dataset (no real data)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import instacart_rates as ir

DEPTS = pd.DataFrame({"department_id": [1, 2, 3, 4],
                      "department": ["produce", "meat seafood", "dairy eggs", "missing"]})
PRODUCTS = pd.DataFrame({
    "product_id": [1, 2, 3, 4, 5],
    "product_name": ["Banana", "Organic Spinach", "Chicken Breast", "Milk", "Mystery"],
    "aisle_id": [1, 1, 2, 3, 4], "department_id": [1, 1, 2, 3, 4]})


def write_dataset(root: Path, users: dict[int, list[dict]]) -> None:
    """users: user_id -> list of orders {dow, hour, gap, products}."""
    orders, contents, oid = [], [], 0
    for uid, seq in users.items():
        for i, o in enumerate(seq):
            oid += 1
            orders.append({"order_id": oid, "user_id": uid,
                           "eval_set": "train" if i == len(seq) - 1 else "prior",
                           "order_number": i + 1, "order_dow": o["dow"],
                           "order_hour_of_day": o["hour"],
                           "days_since_prior_order": np.nan if i == 0 else o["gap"]})
            for pos, pid in enumerate(o["products"]):
                contents.append({"order_id": oid, "product_id": pid,
                                 "add_to_cart_order": pos + 1, "reordered": 0,
                                 "set": orders[-1]["eval_set"]})
    pd.DataFrame(orders).to_csv(root / "orders.csv", index=False)
    c = pd.DataFrame(contents)
    for s in ("prior", "train"):
        c[c.set == s].drop(columns="set").to_csv(root / f"order_products__{s}.csv", index=False)
    PRODUCTS.to_csv(root / "products.csv", index=False)
    DEPTS.to_csv(root / "departments.csv", index=False)


def steady(n: int) -> list[dict]:
    return [{"dow": 2, "hour": 9, "gap": 7, "products": [1, 3, 4]} for _ in range(n)]


def switching(n: int) -> list[dict]:
    """One habit for the first half, another for the second half."""
    a = {"dow": 1, "hour": 9, "gap": 7, "products": [1, 2, 3]}
    b = {"dow": 5, "hour": 19, "gap": 25, "products": [4, 4, 4, 4, 4, 4, 4, 4, 5, 5]}
    return [a if i < n // 2 else b for i in range(n)]


class TestPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_dataset(root, {1: steady(40), 2: switching(40), 3: steady(10)})
        cls.data = ir.load(root, chunksize=7)  # tiny chunks exercise the chunked path
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_eligibility_and_aggregates(self):
        self.assertEqual((self.data.n_users_all, self.data.n_users_eligible), (3, 2))
        self.assertEqual(len(self.data.user), 80)
        first = slice(0, 40)
        self.assertTrue((self.data.n_items[first] == 3).all())
        self.assertTrue(self.data.has_meat[first].all())
        self.assertEqual(int(self.data.n_organic[40]), 1)  # switching user, habit a
        # habit b: 8 dairy products plus 2 from "missing", which is excluded
        self.assertEqual(int(self.data.n_items[79]), 10)
        self.assertEqual(self.data.dept[79].tolist(), [0, 0, 8, 0])

    def test_chunking_does_not_change_results(self):
        big = ir.load(self.root, chunksize=10_000)
        np.testing.assert_array_equal(big.dept, self.data.dept)
        np.testing.assert_array_equal(big.n_items, self.data.n_items)

    def test_steady_user_never_changes_and_switch_changes_once(self):
        _, per_user, _ = ir.change_counts(self.data, 5)
        for fact in ir.FACTS:
            self.assertEqual(per_user[fact][0], 0.0, fact)
        changed = {f for f in ir.FACTS if per_user[f][1] > 0}
        self.assertEqual(changed, {"order_day", "hour_band", "order_gap_band",
                                   "basket_size_band", "top_department", "buys_meat",
                                   "buys_organic"})
        # each switching fact changes exactly once across the 35 window pairs
        self.assertTrue(all(abs(per_user[f][1] - 1 / 35) < 1e-9 for f in ir.FACTS))

    def test_shuffling_breaks_persistence(self):
        table = ir.analyse(self.data, 5, n_shuffles=10).set_index("fact")
        # modes over balanced habits flip far more often once orders are shuffled; a
        # threshold fact can go either way (e.g. organic share reaches 20% only when all
        # five orders are from habit a, which shuffles rarely produce)
        for fact in ("order_day", "hour_band", "top_department"):
            self.assertGreater(table.loc[fact, "shuffled_mean"], table.loc[fact, "real_rate"])
        self.assertTrue(table.loc["order_day", "temporal_structure"])

    def test_mode_recency_tie_break(self):
        vals = np.array([0, 1, 0, 1])          # window of 4: tie 2-2, last seen is 1
        oh = ir.one_hot(vals, 2)
        self.assertEqual(ir.window_mode(oh, oh > 0, 4).tolist(), [1])
        vals = np.array([1, 0, 1, 0])
        oh = ir.one_hot(vals, 2)
        self.assertEqual(ir.window_mode(oh, oh > 0, 4).tolist(), [0])

    def test_hour_bands(self):
        self.assertEqual(ir.hour_band(np.array([5, 6, 11, 12, 16, 17, 21, 22, 0])).tolist(),
                         [3, 0, 0, 1, 1, 2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
