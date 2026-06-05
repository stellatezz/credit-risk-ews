"""Category/sector analysis test — runnable script, mirrors tests/smoke_test.py.

Run from repo root:
    python tests/category_sector_test.py

Exits 0 on success, nonzero with a clear message on any failure.
Requires data/processed/panel_phase2.csv (run pipeline first).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ews.config import FIRMS, PATHS  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


print("\n[1] firm_categories.csv exists and covers every configured ticker")
cat_path = os.path.join(REPO_ROOT, "data", "firm_categories.csv")
check("data/firm_categories.csv exists", os.path.isfile(cat_path), cat_path)
if os.path.isfile(cat_path):
    cats = pd.read_csv(cat_path)
    expected_cols = {"ticker", "sector_raw", "archetype", "purpose"}
    check(
        "schema has ticker, sector_raw, archetype, purpose",
        expected_cols.issubset(set(cats.columns)),
        f"got {sorted(cats.columns)}",
    )
    configured = set(FIRMS.keys())
    in_csv = set(cats["ticker"].astype(str))
    missing = configured - in_csv
    check(
        "every ticker in config.FIRMS has a category row",
        not missing,
        f"missing: {sorted(missing)}" if missing else "",
    )
    valid_archetypes = {
        "Distressed", "Cyclical", "Stable", "Growth",
        "Defensive", "Rate-sensitive", "Commodity-sensitive",
    }
    bad = set(cats["archetype"]) - valid_archetypes
    check(
        "every archetype is one of the 7 buckets",
        not bad,
        f"unexpected: {sorted(bad)}" if bad else "",
    )


print("\n" + "=" * 60)
if FAILURES:
    print(f"CATEGORY/SECTOR TEST FAILED — {len(FAILURES)} assertion(s) failed:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("CATEGORY/SECTOR TEST PASSED")
    sys.exit(0)
