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


print("\n[2] panel has archetype + sector_raw columns after assembly")
panel_path = os.path.join(PATHS.PROCESSED, "panel_phase2.csv")
assert os.path.isfile(panel_path), f"missing panel: {panel_path} — run pipeline first"
panel = pd.read_csv(panel_path)
check(
    "archetype column present in panel_phase2.csv",
    "archetype" in panel.columns,
    f"got columns: {list(panel.columns)[-5:]}",  # tail of columns
)
check(
    "sector_raw column present in panel_phase2.csv",
    "sector_raw" in panel.columns,
)
if "archetype" in panel.columns:
    check(
        "no panel row has missing archetype",
        panel["archetype"].notna().all(),
        f"{panel['archetype'].isna().sum()} rows have NaN archetype",
    )
    check(
        "archetype values match the 7-bucket set",
        set(panel["archetype"].unique()).issubset({
            "Distressed", "Cyclical", "Stable", "Growth",
            "Defensive", "Rate-sensitive", "Commodity-sensitive",
        }),
        f"got: {sorted(panel['archetype'].unique())}",
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
