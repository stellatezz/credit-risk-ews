"""One-off parser: extract firm categories from data/List of sample company.

Reads the Phase 2 markdown doc, walks every | ... | Ticker | ... | Category |
Purpose | row, and writes data/firm_categories.csv with columns
(ticker, sector_raw, archetype, purpose).

`archetype` is normalised to one of seven buckets via case-insensitive
substring match against (Category + Purpose). See the plan
docs/superpowers/plans/2026-06-05-category-sector-error-analysis.md
for the keyword → bucket mapping.

Run:
    python scripts/extract_firm_categories.py

Idempotent — overwrites the CSV.
"""

from __future__ import annotations

import csv
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO_ROOT, "data", "List of sample company")
OUT = os.path.join(REPO_ROOT, "data", "firm_categories.csv")


def normalise_archetype(category: str, purpose: str) -> str:
    """Map (category, purpose) free-text labels to one of seven buckets."""
    blob = f"{category} {purpose}".lower()
    rules: list[tuple[list[str], str]] = [
        (["distress", "restructur", "bankrupt"], "Distressed"),
        (["rate-sensitive", "interest-rate", "real estate"], "Rate-sensitive"),
        (["commodity"], "Commodity-sensitive"),
        (["growth"], "Growth"),
        (["defensive"], "Defensive"),
        (["cyclical"], "Cyclical"),
    ]
    for fragments, bucket in rules:
        if any(f in blob for f in fragments):
            return bucket
    return "Stable"


def parse_rows(doc_text: str) -> list[dict]:
    """Walk the markdown doc, yielding one dict per company row.

    A company row has the shape:
        | <num> | <TICKER> | <Company> | <Category> | <Purpose> |
    Skips header / separator rows by requiring TICKER to look like
    1-5 uppercase letters.
    """
    rows: list[dict] = []
    ticker_re = re.compile(r"^[A-Z]{1,5}$")
    for line in doc_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        # Expected columns: No., Ticker, Company, Category, Purpose
        _num, ticker, _company, category, purpose = cells[0], cells[1], cells[2], cells[3], cells[4]
        if not ticker_re.match(ticker):
            continue
        sector_raw = category.split("/")[0].strip()
        rows.append({
            "ticker": ticker,
            "sector_raw": sector_raw,
            "archetype": normalise_archetype(category, purpose),
            "purpose": purpose,
        })
    return rows


def main() -> int:
    if not os.path.isfile(DOC):
        print(f"ERROR: source doc not found: {DOC}", file=sys.stderr)
        return 1
    with open(DOC) as f:
        rows = parse_rows(f.read())
    if not rows:
        print("ERROR: parser produced 0 rows; check doc format", file=sys.stderr)
        return 1
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "sector_raw", "archetype", "purpose"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
