"""Do the real SEC fundamentals rescue the failing slices?

The aggregate ablation says adding fundamentals HURTS overall AUROC (Full 0.55
vs Sector-rel 0.71), so the deployed model drops them. But the aggregate is
dominated by Distressed/Growth events; fundamentals could still lift the slices
the deployed model is blind to (Stable, Cyclical) while hurting the ones the
market already nails. This tests that per-slice.

Compares per-archetype AUROC (firm-clustered 95% CI, reusing eval.evaluate_by_slice)
on a common row set for three feature sets:
  - deployed   : Market + sector-relative          (no fundamentals)
  - +acct      : deployed + 5 accounting ratios     (marginal value of fundamentals)
  - full       : every feature                       (FEATURE_COLS)

Eval split: validation 2021-2023 (per-slice; the 2024 test is too thin to slice).

Run:  python scripts/fundamentals_slice_test.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ews.config import (  # noqa: E402
    FEATURE_COLS,
    LABEL_COL,
    MARKET_PLUS_REL_FEATURE_COLS,
    MARKET_REL_FEATURE_COLS,
    PATHS,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
)
from ews.eval import evaluate_by_slice  # noqa: E402

ACCT = ["leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability"]

FEATURE_SETS = {
    "deployed": MARKET_PLUS_REL_FEATURE_COLS,
    "+acct": MARKET_PLUS_REL_FEATURE_COLS + ACCT,
    "full": FEATURE_COLS,
}

# slices the deployed model fails on (from category_sector_findings.md)
FAILING = {"Stable", "Cyclical", "Growth", "Rate-sensitive"}


def main() -> None:
    df = pd.read_csv(os.path.join(PATHS.PROCESSED, "panel_phase2.csv"), parse_dates=["date"])
    df["year"] = df["date"].dt.year

    # Common support: drop rows with any NaN in the union of features, so all
    # three feature sets are scored on identical rows (fair comparison).
    keep = sorted(set(FEATURE_COLS + MARKET_PLUS_REL_FEATURE_COLS + MARKET_REL_FEATURE_COLS + ACCT))
    df = df.dropna(subset=keep)

    train = df[df["year"] <= TRAIN_END_YEAR]
    val = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= VAL_END_YEAR)]
    print(f"Common-support rows — train: {len(train)} | val: {len(val)} "
          f"({val[LABEL_COL].sum():.0f} events, {val['ticker'].nunique()} firms)\n")

    # Per-archetype AUROC for each feature set
    per_set = {}
    for name, cols in FEATURE_SETS.items():
        print("=" * 70)
        print(f"FEATURE SET: {name}  ({len(cols)} features)")
        print("=" * 70)
        rdf = evaluate_by_slice(train, val, "archetype", cols, LABEL_COL)
        per_set[name] = rdf.set_index("slice")

    # Build the comparison table
    base = per_set["deployed"]
    rows = []
    for arch in base.index:
        r = {
            "archetype": arch,
            "n_events": int(base.loc[arch, "n_events"]),
            "AUROC_deployed": base.loc[arch, "AUROC"],
            "AUROC_+acct": per_set["+acct"].loc[arch, "AUROC"] if arch in per_set["+acct"].index else float("nan"),
            "AUROC_full": per_set["full"].loc[arch, "AUROC"] if arch in per_set["full"].index else float("nan"),
        }
        r["delta_acct"] = r["AUROC_+acct"] - r["AUROC_deployed"]
        r["delta_full"] = r["AUROC_full"] - r["AUROC_deployed"]
        r["failing_slice"] = arch in FAILING
        rows.append(r)

    comp = pd.DataFrame(rows).sort_values("n_events", ascending=False)

    print("\n" + "=" * 70)
    print("PER-ARCHETYPE AUROC — does adding fundamentals help? (val 2021-2023)")
    print("=" * 70)
    print(comp.round(3).to_string(index=False))

    # Verdict on the failing slices
    print("\n--- VERDICT on the failing slices (deployed AUROC <= 0.5) ---")
    fails = comp[comp["failing_slice"] & (comp["AUROC_deployed"] <= 0.52)]
    if fails.empty:
        print("  (no failing slice met the <=0.52 criterion)")
    for _, r in fails.iterrows():
        best = max(r["AUROC_+acct"], r["AUROC_full"])
        if best > 0.55:
            msg = f"RESCUED -> best {best:.3f} (fundamentals help here)"
        elif best > r["AUROC_deployed"] + 0.03:
            msg = f"nudged -> best {best:.3f} (small lift, still weak)"
        else:
            msg = f"NOT rescued -> best {best:.3f} (fundamentals don't fix the ranking)"
        print(f"  {r['archetype']:14s} deployed {r['AUROC_deployed']:.3f} | "
              f"+acct {r['AUROC_+acct']:.3f} | full {r['AUROC_full']:.3f}  =>  {msg}")

    out = os.path.join(PATHS.OUTPUTS, "fundamentals_slice_test.csv")
    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    comp.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
