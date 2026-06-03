"""Ablation extension test — same runnable-script style as tests/smoke_test.py.

Run from repo root:
    python tests/ablation_test.py

Exits 0 on success, nonzero with a clear message on any failure.
Requires data/processed/panel_phase2.csv to exist (run pipeline first).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ews.config import PATHS  # noqa: E402
from ews.eval import ablation_analysis  # noqa: E402


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# Load Phase 2 panel + simple time split (matches pipeline's TRAIN_END_YEAR=2020)
panel_path = os.path.join(PATHS.PROCESSED, "panel_phase2.csv")
assert os.path.isfile(panel_path), f"missing panel: {panel_path} — run pipeline first"
panel = pd.read_csv(panel_path)
train = panel[panel["year"] <= 2020]
test = panel[panel["year"] > 2020]

print("\n[1] ablation_analysis produces Filing only row")
rdf = ablation_analysis(train, test)
check(
    "Filing only group present",
    "Filing only" in rdf["Feature set"].values,
    f"feature sets: {list(rdf['Feature set'])}",
)
check(
    "Filing only uses exactly 1 feature",
    int(rdf.loc[rdf["Feature set"] == "Filing only", "N"].iloc[0]) == 1
    if "Filing only" in rdf["Feature set"].values else False,
)

print("\n[2] CSV persisted to outputs/")
csv_path = os.path.join(PATHS.OUTPUTS, "ablation_results.csv")
check("outputs/ablation_results.csv exists", os.path.isfile(csv_path), csv_path)
if os.path.isfile(csv_path):
    persisted = pd.read_csv(csv_path)
    check(
        "CSV row count matches in-memory result",
        len(persisted) == len(rdf),
        f"csv={len(persisted)} vs memory={len(rdf)}",
    )

print("\n[3] bootstrap CI columns present and well-formed")
check(
    "AUROC_lo column present",
    "AUROC_lo" in rdf.columns,
    f"columns: {list(rdf.columns)}",
)
check(
    "AUROC_hi column present",
    "AUROC_hi" in rdf.columns,
    f"columns: {list(rdf.columns)}",
)
if {"AUROC_lo", "AUROC_hi"}.issubset(rdf.columns):
    bracketed = ((rdf["AUROC_lo"] <= rdf["AUROC"]) & (rdf["AUROC"] <= rdf["AUROC_hi"])).all()
    check("each AUROC sits inside its own CI", bool(bracketed),
          rdf[["Feature set", "AUROC_lo", "AUROC", "AUROC_hi"]].to_string(index=False))
    nontrivial = ((rdf["AUROC_hi"] - rdf["AUROC_lo"]) > 0).all()
    check("CI widths are nonzero", bool(nontrivial))

print("\n" + "=" * 60)
if FAILURES:
    print(f"ABLATION TEST FAILED — {len(FAILURES)} assertion(s) failed:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ABLATION TEST PASSED")
    sys.exit(0)
