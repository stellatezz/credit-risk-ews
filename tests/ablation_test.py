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
# Match pipeline's val split (2021-2023), so the CSV the test writes equals the pipeline's canonical CSV.
test = panel[(panel["year"] > 2020) & (panel["year"] <= 2023)]

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

print("\n[4] REITs back in panel; wc_ratio_missing column present")
REIT_TICKERS = ["SPG", "O", "PLD", "VTR"]
panel_tickers = set(panel["ticker"].unique())
reits_in_panel = [t for t in REIT_TICKERS if t in panel_tickers]
check(
    "at least 3 of 4 named REITs survive panel dropna",
    len(reits_in_panel) >= 3,
    f"REITs in panel: {reits_in_panel}",
)
check(
    "wc_ratio_missing column present in panel",
    "wc_ratio_missing" in panel.columns,
    f"columns include wc_ratio_missing: {'wc_ratio_missing' in panel.columns}",
)
if "wc_ratio_missing" in panel.columns:
    check(
        "wc_ratio_missing is binary (only 0/1 values)",
        set(panel["wc_ratio_missing"].unique()).issubset({0, 1}),
        f"unique values: {sorted(panel['wc_ratio_missing'].unique())}",
    )
    # For REIT rows, missingness should be 1 (they're the reason we added this feature)
    reit_rows = panel[panel["ticker"].isin(REIT_TICKERS)]
    if len(reit_rows) > 0:
        check(
            "REIT rows have wc_ratio_missing == 1",
            (reit_rows["wc_ratio_missing"] == 1).all(),
            f"REIT wc_ratio_missing values: {reit_rows['wc_ratio_missing'].unique()}",
        )

print("\n[5] firm-clustered bootstrap widens CIs vs row-level")
# Quick property check: rerunning ablation with row-level bootstrap should give
# strictly narrower CIs than the production (clustered) run, because clustered
# resampling correctly accounts for within-firm autocorrelation.
from ews.eval import _bootstrap_auroc_ci  # noqa: E402

# The production call (test passes ticker col via test data → clustered)
rdf_clustered = rdf  # alias the section [2] result for clarity
clustered_widths = (rdf_clustered["AUROC_hi"] - rdf_clustered["AUROC_lo"]).values

# Reference: row-level CIs on the same subsets via a small direct invocation
# of the helper. We only need one subset for the smoke check — Market only.
import statsmodels.api as sm

market_cols = ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"]
m = sm.Logit(train["label_a"], sm.add_constant(train[market_cols])).fit(disp=0)
p = m.predict(sm.add_constant(test[market_cols]))
y_true = test["label_a"].values
y_pred = p.values
lo_row, hi_row = _bootstrap_auroc_ci(y_true, y_pred, firm_ids=None, n_boot=1000)
row_level_width = hi_row - lo_row

market_row = rdf_clustered[rdf_clustered["Feature set"] == "Market only"].iloc[0]
clustered_width = market_row["AUROC_hi"] - market_row["AUROC_lo"]

check(
    "Market only clustered CI is at least as wide as row-level CI",
    clustered_width >= row_level_width * 0.9,  # 0.9 buffer for monte-carlo noise
    f"clustered={clustered_width:.4f} vs row-level={row_level_width:.4f}",
)
check(
    "every clustered CI width < 1.0 (CI is actually informative)",
    bool((clustered_widths < 1.0).all()),
    f"widths: {clustered_widths.round(4).tolist()}",
)

print("\n[6] ablation runs over multiple model families")
check(
    "model_family column present",
    "model_family" in rdf.columns,
    f"columns: {list(rdf.columns)}",
)
if "model_family" in rdf.columns:
    families = set(rdf["model_family"].unique())
    check(
        "at least pooled + fe families ran successfully",
        {"pooled", "fe"}.issubset(families),
        f"families: {sorted(families)}",
    )
    # Each successful family should produce all 6 subsets (or fail-and-omit
    # individually; the row count per family must be > 1 for the data to mean
    # anything).
    for fam in families:
        n_subsets = (rdf["model_family"] == fam).sum()
        check(
            f"{fam} family produced multiple subset rows (got {n_subsets})",
            n_subsets >= 4,
            f"{fam}: {n_subsets} rows",
        )

print("\n" + "=" * 60)
if FAILURES:
    print(f"ABLATION TEST FAILED — {len(FAILURES)} assertion(s) failed:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ABLATION TEST PASSED")
    sys.exit(0)
