"""Threshold / flag-budget sensitivity analysis.

Uses evaluation artifacts produced by `scripts/fp_fn_analysis.py` to show how
held-out test recall/precision/FPR migrate as we change the alert budget.

If artifacts are missing, the page explains how to regenerate them.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Threshold Sensitivity", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG = REPO_ROOT / "outputs" / "figures" / "phase3_fp_fn_frontier.png"
OP_CSV = REPO_ROOT / "outputs" / "fp_fn_operating_points.csv"
COST_CSV = REPO_ROOT / "outputs" / "fp_fn_cost_sensitivity.csv"
FINDINGS_MD = REPO_ROOT / "outputs" / "fp_fn_findings.md"


def _missing(path: Path) -> bool:
    if not path.exists():
        st.error(f"⚠️ `{path.name}` not found. Run `MPLBACKEND=Agg python scripts/fp_fn_analysis.py` to generate it.")
        return True
    return False


st.title("⚖️ Threshold & Flag-Budget Sensitivity")

st.markdown(
    """
Explore how changing the alert budget (share of firm-months flagged) affects
the held-out test confusion matrix. Use the preset budgets (30%, 40%, 50%) or
enter a custom value.
"""
)

st.markdown("---")

# Show frontier figure if available
if not _missing(FIG):
    try:
        st.image(Image.open(FIG), use_container_width=True)
    except Exception:
        st.info("Frontier figure exists but could not be opened as an image.")

st.markdown("---")

st.header("Quick lookup: common flag budgets")
budgets = st.multiselect("Pick flag budgets to inspect (fractions)", options=[0.30, 0.40, 0.50], default=[0.30, 0.40, 0.50])
custom = st.number_input("Or enter a custom flag budget (0-1)", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
if custom not in budgets:
    budgets = budgets + [custom]

if _missing(OP_CSV) and _missing(COST_CSV):
    st.stop()

# Prefer COST_CSV (gives a dense mapping); fall back to OP_CSV if needed
data = None
key_col = None
display_cols = None
try:
    if COST_CSV.exists():
        # allow leading comment lines in CSVs
        data = pd.read_csv(COST_CSV, comment="#")
        key_col = "test_flag_budget"
        display_cols = ["cost_ratio_fn_fp", "thr", "val_flag_budget", "test_flag_budget", "test_recall", "test_precision", "test_fpr"]
    elif OP_CSV.exists():
        data = pd.read_csv(OP_CSV, comment="#")
        key_col = "flag_budget"
        display_cols = ["operating_point", "thr", "flag_budget", "recall", "precision", "fpr"]
    else:
        data = None
except Exception as e:
    st.error(f"Error reading artifact CSV: {e}")
    data = None

if data is None or data.empty:
    st.info("No FP/FN artifact data available to inspect — generate with `scripts/fp_fn_analysis.py` and commit the outputs.")
    st.stop()

# Ensure the key column is present and numeric
if key_col not in data.columns:
    st.warning(f"expected column `{key_col}` not found in artifact; available columns: {data.columns.tolist()}")
    st.stop()

# coerce numeric columns for safe comparisons
data[key_col] = pd.to_numeric(data[key_col], errors="coerce")

st.header("Results")
rows = []
for b in sorted(set(budgets)):
    # find nearest available entry in data by absolute difference in key_col
    if key_col not in data.columns:
        st.warning(f"expected column `{key_col}` not found in artifact; skipping budget {b}.")
        continue

    series = data[key_col].dropna()
    if series.empty:
        st.warning(f"No numeric values found in `{key_col}` to match budgets.")
        continue

    # nearest match
    try:
        idx = (series - float(b)).abs().idxmin()
        row = data.loc[idx]
        rows.append((float(b), row))
    except Exception as e:
        st.warning(f"Could not match budget {b}: {e}")
        continue

if rows:
    # Build display DataFrame
    out_rows = []
    for requested_b, row in rows:
        if key_col == "test_flag_budget":
            out_rows.append({
                "requested_budget": requested_b,
                "matched_budget": row["test_flag_budget"],
                "thr": row["thr"],
                "test_recall": row["test_recall"],
                "test_precision": row["test_precision"],
                "test_fpr": row["test_fpr"],
            })
        else:
            out_rows.append({
                "requested_budget": requested_b,
                "matched_budget": row["flag_budget"],
                "thr": row["thr"],
                "test_recall": row["recall"],
                "test_precision": row["precision"],
                "test_fpr": row["fpr"],
            })

    df_out = pd.DataFrame(out_rows)
    st.dataframe(df_out, hide_index=True)
else:
    st.info("No matching rows found for the selected budgets in the available artifacts.")

st.markdown("---")
st.header("Notes & reproducibility")
st.write("These tables are sourced from committed artifacts produced by `scripts/fp_fn_analysis.py` and saved to `outputs/`. To refresh metrics after retraining, run:\n\n`MPLBACKEND=Agg python scripts/fp_fn_analysis.py`\n\nthen commit the resulting `outputs/` files.")

if FINDINGS_MD.exists():
    md = FINDINGS_MD.read_text()
    if "## Headline findings" in md:
        body = md.split("## Headline findings", 1)[1].split("## Reproducibility", 1)[0]
        st.markdown(body)
    else:
        st.markdown(md)

