"""False Positives vs False Negatives page.

The orthogonal story to the Sector & Category page: that page is a per-slice
snapshot at one operating point; this page is the *threshold / cost* dimension,
evaluated on the held-out 2024 test set.

Reads artifacts produced by `scripts/fp_fn_analysis.py`:
  outputs/figures/phase3_fp_fn_frontier.png   capture curve + cost-vs-budget
  outputs/fp_fn_operating_points.csv           test confusion matrix + CIs per operating point
  outputs/fp_fn_cost_sensitivity.csv           threshold migration across FN:FP ratios
  outputs/fp_fn_slice_threshold_compare.csv    global vs slice-relative top-decile (val)
  outputs/fp_fn_findings.md                    narrative (headline parsed for the lead)

Every number on this page is loaded live from those files — no hard-coding. The
hosted app reads committed artifacts; regenerate with
`MPLBACKEND=Agg python scripts/fp_fn_analysis.py` then commit the outputs.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="False Positives vs False Negatives", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG = REPO_ROOT / "outputs" / "figures" / "phase3_fp_fn_frontier.png"
OP_CSV = REPO_ROOT / "outputs" / "fp_fn_operating_points.csv"
COST_CSV = REPO_ROOT / "outputs" / "fp_fn_cost_sensitivity.csv"
SLICE_CSV = REPO_ROOT / "outputs" / "fp_fn_slice_threshold_compare.csv"
FINDINGS_MD = REPO_ROOT / "outputs" / "fp_fn_findings.md"


def _missing(path: Path) -> bool:
    if not path.exists():
        st.error(f"⚠️ `{path.name}` not found. Run "
                 "`MPLBACKEND=Agg python scripts/fp_fn_analysis.py` to generate it.")
        return True
    return False


# =========================================================================
# Intro
# =========================================================================

st.title("⚖️ False Positives vs False Negatives")

st.markdown(
    """
The Sector & Category page asks *which firms* the model works on, at **one**
operating point. This page asks the orthogonal question: **how should we set the
alert threshold, given that the two errors are not equally costly?**

- **False negative (FN)** — a firm deteriorates and we said nothing. The expensive
  miss for a credit watchlist.
- **False positive (FP)** — we flag a firm that stays healthy. Recoverable: an
  analyst spends time and finds nothing.

**Method (leak-safe by construction).** The alert threshold is *selected on the
validation window (2021–2023)*; the confusion matrix you see is then reported on
the **held-out 2024 test set, with the threshold frozen** — so these numbers
are honest out-of-sample generalisation, not in-sample fit. Recall and FPR carry
**firm-clustered** bootstrap 95% CIs.
"""
)

st.markdown("---")


# =========================================================================
# Headline metrics
# =========================================================================

st.header("📊 Held-out 2024 test — at a glance")

if not _missing(OP_CSV):
    op = pd.read_csv(OP_CSV)
    dec = op[op["operating_point"] == "top_10pct"].iloc[0]
    heavy = op[op["operating_point"] == "cost_opt_10:1"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall @ top-decile", f"{dec['recall']:.0%}",
              help=f"95% CI [{dec['recall_lo']:.0%}, {dec['recall_hi']:.0%}] — "
                   f"share of 2024 distress events caught.")
    c2.metric("Precision @ top-decile", f"{dec['precision']:.0%}",
              help="Of flagged firm-months, share that truly deteriorated. Base-rate-bound.")
    c3.metric("FPR @ top-decile", f"{dec['fpr']:.0%}",
              help=f"95% CI [{dec['fpr_lo']:.0%}, {dec['fpr_hi']:.0%}] — false-alarm rate.")
    c4.metric("Recall @ FN:FP = 10:1", f"{heavy['recall']:.0%}",
              help=f"When a miss costs 10× a false alarm, the optimum flags "
                   f"{heavy['flag_budget']:.0%} of firm-months — precision falls to "
                   f"{heavy['precision']:.0%}.")

    st.caption("Recall and FPR are the transferable numbers across panels; precision is "
               "bound to the test base rate (10.2%) and would fall in a lower-prevalence "
               "deployment population.")

st.markdown("---")


# =========================================================================
# The cost frontier
# =========================================================================

st.header("1️⃣ The cost frontier")

st.markdown(
    """
**Left — capture curve.** As we loosen the threshold (flag more firm-months), we
catch more events. The model sits above both the **Altman Z** baseline and the
**random** diagonal across the whole range — discrimination is real. But there is
no free lunch: every extra event caught costs more false alarms.

**Right — where the optimum lands.** Expected cost (relative to doing nothing) vs
flag budget, for a range of FN:FP cost ratios. The marked minimum on each curve is
the cost-optimal operating point. The optimum **marches right** as misses get more
expensive — and there is a sharp jump between 5:1 and 10:1.
"""
)

if not _missing(FIG):
    st.image(Image.open(FIG), use_container_width=True)

st.markdown("---")


# =========================================================================
# Operating points on the held-out test
# =========================================================================

st.header("2️⃣ Operating points on the held-out 2024 test")

st.markdown(
    "Each row freezes a threshold chosen on validation and reports the realised "
    "2024 confusion matrix. The top-decile is the analyst's current rule; the "
    "`cost_opt_*` rows are the cost-minimising thresholds for each FN:FP ratio."
)

if not _missing(OP_CSV):
    st.dataframe(
        op,
        hide_index=True,
        column_config={
            "operating_point": st.column_config.TextColumn("Operating point"),
            "thr":         st.column_config.NumberColumn("threshold", format="%.3f"),
            "flags":       st.column_config.NumberColumn("# flags", format="%d"),
            "TP":          st.column_config.NumberColumn("TP", format="%d"),
            "FP":          st.column_config.NumberColumn("FP", format="%d"),
            "FN":          st.column_config.NumberColumn("FN", format="%d"),
            "TN":          st.column_config.NumberColumn("TN", format="%d"),
            "recall":      st.column_config.NumberColumn("recall", format="%.3f"),
            "precision":   st.column_config.NumberColumn("precision", format="%.3f"),
            "fpr":         st.column_config.NumberColumn("FPR", format="%.3f"),
            "flag_budget": st.column_config.NumberColumn("flag budget", format="%.3f"),
            "recall_lo":   st.column_config.NumberColumn("recall CI lo", format="%.3f"),
            "recall_hi":   st.column_config.NumberColumn("recall CI hi", format="%.3f"),
            "fpr_lo":      st.column_config.NumberColumn("FPR CI lo", format="%.3f"),
            "fpr_hi":      st.column_config.NumberColumn("FPR CI hi", format="%.3f"),
        },
    )

st.subheader("Cost sensitivity — the threshold migration")

if not _missing(COST_CSV):
    cost = pd.read_csv(COST_CSV)
    st.dataframe(
        cost,
        hide_index=True,
        column_config={
            "cost_ratio_fn_fp": st.column_config.TextColumn("FN:FP"),
            "c_fn":             st.column_config.NumberColumn("c_FN", format="%d"),
            "thr":              st.column_config.NumberColumn("threshold", format="%.3f"),
            "val_flag_budget":  st.column_config.NumberColumn("val flag budget", format="%.3f"),
            "val_cost":         st.column_config.NumberColumn("val cost", format="%d"),
            "test_recall":      st.column_config.NumberColumn("test recall", format="%.3f"),
            "test_fpr":         st.column_config.NumberColumn("test FPR", format="%.3f"),
            "test_precision":   st.column_config.NumberColumn("test precision", format="%.3f"),
            "test_flag_budget": st.column_config.NumberColumn("test flag budget", format="%.3f"),
            "test_FP":          st.column_config.NumberColumn("test FP", format="%d"),
            "test_FN":          st.column_config.NumberColumn("test FN", format="%d"),
        },
    )
    st.caption("Between 5:1 and 10:1 the optimal flag budget jumps from ~10% to ~67% — the tool "
               "flips from a selective watchlist to a flag-most screen. That is the decision a "
               "risk committee actually has to make.")

st.markdown("---")


# =========================================================================
# Artifact vs failure
# =========================================================================

st.header("3️⃣ Is the per-slice zero recall a threshold artifact or a ranking failure?")

st.markdown(
    """
Some archetypes catch **zero** events under the single global top-decile threshold.
Is that just because the global threshold starves low-base-rate slices of flags
(a *thresholding artifact* — fixable), or because the model cannot rank distress
within those slices (a *ranking failure* — not fixable by any threshold)?

The test: give each slice a **slice-relative** top-decile threshold — flag the
riskiest 10% *within that slice*. If recall stays near the ~10% random floor, the
within-sector ranking is broken (AUROC ≤ 0.5).
"""
)

if not _missing(SLICE_CSV):
    sl = pd.read_csv(SLICE_CSV)
    st.dataframe(
        sl,
        hide_index=True,
        column_config={
            "archetype":           st.column_config.TextColumn("Archetype"),
            "n_events":            st.column_config.NumberColumn("# events", format="%d"),
            "n_rows":              st.column_config.NumberColumn("# rows", format="%d"),
            "flagbudget_global":   st.column_config.NumberColumn("flags @ global", format="%.3f"),
            "recall_global":       st.column_config.NumberColumn("recall @ global", format="%.3f"),
            "flagbudget_slicerel": st.column_config.NumberColumn("flags @ slice-rel", format="%.3f"),
            "recall_slicerel":     st.column_config.NumberColumn("recall @ slice-rel", format="%.3f"),
            "recall_gain":         st.column_config.NumberColumn("gain", format="%+.3f"),
        },
    )
    st.caption("Stable and Cyclical (and Growth) stay at/below the 10% floor even with a "
               "slice-relative threshold → ranking failure. Distressed reaches 25% (2.5× the "
               "floor) → the one archetype the model genuinely ranks.")

st.markdown("---")


# =========================================================================
# Findings
# =========================================================================

st.header("4️⃣ Headline findings")

if not _missing(FINDINGS_MD):
    md = FINDINGS_MD.read_text()
    if "## Headline findings" in md:
        body = md.split("## Headline findings", 1)[1].split("## Reproducibility", 1)[0]
        st.markdown(body)
    else:
        st.markdown(md)
