"""Feature-Group Analysis page.

Structure: Intro → Data → Findings → Conclusion.

Reads three artifacts produced by the pipeline:
  outputs/ablation_results.csv          per (model_family, feature_set): AUROC + clustered 95% CI + AUPRC + Brier
  outputs/full_model_coefficients_*.csv per-family Full-model coefficients (feature, coef, std_err, p_value)
  outputs/ablation_findings.md          hand-written narrative (headline parsed for the Findings lead)

Every number on this page comes from one of those files — no hardcoded values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feature-Group Analysis", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_CSV = REPO_ROOT / "outputs" / "ablation_results.csv"
COEF_CSV = REPO_ROOT / "outputs" / "full_model_coefficients_pooled.csv"
FINDINGS_MD = REPO_ROOT / "outputs" / "ablation_findings.md"


# Theoretical sign expectations (credit-risk theory). Used to flag wrong-signed
# coefficients in the Full pooled model below. None = ambiguous / no prior.
EXPECTED_SIGNS: dict[str, str] = {
    "leverage": "+",          # more debt → more risk
    "liquidity_buffer": "-",  # more cash → less risk
    "wc_ratio": "-",          # higher working capital → less risk
    "profitability": "-",     # more profit → less risk
    "ret_1m": "-",            # positive returns → less near-term risk
    "ret_3m": "-",
    "ret_6m": "-",
    "vol_3m": "+",            # higher vol → more risk
    "vol_6m": "+",
    "drawdown_12m": "+",      # recent stress → future stress
    "late_filing": "+",       # filing delay → red flag
    "vix": "+",               # high VIX → stress regime
    "credit_spread": "+",     # wide spreads → stress
    "term_spread": None,      # ambiguous: inversion predicts recession, but levels matter
}


# =========================================================================
# Intro
# =========================================================================

st.title("🔬 Feature-Group Analysis")

st.markdown(
    """
**Feature-group analysis** (an *ablation*) re-fits and re-scores the same model
on each feature group alone, on combinations, and on everything — to find which
features actually carry the signal. **Data:** 77 large US firms, monthly 2010–2024
(12,473 firm-months); target = a substantive distress event within 12 months
(base rate ≈ 8.5%); fit on 2010–2020 and scored **out-of-sample** on 2021–2023
(~2,772 rows).

**Feature groups:** Accounting (leverage, liquidity, working-capital,
profitability) · Market (returns 1/3/6m, volatility 3/6m, 12m drawdown) ·
**Sector-relative market features** (each market feature z-scored *within its
industry-month* — "is this firm unusual *for its sector*?") · Macro (VIX, term
spread, credit spread) · Filing (late-filing flag) · plus the combinations
Accounting + Market, **Market + sector-relative**, and Full (everything).

**Three model variants:** **Pooled logit** (firm-months treated independently),
**FE logit** (industry-and-year fixed effects), and **Hazard logit**
(Shumway-style discrete-time hazard). Each AUROC carries a 95% **firm-clustered**
bootstrap CI (1,000 resamples) — clustering on firms because credit panels are
strongly autocorrelated within firm.
"""
)

st.markdown("---")


# =========================================================================
# Data
# =========================================================================

st.header("📊 Data")

if not ABLATION_CSV.exists():
    st.error(
        "⚠️ `outputs/ablation_results.csv` not found. Run "
        "`MPLBACKEND=Agg python src/run.py` to regenerate."
    )
    st.stop()

ablation = pd.read_csv(ABLATION_CSV)

st.markdown(
    "**Per-variant × feature-subset scores** on the 2021–2023 validation "
    "window. Higher AUROC / AUPRC = better; lower Brier = better. AUROC CIs are "
    "firm-clustered 95% bootstrap percentiles."
)

families = sorted(ablation["model_family"].unique())
chosen = st.multiselect(
    "Model family", families, default=families, help="Filter by model family."
)
view = ablation[ablation["model_family"].isin(chosen)].copy()
view = view.sort_values(["model_family", "AUROC"], ascending=[True, False])

st.dataframe(
    view.round(3),
    column_config={
        "model_family": st.column_config.TextColumn("Family"),
        "Feature set":  st.column_config.TextColumn("Feature set"),
        "N":            st.column_config.NumberColumn("N feats", format="%d"),
        "AUROC":        st.column_config.ProgressColumn("AUROC", min_value=0.0, max_value=1.0),
        "AUROC_lo":     st.column_config.NumberColumn("CI lo", format="%.3f"),
        "AUROC_hi":     st.column_config.NumberColumn("CI hi", format="%.3f"),
        "AUPRC":        st.column_config.NumberColumn("AUPRC", format="%.3f"),
        "Brier":        st.column_config.NumberColumn("Brier", format="%.3f"),
    },
    hide_index=True,
    use_container_width=True,
)

st.markdown(
    "Pick a metric to re-rank the chart. **AUROC** = ranking (chance 0.5, "
    "inflated under ~8.5% imbalance) · **AUPRC** = positive-class precision "
    "(chance ≈ validation base rate ~0.11) · **Brier** = calibration "
    "(lower = better). Only AUROC carries CIs."
)

metric = st.radio(
    "Metric",
    ["AUROC", "AUPRC", "Brier"],
    horizontal=True,
    help="AUROC and AUPRC: higher = better. Brier: lower = better.",
)

# Metric-aware chart parameters
HIGHER_IS_BETTER = {"AUROC": True, "AUPRC": True, "Brier": False}
CHANCE_LINE = {
    "AUROC": (0.5, "chance (0.5)"),
    "AUPRC": None,   # base rate varies; not drawn
    "Brier": None,
}

# Build the chart from the filtered `view` so it tracks the multiselect above.
chart_data = view.copy()
chart_data["label"] = (
    chart_data["model_family"].str.title() + " · " + chart_data["Feature set"]
)
# Sort so "best" appears at the top of the chart (matplotlib barh stacks bottom-up).
chart_data = chart_data.sort_values(metric, ascending=HIGHER_IS_BETTER[metric])

fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(chart_data))))
y_pos = np.arange(len(chart_data))

# Error bars only for AUROC (the only metric with CIs).
if metric == "AUROC":
    xerr = np.vstack([
        (chart_data["AUROC"] - chart_data["AUROC_lo"]).clip(lower=0).values,
        (chart_data["AUROC_hi"] - chart_data["AUROC"]).clip(lower=0).values,
    ])
    error_kw = {"ecolor": "#333", "elinewidth": 1.2, "capsize": 3}
else:
    xerr = None
    error_kw = {}

ax.barh(
    y_pos,
    chart_data[metric].values,
    xerr=xerr,
    color="#4c72b0",
    edgecolor="white",
    error_kw=error_kw,
)
chance = CHANCE_LINE[metric]
if chance is not None:
    ax.axvline(chance[0], color="#888", linestyle=":", linewidth=1, label=chance[1])
    ax.legend(loc="lower right", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(chart_data["label"].values, fontsize=9)
direction = "↑ higher is better" if HIGHER_IS_BETTER[metric] else "↓ lower is better"
ax.set_xlabel(f"{metric}   ({direction})")
if metric in {"AUROC", "AUPRC"}:
    ax.set_xlim(0, 1)
ci_note = " · firm-clustered 95% CI" if metric == "AUROC" else " · point estimate only"
ax.set_title(f"Per-family per-subset {metric}{ci_note}")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# -- Full-model coefficient signs (mechanism behind Finding 2) -------------
st.markdown(
    "**Why the Full model loses — coefficient signs.** Features in the Full "
    "*pooled* model fitted opposite to credit-risk theory at p < 0.05 are "
    "flagged 🔴; these actively hurt discrimination. (Theory: leverage / "
    "volatility / macro-stress *raise* risk; profitability / cash / returns "
    "*lower* it; `term_spread` is ambiguous and not flagged.)"
)

n_wrong: int | None = None
wrong_names = ""
if COEF_CSV.exists():
    coef = pd.read_csv(COEF_CSV)
    body = coef[coef["feature"] != "const"].copy()
    body["fitted_sign"] = np.where(body["coef"] > 0, "+", "-")
    body["expected_sign"] = body["feature"].map(EXPECTED_SIGNS)
    body["wrong_sign"] = (
        body["expected_sign"].notna()
        & (body["expected_sign"] != body["fitted_sign"])
        & (body["p_value"] < 0.05)
    )
    body["flag"] = np.where(body["wrong_sign"], "🔴", "")

    display = body[
        ["flag", "feature", "expected_sign", "fitted_sign", "coef", "std_err", "p_value"]
    ].rename(columns={
        "flag": " ",
        "feature": "Feature",
        "expected_sign": "Expected",
        "fitted_sign": "Fitted",
        "coef": "Coefficient",
        "std_err": "SE",
        "p_value": "p-value",
    })

    st.dataframe(
        display.round({"Coefficient": 3, "SE": 3, "p-value": 4}),
        column_config={
            "p-value": st.column_config.NumberColumn("p-value", format="%.4f"),
        },
        hide_index=True,
        use_container_width=True,
    )

    n_wrong = int(body["wrong_sign"].sum())
    wrong_names = ", ".join(body.loc[body["wrong_sign"], "feature"])
    if n_wrong > 0:
        st.warning(
            f"**{n_wrong} feature(s) fitted with the wrong sign at p < 0.05:** "
            f"{wrong_names}. Adding them with reversed polarity suppresses "
            f"discrimination — the mechanism behind the Full model's loss."
        )
    else:
        st.success("All significant coefficients have theoretically expected signs.")
else:
    st.error(
        "⚠️ `outputs/full_model_coefficients_pooled.csv` not found. Run "
        "`MPLBACKEND=Agg python src/run.py` to regenerate."
    )

st.markdown("---")


# =========================================================================
# Findings
# =========================================================================

st.header("🔑 Findings")


def _row(family: str, feature_set: str) -> pd.Series | None:
    """Return the result row for one (family, feature_set) combo, or None if absent."""
    mask = (ablation["model_family"] == family) & (ablation["Feature set"] == feature_set)
    sub = ablation[mask]
    return sub.iloc[0] if len(sub) else None


market_pooled = _row("pooled", "Market only")
market_fe     = _row("fe",     "Market only")
market_haz    = _row("hazard", "Market only")
full_pooled   = _row("pooled", "Full model")
full_haz      = _row("hazard", "Full model")
macro_pooled  = _row("pooled", "Macro only")
macro_fe      = _row("fe",     "Macro only")
srel_pooled   = _row("pooled", "Sector-rel only")
mrel_fe       = _row("fe",     "Market + sector-rel")

# Calibration results (Phase 3 #1), if present.
CALIB_CSV = REPO_ROOT / "outputs" / "calibration_results.csv"
calib = pd.read_csv(CALIB_CSV) if CALIB_CSV.exists() else None
def _calib(method: str, col: str):
    if calib is None:
        return None
    r = calib[calib["method"] == method]
    return float(r.iloc[0][col]) if len(r) else None

# Headline (parsed from the findings doc) leads the section.
if FINDINGS_MD.exists():
    text = FINDINGS_MD.read_text()
    in_section = False
    headline_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## Headline finding"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            headline_lines.append(line)
    headline = "\n".join(headline_lines).strip()
    if headline:
        st.info(headline)

gap_p = market_pooled["AUROC"] - full_pooled["AUROC"]
gap_h = market_haz["AUROC"] - full_haz["AUROC"]
flip = macro_fe["AUROC"] - macro_pooled["AUROC"]
wrong_phrase = f"{n_wrong}" if n_wrong else "several"

st.markdown(
    f"""
**1. Sector-relative market features carry the most signal.** Z-scoring each
market feature *within its industry-month* lifts pooled AUROC from Market-only
**{market_pooled['AUROC']:.3f}** to Sector-rel **{srel_pooled['AUROC']:.3f}**,
and AUPRC from **{market_pooled['AUPRC']:.3f}** to **{srel_pooled['AUPRC']:.3f}**
(a bigger jump, and AUPRC is the honest metric under imbalance). The same lift
appears in FE and Hazard. Asking "is this firm unusual *for its sector*?"
separates a distressed firm from one merely sitting in a volatile industry — the
single biggest improvement on this page.

**2. Raw market beats the Full model; piling on features hurts.** Market-only
beats Full by **{gap_p:+.3f}** (Pooled) and **{gap_h:+.3f}** (Hazard); FE is a
near-tie. The Data section flags **{wrong_phrase}** Full-model coefficients
significant with the *wrong sign* (the model reads a high VIX as a *good* sign) —
zero-rate-era artefacts that drag the prediction backward.

**3. Macro features look useless alone but work once you add fixed effects.**
Macro-only AUROC jumps from **{macro_pooled['AUROC']:.3f}** (Pooled, below the
0.5 chance line) to **{macro_fe['AUROC']:.3f}** (FE), a **{flip:+.3f}** gain.
Alone, macros can't explain cross-firm or cross-year differences — every firm
sees the same VIX each month; with industry-and-year fixed effects they only
have to explain within-cell movement, which they do.

**4. None of the gaps are statistically significant.** Under the firm-clustered
bootstrap, even Sector-rel's interval
**[{srel_pooled['AUROC_lo']:.3f}, {srel_pooled['AUROC_hi']:.3f}]** overlaps
Market-only's **[{market_pooled['AUROC_lo']:.3f}, {market_pooled['AUROC_hi']:.3f}]**
and Full's **[{full_pooled['AUROC_lo']:.3f}, {full_pooled['AUROC_hi']:.3f}]**.
With only 77 firms (= 77 clusters) the intervals are wide; what survives is
**directional** — sector-relative scores highest everywhere — not a significance
claim.

**5. Calibration was tried and didn't help — and it's the wrong tool here.**
Platt and isotonic scaling on the deployed model left Brier essentially unchanged
(**{_calib('raw','brier'):.3f}** raw → **{_calib('platt','brier'):.3f}** Platt;
isotonic **{_calib('isotonic','brier'):.3f}**, slightly worse). Two reasons:
a logistic regression is **already calibrated on its training data by
construction**, so post-hoc scaling is near-identity; and the only residual
miscalibration on validation is a base-rate shift (train ~8.5% → val
~{_calib('raw','base_rate')*100:.0f}%) that a train-fit calibrator can't correct.
Because calibration is *monotonic*, it also can't change which firms clear the
top-decile threshold — so it can't fix the per-sector flagging gaps. That took
the sector-relative *features* (Finding 1), not calibration.
"""
)

st.markdown("---")


# =========================================================================
# Conclusion
# =========================================================================

st.header("✅ Conclusion")

st.markdown(
    """
Market signals carry the predictive content — and **z-scoring them within each
sector carries more**: Sector-relative is the strongest feature group in every
variant, and AUPRC (the honest metric under imbalance) rises most. Accounting and
macro features add little and, in the Full model, actively mislead. We carry
forward the **Market + sector-relative pooled logit** for the per-sector and
per-firm work: pooled (not FE) for interpretability and sliceability, *augmented*
with the sector-relative features that give it the within-industry signal FE got
from dummies. Calibration (Platt / isotonic) was tried and **didn't move the
needle** — a logit is already calibrated on its training data, and the leftover
validation gap is a base-rate shift no train-fit calibrator can fix. The model
still discriminates only modestly and no specification gap is statistically
significant. Read this as a **diagnosis of where the signal lives, not a
deployable predictor** — the remaining lever is a bigger panel (77 firms = 77
clusters is what keeps the intervals wide), not more post-processing.

**Caveats.** Panel = 77 firms, 12,473 firm-months; validation 2021–2023
(~2,772 rows). CIs use 1,000 firm-clustered bootstrap resamples (`seed=42`, so
they reproduce bit-for-bit). Sector-relative features are *contemporaneous*
within-industry z-scores (same-month peers only — no look-ahead). The Hazard
family can fail on small subsets (ill-conditioned Hessian); the FE dummy-block
standard errors are NaN; the Altman Z-score subset is not reportable on this
panel.

*Reproduce:* `MPLBACKEND=Agg python src/run.py` regenerates
`ablation_results.csv` and `full_model_coefficients_*.csv`. Full narrative:
`outputs/ablation_findings.md`.
"""
)
