"""
Evaluation metrics + diagnostic suites.

Core metrics: AUROC, AUPRC, Brier score, top-K capture, top-K lift.
Qualitative diagnostic: lead-time (how far in advance the model flagged
distress before the realized event).

Diagnostic suites (ablation across feature groups, rolling-window robustness)
are included here because they are regression-based and share `evaluate_model`
internals. Called from run.py after the main eval, wrapped in try/except per
the 3-tier fallback policy.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .config import FEATURE_COLS, FIRMS, LABEL_COL, LEAD_TIME_THRESHOLD, PATHS, TOP_K_FRACTION


# =============================================================================
# Primary metric: evaluate_model
# =============================================================================

def evaluate_model(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    model_name: str,
    split_name: str,
) -> dict[str, float]:
    """All required metrics: AUROC, AUPRC, Brier, Top-K capture & lift."""
    results: dict[str, float] = {}
    y_true = y_true.reset_index(drop=True)

    results["auroc"] = roc_auc_score(y_true, y_pred_proba)
    results["auprc"] = average_precision_score(y_true, y_pred_proba)
    results["brier"] = brier_score_loss(y_true, y_pred_proba)

    k = max(1, int(len(y_true) * TOP_K_FRACTION))
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    events_in_top_k = y_true.iloc[top_k_idx].sum()
    total_events = y_true.sum()
    results["top10_capture"] = events_in_top_k / total_events if total_events > 0 else 0
    results["top10_lift"] = results["top10_capture"] / TOP_K_FRACTION

    print(f"\n  {model_name} | {split_name}:")
    print(f"    AUROC:           {results['auroc']:.4f}")
    print(f"    AUPRC:           {results['auprc']:.4f}")
    print(f"    Brier Score:     {results['brier']:.4f}")
    print(f"    Top-10% Capture: {results['top10_capture']:.1%} of events")
    print(f"    Top-10% Lift:    {results['top10_lift']:.2f}x")
    return results


# =============================================================================
# Bootstrap CI helper (used by ablation_analysis)
# =============================================================================

def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC.

    Resamples rows with replacement `n_boot` times, computes AUROC per resample,
    returns the (alpha/2, 1-alpha/2) percentiles. Resamples that draw only one
    class are skipped (AUROC undefined). Returns (nan, nan) if no resample is
    usable.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    aurocs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    if not aurocs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aurocs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# =============================================================================
# Lead time
# =============================================================================

def compute_lead_time(
    df: pd.DataFrame,
    pred_proba: pd.Series,
    threshold: float = LEAD_TIME_THRESHOLD,
) -> list[dict]:
    """For each firm with an event in `df`, compute months between first
    flag (pred > threshold) and first event."""
    df = df.copy().reset_index(drop=True)
    df["pred"] = pred_proba.values if hasattr(pred_proba, "values") else pred_proba
    df["flagged"] = df["pred"] > threshold

    lead_times: list[dict] = []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date")
        events = group[group[LABEL_COL] == 1]
        if len(events) == 0:
            continue
        first_event = events["date"].iloc[0]
        flags_before = group[(group["flagged"]) & (group["date"] < first_event)]
        if len(flags_before) > 0:
            first_flag = flags_before["date"].iloc[0]
            months = (first_event.year - first_flag.year) * 12 + (first_event.month - first_flag.month)
            lead_times.append({
                "ticker": ticker,
                "firm": FIRMS.get(ticker, {}).get("name", ticker),
                "lead_months": months,
            })

    if lead_times:
        lt_df = pd.DataFrame(lead_times)
        print(f"\n  Lead Time Analysis:")
        for _, row in lt_df.iterrows():
            print(f"    {row['ticker']:5s} ({row['firm']}): {row['lead_months']} months early")
        print(f"    Average: {lt_df['lead_months'].mean():.1f} months | "
              f"Median: {lt_df['lead_months'].median():.0f} months")
    else:
        print("\n  Lead Time: no firms flagged before their event")
    return lead_times


# =============================================================================
# Diagnostic: ablation across feature groups
# =============================================================================

def ablation_analysis(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Compare feature subsets: which feature groups are carrying the signal?"""
    print("\n" + "=" * 70)
    print("ABLATION: FEATURE GROUP COMPARISON")
    print("=" * 70)

    subsets = {
        "Accounting only":   ["leverage", "liquidity_buffer", "wc_ratio", "profitability"],
        "Market only":       ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Macro only":        ["vix", "term_spread", "credit_spread"],
        "Filing only":       ["late_filing"],
        "Acct + Market":     ["leverage", "liquidity_buffer", "wc_ratio", "profitability",
                              "ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Full model":        FEATURE_COLS,
        "Altman Z-score":    ["z_score"],
    }

    results = []
    for name, cols in subsets.items():
        try:
            m = sm.Logit(train[LABEL_COL], sm.add_constant(train[cols])).fit(disp=0)
            p = m.predict(sm.add_constant(test[cols]))
            y_true = test[LABEL_COL].values
            y_pred = p.values if hasattr(p, "values") else np.asarray(p)
            auroc = roc_auc_score(y_true, y_pred)
            lo, hi = _bootstrap_auroc_ci(y_true, y_pred)
            results.append({
                "Feature set": name,
                "N": len(cols),
                "AUROC": auroc,
                "AUROC_lo": lo,
                "AUROC_hi": hi,
                "AUPRC": average_precision_score(y_true, y_pred),
                "Brier": brier_score_loss(y_true, y_pred),
            })
        except Exception as e:
            print(f"  {name}: failed ({e})")
    rdf = pd.DataFrame(results)
    print("\n" + rdf.round(4).to_string(index=False))

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    out_path = os.path.join(PATHS.OUTPUTS, "ablation_results.csv")
    rdf.to_csv(out_path, index=False)
    print(f"\n  Saved ablation results to: {out_path}")

    return rdf


# =============================================================================
# Diagnostic: rolling-window robustness
# =============================================================================

def robustness_rolling_window(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window validation: train on everything before year Y,
    test on year Y. Reports AUROC / AUPRC per year."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: ROLLING/EXPANDING WINDOW")
    print("=" * 70)

    results = []
    for test_year in range(2015, df["year"].max() + 1):
        tr = df[df["year"] < test_year]
        te = df[df["year"] == test_year]
        if len(te) == 0 or te[LABEL_COL].nunique() < 2:
            continue
        try:
            m = sm.Logit(tr[LABEL_COL], sm.add_constant(tr[FEATURE_COLS])).fit(disp=0)
            p = m.predict(sm.add_constant(te[FEATURE_COLS]))
            results.append({
                "test_year": test_year,
                "auroc": roc_auc_score(te[LABEL_COL], p),
                "auprc": average_precision_score(te[LABEL_COL], p),
            })
        except Exception as e:
            print(f"  Year {test_year}: skipped ({e})")

    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))
    if len(rdf) > 0:
        print(f"\nMean AUROC: {rdf['auroc'].mean():.4f} +/- {rdf['auroc'].std():.4f}")
    return rdf
