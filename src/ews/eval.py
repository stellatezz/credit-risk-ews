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
    firm_ids: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC.

    If `firm_ids` is provided, performs *firm-clustered* bootstrap: each
    resample draws firm IDs with replacement and includes all rows for each
    drawn firm. This correctly accounts for within-firm autocorrelation that
    row-level bootstrap ignores (a major issue in credit-risk panels where
    consecutive firm-months are highly correlated).

    If `firm_ids` is None, falls back to row-level bootstrap (resample row
    indices with replacement).

    Resamples that yield single-class y_true are skipped (AUROC undefined).
    Returns (nan, nan) if no resample is usable.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if firm_ids is not None:
        firm_ids = np.asarray(firm_ids)
        unique_firms = np.unique(firm_ids)
        # Pre-index rows-per-firm once for speed
        firm_to_rows = {f: np.where(firm_ids == f)[0] for f in unique_firms}
        aurocs: list[float] = []
        for _ in range(n_boot):
            sampled_firms = rng.choice(unique_firms, size=len(unique_firms), replace=True)
            idx = np.concatenate([firm_to_rows[f] for f in sampled_firms])
            if len(np.unique(y_true[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    else:
        aurocs: list[float] = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))

    if not aurocs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aurocs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# =============================================================================
# Per-family fit wrappers (used by ablation_analysis to loop over model types)
# =============================================================================

def _fit_pooled(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Plain pooled logit: y ~ const + cols. Returns predicted probabilities on test."""
    m = sm.Logit(train[LABEL_COL], sm.add_constant(train[cols])).fit(disp=0)
    return m.predict(sm.add_constant(test[cols]))


def _fit_fe(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Pooled logit with industry + year dummies (lightweight FE for ablation).

    Uses drop_first=False and lets statsmodels absorb perfect collinearity rather
    than risk a singular matrix when one category dominates a small feature subset.
    """
    train_fe = pd.concat([
        train[cols].reset_index(drop=True),
        pd.get_dummies(train["industry"], prefix="ind", drop_first=False).astype(float).reset_index(drop=True),
        pd.get_dummies(train["year"], prefix="yr", drop_first=False).astype(float).reset_index(drop=True),
    ], axis=1)
    test_fe = pd.concat([
        test[cols].reset_index(drop=True),
        pd.get_dummies(test["industry"], prefix="ind", drop_first=False).astype(float).reset_index(drop=True),
        pd.get_dummies(test["year"], prefix="yr", drop_first=False).astype(float).reset_index(drop=True),
    ], axis=1)
    # Align test columns to train (dummies may differ if a year/industry appears
    # in only one split)
    test_fe = test_fe.reindex(columns=train_fe.columns, fill_value=0.0)
    m = sm.Logit(train[LABEL_COL].reset_index(drop=True), sm.add_constant(train_fe)).fit(disp=0)
    return m.predict(sm.add_constant(test_fe))


def _fit_hazard(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Discrete-time hazard logit: pooled logit + log-duration baseline.

    Duration = months since first observation per firm. Approximates the
    Shumway (2001) form without the full hazard panel restructure.
    """
    def with_log_duration(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["ticker", "date"])
        df["months_obs"] = df.groupby("ticker").cumcount() + 1
        df["log_duration"] = np.log(df["months_obs"])
        return df

    train_h = with_log_duration(train)
    test_h = with_log_duration(test)
    X_train = sm.add_constant(train_h[cols + ["log_duration"]])
    X_test = sm.add_constant(test_h[cols + ["log_duration"]])
    m = sm.Logit(train_h[LABEL_COL], X_train).fit(disp=0)
    # Predictions must align back to the original test row order so the bootstrap
    # firm_ids index matches.
    preds = m.predict(X_test)
    preds.index = test_h.index
    return preds.reindex(test.index)


MODEL_FAMILIES: dict[str, callable] = {
    "pooled": _fit_pooled,
    "fe":     _fit_fe,
    "hazard": _fit_hazard,
}


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
        "Accounting only":   ["leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability"],
        "Market only":       ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Macro only":        ["vix", "term_spread", "credit_spread"],
        "Filing only":       ["late_filing"],
        "Acct + Market":     ["leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability",
                              "ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Full model":        FEATURE_COLS,
        "Altman Z-score":    ["z_score"],
    }

    results = []
    firm_ids = test["ticker"].values
    y_true = test[LABEL_COL].values

    for family_name, fit_fn in MODEL_FAMILIES.items():
        print(f"\n-- {family_name} --")
        for subset_name, cols in subsets.items():
            try:
                p = fit_fn(train, test, cols)
                y_pred = p.values if hasattr(p, "values") else np.asarray(p)
                auroc = roc_auc_score(y_true, y_pred)
                lo, hi = _bootstrap_auroc_ci(y_true, y_pred, firm_ids=firm_ids)
                results.append({
                    "model_family": family_name,
                    "Feature set": subset_name,
                    "N": len(cols),
                    "AUROC": auroc,
                    "AUROC_lo": lo,
                    "AUROC_hi": hi,
                    "AUPRC": average_precision_score(y_true, y_pred),
                    "Brier": brier_score_loss(y_true, y_pred),
                })
            except Exception as e:
                print(f"  {subset_name}: failed ({type(e).__name__}: {str(e)[:80]})")

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
