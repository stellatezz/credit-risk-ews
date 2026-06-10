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

from .config import (
    FEATURE_COLS,
    FIRMS,
    LABEL_COL,
    LEAD_TIME_THRESHOLD,
    MARKET_FEATURE_COLS,
    MARKET_REL_FEATURE_COLS,
    PATHS,
    TOP_K_FRACTION,
)


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
    m = sm.Logit(train[LABEL_COL].reset_index(drop=True), sm.add_constant(train_fe)).fit(
        disp=0, method="bfgs", maxiter=200
    )
    return m.predict(sm.add_constant(test_fe))


def _fit_hazard(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Discrete-time hazard logit: pooled logit + log-duration baseline.

    Duration = months since each firm's first observation in the combined
    train + test panel — continuous across the split boundary, so test
    predictions are not reset to duration = 1. Approximates Shumway (2001)
    without the full hazard panel restructure.
    """
    from .models import _global_duration_map  # local to avoid circular risk

    dur = _global_duration_map(train, test)

    def with_log_duration(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["months_obs"] = df.set_index(["ticker", "date"]).index.map(dur).values
        if df["months_obs"].isna().any():
            raise ValueError(
                f"_fit_hazard: {int(df['months_obs'].isna().sum())} rows "
                "have no entry in the global duration lookup"
            )
        df["log_duration"] = np.log(df["months_obs"])
        return df

    train_h = with_log_duration(train)
    test_h = with_log_duration(test)
    X_train = sm.add_constant(train_h[cols + ["log_duration"]])
    X_test = sm.add_constant(test_h[cols + ["log_duration"]])
    m = sm.Logit(train_h[LABEL_COL], X_train).fit(disp=0)
    # Predictions inherit test_h's index (we never sorted), so they align
    # with the bootstrap's firm_ids array taken from test["ticker"].values.
    return m.predict(X_test)


MODEL_FAMILIES: dict[str, callable] = {
    "pooled": _fit_pooled,
    "fe":     _fit_fe,
    "hazard": _fit_hazard,
}


# =============================================================================
# Coefficient persistence (Full model per family)
# =============================================================================

def persist_full_model_coefficients(
    train: pd.DataFrame,
    family_name: str,
) -> str:
    """Fit the Full model with the given family and write its coefficient table
    to `outputs/full_model_coefficients_<family>.csv`.

    Persists: feature, coef, std_err, p_value.
    Returns the absolute output path written.
    """
    # We need the fitted statsmodels Results object, not just predictions,
    # to read coefficients. Inline the fit here so we can keep the wrappers
    # in MODEL_FAMILIES focused on predictions.
    cols = FEATURE_COLS
    if family_name == "pooled":
        m = sm.Logit(train[LABEL_COL], sm.add_constant(train[cols])).fit(disp=0)
    elif family_name == "fe":
        train_fe = pd.concat([
            train[cols].reset_index(drop=True),
            pd.get_dummies(train["industry"], prefix="ind", drop_first=False).astype(float).reset_index(drop=True),
            pd.get_dummies(train["year"], prefix="yr", drop_first=False).astype(float).reset_index(drop=True),
        ], axis=1)
        m = sm.Logit(train[LABEL_COL].reset_index(drop=True), sm.add_constant(train_fe)).fit(
            disp=0, method="bfgs", maxiter=200
        )
    elif family_name == "hazard":
        tr = train.copy().sort_values(["ticker", "date"])
        tr["months_obs"] = tr.groupby("ticker").cumcount() + 1
        tr["log_duration"] = np.log(tr["months_obs"])
        X = sm.add_constant(tr[cols + ["log_duration"]])
        m = sm.Logit(tr[LABEL_COL], X).fit(disp=0)
    else:
        raise ValueError(f"unknown family: {family_name}")

    coef_df = pd.DataFrame({
        "feature": m.params.index,
        "coef": m.params.values,
        "std_err": m.bse.values,
        "p_value": m.pvalues.values,
    })

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    out_path = os.path.join(PATHS.OUTPUTS, f"full_model_coefficients_{family_name}.csv")
    coef_df.to_csv(out_path, index=False)
    print(f"  Saved {family_name} full-model coefficients to: {out_path}")
    return out_path


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
        "Sector-rel only":   MARKET_REL_FEATURE_COLS,
        "Market + sector-rel": MARKET_FEATURE_COLS + MARKET_REL_FEATURE_COLS,
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
# Per-slice evaluation (Phase 3 items #3 + #6)
# =============================================================================

def evaluate_by_slice(
    train: pd.DataFrame,
    val: pd.DataFrame,
    slice_col: str,
    feature_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    """Fit a pooled logit on `feature_cols` and evaluate per unique value of
    `slice_col` on the val set. AUROC uses firm-clustered bootstrap CIs.

    Returns DataFrame with columns:
      slice, n_rows, n_firms, n_events, event_rate, AUROC, AUROC_lo, AUROC_hi

    Slices with fewer than 2 events or fewer than 30 rows are returned with
    NaN AUROC/CI (uninformative) but kept in the result so the reader sees
    the sample-size landscape.
    """
    print(f"\nPer-slice evaluation on '{slice_col}':")

    # Fit once on train using ALL features in feature_cols.
    m = sm.Logit(train[label_col], sm.add_constant(train[feature_cols])).fit(disp=0)
    val_pred = m.predict(sm.add_constant(val[feature_cols]))

    rows: list[dict] = []
    for slice_val, sub in val.groupby(slice_col, dropna=False):
        sub_idx = sub.index
        y_true = sub[label_col].values
        y_pred = val_pred.loc[sub_idx].values
        firm_ids = sub["ticker"].values
        n_rows = len(sub)
        n_firms = sub["ticker"].nunique()
        n_events = int(y_true.sum())
        event_rate = float(y_true.mean()) if n_rows else float("nan")
        # AUROC undefined if only one class or sample too small
        if n_events < 2 or n_events == n_rows or n_rows < 30:
            auroc = float("nan")
            lo = hi = float("nan")
        else:
            auroc = roc_auc_score(y_true, y_pred)
            lo, hi = _bootstrap_auroc_ci(y_true, y_pred, firm_ids=firm_ids)
        rows.append({
            "slice": slice_val,
            "n_rows": n_rows,
            "n_firms": n_firms,
            "n_events": n_events,
            "event_rate": event_rate,
            "AUROC": auroc,
            "AUROC_lo": lo,
            "AUROC_hi": hi,
        })

    rdf = pd.DataFrame(rows).sort_values("AUROC", ascending=False, na_position="last")
    print(rdf.round(3).to_string(index=False))
    return rdf


def error_analysis_by_slice(
    train: pd.DataFrame,
    val: pd.DataFrame,
    slice_col: str,
    feature_cols: list[str],
    label_col: str,
    top_k_fraction: float = TOP_K_FRACTION,
) -> pd.DataFrame:
    """Per-slice false-positive / false-negative breakdown at top-decile
    flagging threshold.

    Threshold is chosen as the val-set quantile (1 - top_k_fraction) of
    predicted probabilities — the same population the analyst sees. Slices
    inherit that global threshold so per-slice precision/recall is comparable.

    Returns DataFrame:
      slice, n_rows, n_events, n_flags, TP, FP, FN, TN, precision, recall
    """
    print(f"\nPer-slice error analysis on '{slice_col}' (threshold = top {top_k_fraction:.0%}):")

    m = sm.Logit(train[label_col], sm.add_constant(train[feature_cols])).fit(disp=0)
    val_pred = m.predict(sm.add_constant(val[feature_cols]))
    threshold = float(val_pred.quantile(1 - top_k_fraction))
    print(f"  Global flagging threshold (val top-{int(top_k_fraction*100)}%): {threshold:.4f}")

    val_flagged = (val_pred >= threshold).astype(int)
    y = val[label_col].astype(int)

    rows: list[dict] = []
    for slice_val, sub in val.groupby(slice_col, dropna=False):
        sub_idx = sub.index
        y_sub = y.loc[sub_idx].values
        f_sub = val_flagged.loc[sub_idx].values
        TP = int(((f_sub == 1) & (y_sub == 1)).sum())
        FP = int(((f_sub == 1) & (y_sub == 0)).sum())
        FN = int(((f_sub == 0) & (y_sub == 1)).sum())
        TN = int(((f_sub == 0) & (y_sub == 0)).sum())
        n_rows = len(sub)
        n_events = int(y_sub.sum())
        n_flags = int(f_sub.sum())
        precision = TP / n_flags if n_flags > 0 else float("nan")
        recall = TP / n_events if n_events > 0 else float("nan")
        rows.append({
            "slice": slice_val,
            "n_rows": n_rows,
            "n_events": n_events,
            "n_flags": n_flags,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": precision,
            "recall": recall,
        })

    rdf = pd.DataFrame(rows).sort_values("recall", ascending=False, na_position="last")
    print(rdf.round(3).to_string(index=False))
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


# =============================================================================
# Calibration (Phase 3 #1): Platt + isotonic on the deployed pooled logit
# =============================================================================

def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: mean |confidence − accuracy| over equal-width
    probability bins, weighted by bin population."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_analysis(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit the deployed pooled logit, then Platt and isotonic calibration on top.

    The base model is fit on train; the two calibrators are fit on the *train*
    predictions and applied to the unseen *val* set. Reports Brier + ECE for
    raw / Platt / isotonic (AUROC is unchanged — both calibrators are monotone,
    so ranking is preserved; only the probability scale moves).

    Caveat: fitting the calibrator on in-sample train scores is mildly
    optimistic; a held-out calibration fold would be more rigorous. Val remains
    unseen by both the base model and the calibrators, so the reported Brier/ECE
    improvements are out-of-sample.

    Returns (results_df, {"y_true", "raw", "platt", "isotonic"}) — the second
    element feeds the reliability-curve plot.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    print("\n" + "=" * 70)
    print("CALIBRATION: Platt + isotonic on the deployed pooled logit")
    print("=" * 70)

    m = sm.Logit(train[label_col], sm.add_constant(train[feature_cols])).fit(disp=0)
    tr_p = np.clip(m.predict(sm.add_constant(train[feature_cols])).values, 1e-6, 1 - 1e-6)
    va_p = np.clip(m.predict(sm.add_constant(val[feature_cols])).values, 1e-6, 1 - 1e-6)
    y_tr = train[label_col].values.astype(int)
    y_va = val[label_col].values.astype(int)

    # Platt: logistic regression on the logit of the base score.
    tr_logit = np.log(tr_p / (1 - tr_p)).reshape(-1, 1)
    va_logit = np.log(va_p / (1 - va_p)).reshape(-1, 1)
    platt = LogisticRegression().fit(tr_logit, y_tr)
    va_platt = platt.predict_proba(va_logit)[:, 1]

    # Isotonic: non-parametric monotone fit on the raw base probability.
    iso = IsotonicRegression(out_of_bounds="clip").fit(tr_p, y_tr)
    va_iso = iso.predict(va_p)

    methods = {"raw": va_p, "platt": va_platt, "isotonic": va_iso}
    base_rate = float(y_va.mean())
    rows = []
    for name, p in methods.items():
        rows.append({
            "method": name,
            "brier": brier_score_loss(y_va, p),
            "ece": _expected_calibration_error(y_va, p),
            "auroc": roc_auc_score(y_va, p),
            "mean_pred": float(np.mean(p)),
            "base_rate": base_rate,
        })
    rdf = pd.DataFrame(rows)
    print(rdf.round(4).to_string(index=False))

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    out_path = os.path.join(PATHS.OUTPUTS, "calibration_results.csv")
    rdf.to_csv(out_path, index=False)
    print(f"\n  Saved calibration results to: {out_path}")

    return rdf, {"y_true": y_va, **methods}
