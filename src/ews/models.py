"""
Three interpretable regressions: pooled logit, fixed-effects logit, hazard logit.

All three share the same FEATURE_COLS from config and return a
(model, preds_dict) tuple where preds_dict maps split-name -> pd.Series of
predicted probabilities for that split.

Model failure policy: each model may legitimately fail to fit on small-data
Phase 1 panels (e.g., perfect separation, insufficient industry variation).
Caller (run.py) wraps these calls in try/except and falls back to pooled
predictions with a loud warning — see the 3-tier policy block at the top of
run.py.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import FEATURE_COLS, LABEL_COL


def _global_duration_map(*splits: pd.DataFrame) -> pd.Series:
    """Build a (ticker, date) -> duration lookup spanning multiple splits.

    Duration is the number of monthly observations since each firm's first
    appearance in the combined panel, starting at 1. The hazard model uses
    this to ensure `log_duration` is continuous across train/val/test
    boundaries — without it, a firm observed for 132 months in training
    would reset to duration = 1 on its first val row, and the model's
    log_duration coefficient would be applied at a value far outside the
    range it was trained on.

    Returns a Series indexed by (ticker, date) with integer duration values.
    """
    combined = pd.concat(splits, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"])
    combined["_dur"] = combined.groupby("ticker").cumcount() + 1
    return combined.set_index(["ticker", "date"])["_dur"]


def _coef_table(model) -> pd.DataFrame:
    """Formatted coefficient summary with odds ratios + significance stars."""
    return pd.DataFrame({
        "coef": model.params,
        "odds_ratio": np.exp(model.params),
        "std_err": model.bse,
        "p_value": model.pvalues,
        "sig": model.pvalues.map(
            lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        ),
    })


# =============================================================================
# Model 1: Pooled logit (Ohlson-style baseline)
# =============================================================================

def model_pooled_logit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[object, dict[str, pd.Series]]:
    """Pooled logistic regression: every firm-month is an independent obs."""
    print("\n" + "=" * 70)
    print("MODEL 1: POOLED LOGISTIC REGRESSION (Baseline)")
    print("=" * 70)

    X_train = sm.add_constant(train[FEATURE_COLS])
    y_train = train[LABEL_COL]
    model = sm.Logit(y_train, X_train).fit(disp=0)

    print("\nCoefficient Summary (Odds Ratios):")
    print("-" * 60)
    print(_coef_table(model).round(4).to_string())

    preds = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        X = sm.add_constant(split[FEATURE_COLS])
        preds[name] = model.predict(X)
    return model, preds


# =============================================================================
# Model 2: Fixed-effects panel logit (industry + year dummies, clustered SEs)
# =============================================================================

def model_fe_logit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[object, dict[str, pd.Series]]:
    """Pooled logit + industry & year dummies + SEs clustered on year-month."""
    print("\n" + "=" * 70)
    print("MODEL 2: FIXED-EFFECTS PANEL LOGIT")
    print("=" * 70)

    train_fe = train.copy().reset_index(drop=True)
    val_fe = val.copy().reset_index(drop=True)
    test_fe = test.copy().reset_index(drop=True)

    ind_dum_tr = pd.get_dummies(train_fe["industry"], prefix="ind", drop_first=True, dtype=float)
    ind_dum_va = pd.get_dummies(val_fe["industry"], prefix="ind", drop_first=True, dtype=float)
    ind_dum_te = pd.get_dummies(test_fe["industry"], prefix="ind", drop_first=True, dtype=float)

    yr_dum_tr = pd.get_dummies(train_fe["year"], prefix="yr", drop_first=True, dtype=float)
    yr_dum_va = pd.get_dummies(val_fe["year"], prefix="yr", drop_first=True, dtype=float)
    yr_dum_te = pd.get_dummies(test_fe["year"], prefix="yr", drop_first=True, dtype=float)

    X_train = sm.add_constant(
        pd.concat([train_fe[FEATURE_COLS], ind_dum_tr, yr_dum_tr], axis=1)
    )
    y_train = train_fe[LABEL_COL]
    cluster_var = train_fe["year"].astype(str) + "_" + train_fe["month"].astype(str)

    model = sm.Logit(y_train, X_train).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_var},
        disp=0,
    )

    print("\nCoefficient Summary (clustered SEs, main features only):")
    print("-" * 60)
    coef_df = _coef_table(model)
    main_rows = [c for c in coef_df.index if not c.startswith("yr_")]
    print(coef_df.loc[main_rows].round(4).to_string())
    n_yr = sum(1 for c in coef_df.index if c.startswith("yr_"))
    print(f"\n(+ {n_yr} year fixed-effect dummies, not shown)")

    preds = {}
    for name, split, id_, yd_ in [
        ("train", train_fe, ind_dum_tr, yr_dum_tr),
        ("val", val_fe, ind_dum_va, yr_dum_va),
        ("test", test_fe, ind_dum_te, yr_dum_te),
    ]:
        X = pd.concat([
            split[FEATURE_COLS].reset_index(drop=True),
            id_.reset_index(drop=True),
            yd_.reset_index(drop=True),
        ], axis=1)
        X = X.reindex(columns=X_train.columns[1:], fill_value=0)
        X = sm.add_constant(X)
        preds[name] = model.predict(X)
    return model, preds


# =============================================================================
# Model 3: Discrete-time hazard logit (Shumway 2001)
# =============================================================================

def _build_hazard_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each firm only until their first event (Shumway 2001 spec)."""
    rows = []
    for _, group in df.groupby("ticker"):
        group = group.sort_values("date")
        for _, row in group.iterrows():
            rows.append(row)
            if row[LABEL_COL] == 1:
                break
    return pd.DataFrame(rows)


def model_hazard_logit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[object, dict[str, pd.Series]]:
    """Discrete-time hazard logit with log-duration baseline (Shumway-style).

    Duration is computed across the combined train + val + test panel via
    `_global_duration_map`, so that val and test predictions inherit each
    firm's cumulative observation count rather than resetting to 1 at the
    split boundary.
    """
    print("\n" + "=" * 70)
    print("MODEL 3: DISCRETE-TIME HAZARD LOGIT (Shumway-style)")
    print("=" * 70)

    # Build a (ticker, date) -> duration lookup that spans all three splits
    # so log_duration is continuous across boundaries.
    duration_lookup = _global_duration_map(train, val, test)

    def _with_duration(split: pd.DataFrame) -> pd.DataFrame:
        """Attach `duration` and `log_duration` columns to a split without
        reordering its rows."""
        s = split.copy()
        s["duration"] = s.set_index(["ticker", "date"]).index.map(duration_lookup).values
        missing = int(s["duration"].isna().sum())
        if missing:
            raise ValueError(
                f"hazard model: {missing} rows in split have no entry in the "
                "global duration lookup (rows must come from the same panel "
                "passed to _global_duration_map)"
            )
        s["log_duration"] = np.log(s["duration"])
        return s

    # Build the Shumway training panel from train rows that already carry
    # duration. This means train_haz duration matches whatever we use at
    # predict time — no split-relative resetting.
    train_with_dur = _with_duration(train)
    train_haz = (
        _build_hazard_panel(train_with_dur)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    print(f"Hazard panel: {len(train_haz)} rows (was {len(train)})")
    print(f"  Firms with event in training: "
          f"{train_haz.groupby('ticker')[LABEL_COL].max().sum():.0f}")

    hazard_features = FEATURE_COLS + ["log_duration"]
    X_train = sm.add_constant(train_haz[hazard_features])
    y_train = train_haz[LABEL_COL]
    model = sm.Logit(y_train, X_train).fit(disp=0)

    print("\nCoefficient Summary:")
    print("-" * 60)
    print(_coef_table(model).round(4).to_string())

    # Predict on each raw split using the same global duration lookup.
    # We deliberately do NOT sort or reset_index — predictions inherit each
    # split's caller-supplied row order, matching the other two models.
    preds = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        s = _with_duration(split)
        X = sm.add_constant(s[hazard_features])
        preds[name] = model.predict(X)
    return model, preds
