"""
Panel assembly + time split.

`assemble_panel` merges the four loader outputs into a single firm-month
panel, joins firm metadata, filters to the modeling window, and drops rows
with missing features. `time_split` partitions the panel by year.

The panel is the single modeling input — every downstream model, eval,
and chart reads from here.

ASCII merge pipeline:

    market_features ─┐
                     ├─inner join on (ticker, date)─┐
    fundamentals    ─┘                              │
                                                    ├─ left  join on date
                                          macros ──┘
                                                    │
                                                    ├─ inner join on (ticker, date)
                                          labels ──┘
                                                    │
                                                    ▼
                                          add industry / year / month / firm_name
                                                    │
                                                    ▼
                                          filter year >= PANEL_START_YEAR
                                                    │
                                                    ▼
                                          sort by (ticker, date)
                                                    │
                                                    ▼
                                          impute wc_ratio NaN → 0
                                          + emit wc_ratio_missing
                                                    │
                                                    ▼
                                          dropna on FEATURE_COLS + label_a
                                                    │
                                                    ▼
                                                 PANEL
"""

import os

import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLS,
    FIRMS,
    LABEL_COL,
    PANEL_START_YEAR,
    PATHS,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
)

# Raw market features that get a within-(industry, month) z-scored counterpart.
_REL_BASE_COLS = ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"]


def _add_industry_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add within-(industry, month) z-scores of the raw market features.

    For each market feature, ``rel = (x - mean) / std`` is computed across all
    firms in the same industry in the same month. This is a *contemporaneous*
    cross-sectional normalisation — same-month peers only, no future or label
    information — that lets a pooled model ask "is this firm unusual *for its
    sector* right now?" (e.g. a distressed airline vs. ordinary airline
    volatility). Single-firm industry-months have no peer spread, so their
    relative value is set to 0 (no relative signal).
    """
    for col in _REL_BASE_COLS:
        grp = panel.groupby(["industry", "date"])[col]
        mean = grp.transform("mean")
        std = grp.transform("std")  # ddof=1 → NaN when the group has < 2 firms
        rel = (panel[col] - mean) / std
        panel[f"{col}_rel"] = rel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel


def _load_firm_categories() -> pd.DataFrame:
    """Read data/firm_categories.csv → DataFrame[ticker, sector_raw, archetype, purpose].

    Repo-root-anchored path so callers don't need a working directory.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "data", "firm_categories.csv")
    return pd.read_csv(path)


def assemble_panel(
    market_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the four loader outputs into a firm-month panel.

    Preserves the exact merge order from the Phase 1 monolith so the output
    SHA256 matches the baseline. If you change join order, row filter, or
    dropna subset, the panel hash changes — re-baseline before committing.
    """
    print("\nBuilding firm-month panel...")

    # 1. Market + fundamentals: inner join on (ticker, date)
    panel = market_df.merge(fund_df, on=["ticker", "date"], how="inner")

    # 2. Macro: left join on date (same for all firms in a month)
    panel = panel.merge(macro_df, on="date", how="left")

    # 3. Labels: inner join on (ticker, date). Take only the subset of columns
    #    the Phase 1 panel carries — label_b is reserved for Phase 2.
    panel = panel.merge(
        label_df[["ticker", "date", "label_a", "forward_max_drawdown"]],
        on=["ticker", "date"],
        how="inner",
    )

    # 4. Firm metadata
    panel["industry"] = panel["ticker"].map(lambda t: FIRMS[t]["industry"])
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["firm_name"] = panel["ticker"].map(lambda t: FIRMS[t]["name"])

    # 4b. Attach archetype + raw sector (parsed from Phase 2 sample-company doc).
    #     Left-join so a firm missing from firm_categories.csv keeps a NaN
    #     archetype rather than dropping the row silently.
    cats = _load_firm_categories()[["ticker", "sector_raw", "archetype"]]
    panel = panel.merge(cats, on="ticker", how="left")

    # 5. Filter to modeling window + stable sort
    panel = panel[panel["year"] >= PANEL_START_YEAR].copy()
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 6. Impute structurally-undefined wc_ratio (REITs file unclassified balance
    #    sheets and don't report current assets/liabilities). We fill with 0.0
    #    as a neutral value and emit a binary missingness indicator so the
    #    model can learn "no working-capital signal" as its own feature.
    panel["wc_ratio_missing"] = panel["wc_ratio"].isna().astype(int)
    panel["wc_ratio"] = panel["wc_ratio"].fillna(0.0)

    # 7. Drop rows with any remaining missing feature / label.
    before = len(panel)
    panel = panel.dropna(subset=FEATURE_COLS + [LABEL_COL])
    print(f"  Panel: {len(panel)} firm-months ({before - len(panel)} dropped for NaN)")
    print(f"  Firms: {panel['ticker'].nunique()}, "
          f"Date range: {panel['date'].min().strftime('%Y-%m')} → "
          f"{panel['date'].max().strftime('%Y-%m')}")
    print(f"  Overall event rate: {panel[LABEL_COL].mean():.1%}")

    # 8. Sector-relative market features (Phase 3 #2). Computed on the final
    #    cleaned panel so the peer group = firms actually in the model.
    panel = _add_industry_relative_features(panel)

    return panel


def time_split(
    df: pd.DataFrame,
    train_end: int = TRAIN_END_YEAR,
    val_end: int = VAL_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition the panel by year.

    train <= train_end; val in (train_end, val_end]; test > val_end.
    Splits by TIME not FIRM — every firm appears in every window, which is
    what lets the models avoid learning firm identity from training.
    """
    train = df[df["year"] <= train_end].copy()
    val = df[(df["year"] > train_end) & (df["year"] <= val_end)].copy()
    test = df[df["year"] > val_end].copy()

    print(f"\nTime split:")
    print(f"  Train: {len(train)} rows ({train['year'].min()}-{train['year'].max()}), "
          f"event rate: {train[LABEL_COL].mean():.1%}")
    print(f"  Val:   {len(val)} rows ({val['year'].min()}-{val['year'].max()}), "
          f"event rate: {val[LABEL_COL].mean():.1%}")
    if len(test) > 0:
        print(f"  Test:  {len(test)} rows ({test['year'].min()}-{test['year'].max()}), "
              f"event rate: {test[LABEL_COL].mean():.1%}")
    else:
        print(f"  Test:  0 rows")
    return train, val, test
