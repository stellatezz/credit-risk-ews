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

import pandas as pd

from .config import (
    FEATURE_COLS,
    FIRMS,
    LABEL_COL,
    PANEL_START_YEAR,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
)


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
