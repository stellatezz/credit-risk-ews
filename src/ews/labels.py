"""
Label construction.

Owns the logic that turns raw prices (or future SEC 8-K filings) into binary
distress labels. `load_labels` in loaders.py dispatches here by `source=` kwarg.

Phase 1 implements Label A (forward drawdown) only; `label_b` is reserved
for Darren's 8-K Item 1.03 (bankruptcy) loader and stays NaN until then.
Pipeline tolerates the NaN column.

Label A definition (preserved bit-identical from the monolith):
    For each firm-month, check whether the forward 13-month window
    (current + next 12 months) contains a peak-to-trough drawdown
    >= |threshold|. Label = 1 if yes, else 0.
"""

import numpy as np
import pandas as pd

from .config import FIRMS


def compute_label_a_from_prices(
    prices_df: pd.DataFrame,
    horizon_months: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Build Label A from long-format prices.

    Returns: DataFrame with columns [ticker, date, label_a,
             forward_max_drawdown, label_b].  label_b is filled with NaN.

    Implementation pivots to WIDE internally to reuse the original per-firm
    drawdown math — keeps the baseline panel SHA256 stable.
    """
    print(f"\nComputing Label A (forward drawdown >= {abs(threshold):.0%})...")

    wide = (
        prices_df.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )
    monthly_prices = wide.resample("ME").last()

    records = []
    for ticker in monthly_prices.columns:
        series = monthly_prices[ticker].dropna()
        for i in range(len(series) - horizon_months):
            current_date = series.index[i]
            forward_window = series.iloc[i:i + horizon_months + 1]
            if len(forward_window) < 6:
                continue

            running_max = forward_window.cummax()
            drawdowns = (forward_window - running_max) / running_max
            max_drawdown = drawdowns.min()

            label = 1 if max_drawdown <= threshold else 0

            records.append({
                "ticker": ticker,
                "date": current_date,
                "label_a": label,
                "forward_max_drawdown": max_drawdown,
                "label_b": np.nan,  # Darren's 8-K labels go here
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    event_rate = df["label_a"].mean()
    print(f"  Labels computed: {len(df)} firm-months, event rate = {event_rate:.1%}")

    firm_events = df.groupby("ticker")["label_a"].agg(["mean", "sum", "count"])
    firm_events.columns = ["event_rate", "events", "months"]
    print("\n  Per-firm event rates:")
    for ticker, row in firm_events.iterrows():
        name = FIRMS.get(ticker, {}).get("name", ticker)
        bar = "#" * int(row["event_rate"] * 30)
        print(f"    {ticker:5s} ({name:25s}): {row['event_rate']:5.1%} "
              f"({int(row['events']):3d}/{int(row['months']):3d}) {bar}")

    return df


# -----------------------------------------------------------------------------
# Darren: your 8-K Label B loader goes here.
#
# Signature should be:
#
#   def compute_label_b_from_8k(filings_df: pd.DataFrame,
#                                horizon_months: int) -> pd.DataFrame:
#       """Returns (ticker, date, label_b) — 1 if firm filed 8-K Item 1.03
#          (bankruptcy) within the next `horizon_months`."""
#
# Then loaders.py `load_labels(source='8k')` dispatches here and merges
# label_b into the existing Label A DataFrame on (ticker, date).
# -----------------------------------------------------------------------------
