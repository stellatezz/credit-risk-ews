"""Prediction horizon sensitivity analysis page.

Recomputes Label A from prices for selected horizons (3, 6, 12 months) and
reports event rates, per-firm event distributions, and overlap between horizons.

This page is lightweight: it only computes labels from `data/interim/prices.csv`
and summarises where events fall as horizon changes. It does not retrain models
— use `python src/run.py` to regenerate full evaluation artifacts if needed.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ews.labels import compute_label_a_from_prices
from src.ews.config import LABEL_A_THRESHOLD


st.set_page_config(page_title="Prediction Horizon Analysis", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent
PRICES_CSV = REPO_ROOT / "data" / "interim" / "prices.csv"


def _missing(path: Path) -> bool:
    if not path.exists():
        st.error(f"⚠️ `{path.name}` not found. Run the pipeline to create it (e.g. `python src/run.py`).")
        return True
    return False


st.title("⏳ Prediction Horizon Analysis")

st.markdown(
    """
This tool recomputes the forward-drawdown label (Label A) for alternative
prediction horizons and summarises how the event prevalence and per-firm
event distribution change. Useful for understanding how horizon selection
affects base rates and analyst workload.
"""
)

st.markdown("---")

# User controls
horizons = st.multiselect("Select horizons (months)", options=[3, 6, 12], default=[3, 6, 12])
st.caption("Note: the label routine skips forward windows shorter than 6 months; very short horizons (e.g. 3m) may therefore produce no labels.")
threshold = st.number_input("Drawdown threshold (negative fraction)",
                            value=float(LABEL_A_THRESHOLD), format="%.2f",
                            help="Peak-to-trough drawdown threshold used to mark an event (e.g. -0.40 = 40%).")

run = st.button("Compute labels & summarise")

if run:
    if _missing(PRICES_CSV):
        st.stop()

    with st.spinner("Loading prices and computing labels — this may take a moment..."):
        # read only the necessary columns to keep memory usage small
        # the prices.csv begins with a comment line; tell pandas to ignore comment lines
        prices = pd.read_csv(PRICES_CSV, usecols=["ticker", "date", "adj_close"], parse_dates=["date"], comment="#")  # long format: ticker,date,adj_close

        results = {}
        labels_by_h = {}
        for h in sorted(horizons):
            try:
                lbl = compute_label_a_from_prices(prices, horizon_months=int(h), threshold=float(threshold))
            except Exception as e:
                st.warning(f"Failed to compute labels for horizon={h}: {e}")
                continue

            if lbl is None or lbl.empty:
                st.warning(f"No labels produced for horizon={h} (forward-window too short or no data)")
                continue

            labels_by_h[h] = lbl
            total_rows = len(lbl)
            total_events = int(lbl["label_a"].sum())
            event_rate = lbl["label_a"].mean()
            per_firm = lbl.groupby("ticker")["label_a"].sum()
            results[h] = {
                "rows": total_rows,
                "events": total_events,
                "event_rate": event_rate,
                "firms_with_event_pct": (per_firm > 0).mean(),
                "median_events_per_firm": per_firm.median(),
                "mean_events_per_firm": per_firm.mean(),
            }

    # Summary table
    summary = pd.DataFrame.from_dict(results, orient="index")
    summary = summary.rename_axis("horizon_months").reset_index()
    st.header("Summary: event rates & per-firm statistics")
    st.table(summary.style.format({"event_rate": "{:.2%}", "firms_with_event_pct": "{:.2%}",
                                    "median_events_per_firm": "{:.1f}", "mean_events_per_firm": "{:.1f}"}))

    # Top firms per horizon
    st.header("Top firms by event count for each horizon")
    for h, df in labels_by_h.items():
        st.subheader(f"Horizon = {h} months")
        top = df.groupby("ticker")["label_a"].sum().sort_values(ascending=False).head(10).reset_index()
        top.columns = ["ticker", "events"]
        st.dataframe(top, hide_index=True)

    # Overlap matrix (pairwise)
    if len(labels_by_h) >= 2:
        st.header("Overlap between horizons")
        # Build wide table of indicators per (ticker,date)
        merged = None
        for h, df in labels_by_h.items():
            tmp = df[["ticker", "date", "label_a"]].rename(columns={"label_a": f"h{h}"})
            if merged is None:
                merged = tmp
            else:
                merged = pd.merge(merged, tmp, on=["ticker", "date"], how="outer")

        merged = merged.fillna(0)
        # pairwise intersection counts
        pairs = []
        hs = sorted(labels_by_h.keys())
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                a, b = hs[i], hs[j]
                both = ((merged[f"h{a}"] == 1) & (merged[f"h{b}"] == 1)).sum()
                only_a = ((merged[f"h{a}"] == 1) & (merged[f"h{b}"] == 0)).sum()
                only_b = ((merged[f"h{a}"] == 0) & (merged[f"h{b}"] == 1)).sum()
                pairs.append({"horizon_a": a, "horizon_b": b, "both": int(both), "only_a": int(only_a), "only_b": int(only_b)})

        st.dataframe(pd.DataFrame(pairs), hide_index=True)

    st.info("Tip: to evaluate model performance at different horizons (AUROC/AUPRC/top-K), run the full pipeline `python src/run.py` after changing `LABEL_A_HORIZON_MONTHS` in `src/ews/config.py` or by using this page to inspect label prevalence before retraining.")

