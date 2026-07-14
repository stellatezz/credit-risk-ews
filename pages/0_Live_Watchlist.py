"""Live Credit Risk Early-Warning Monitor — the analyst-facing product.

Pages 1-9 evaluate the model; this page IS the model deployed: the committed
pooled-logit coefficients applied to live market data, EDGAR fundamentals,
and held macro inputs, for the panel universe plus any analyst-added ticker.

Presentation-tuned: light theme (.streamlit/config.toml), collapsed sidebar,
hidden Streamlit chrome, ECharts drill-down visuals.

Before a live demo, run:  python scripts/warm_live_cache.py
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from src.ews import scoring
from src.ews.config import LABEL_A_THRESHOLD, LABEL_A_HORIZON_MONTHS, TRAIN_END_YEAR

st.set_page_config(page_title="Live Watchlist", page_icon="🚨", layout="wide",
                   initial_sidebar_state="expanded")

# Dataviz palette (validated for the light surface): blue = series /
# risk-easing, red = risk-raising; grays are recessive chrome.
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#898781"
INK, INK2, GRID, BASELINE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

BAND_ALERT, BAND_ELEVATED, BAND_NORMAL = "🔴 Alert", "🟠 Elevated", "🟢 Normal"

# Presentation mode: hide Streamlit's toolbar/menu/footer (keep the header so
# the sidebar can still be reopened), tighten the content column.
st.markdown("""<style>
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
#MainMenu, footer {visibility: hidden; height: 0;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1320px;}
h1 {font-size: 1.85rem !important; letter-spacing: -0.01em;}
</style>""", unsafe_allow_html=True)


# =============================================================================
# Cached data loading
# =============================================================================

@st.cache_data(show_spinner="Loading model + panel history…")
def _model_context():
    panel = scoring.load_scored_panel()
    coefs = scoring.load_coefficients()
    ref = scoring.build_reference(panel)
    meta = scoring.firm_directory(panel)
    return panel, coefs, ref, meta


@st.cache_data(show_spinner="Scoring the watchlist on live data… "
                            "(first build fetches prices + filings, ~2 min; "
                            "cached afterwards)")
def _scores(fingerprint: str, force: bool):
    return scoring.build_live_scores(force=force)


def _disk_fingerprint() -> str:
    """Changes whenever the live caches change on disk — even from outside the
    app (warm script, another session) — so st.cache_data can't go stale."""
    parts = []
    for p in (os.path.join(scoring.LIVE_DIR, "live_meta.json"),
              scoring.CUSTOM_WATCHLIST_PATH):
        parts.append(str(os.path.getmtime(p)) if os.path.exists(p) else "none")
    return "|".join(parts)


def _rerun_with_fresh_data(force: bool = False):
    st.session_state["force_fetch"] = force
    _model_context.clear()
    st.rerun()


panel, coefs, ref, meta = _model_context()
scores, status = _scores(
    _disk_fingerprint(),
    st.session_state.pop("force_fetch", False),
)
custom = scoring.load_custom_watchlist()
panel_end = panel["date"].max()


# =============================================================================
# Sidebar — demo utilities only (collapsed during the presentation)
# =============================================================================

with st.sidebar:
    if st.button("🔄 Refresh live data", width="stretch"):
        _rerun_with_fresh_data(force=True)

    if custom:
        with st.expander("Manage added firms"):
            drop = st.selectbox("Remove from watchlist", sorted(custom))
            if st.button("Remove", width="stretch"):
                scoring.remove_custom_firm(drop)
                _rerun_with_fresh_data()

    st.divider()
    st.caption(
        f"**Model:** pooled logistic regression (deployed), trained on "
        f"2010–{TRAIN_END_YEAR}, 15 interpretable features. "
        f"**Event:** ≥{abs(LABEL_A_THRESHOLD):.0%} peak-to-trough equity "
        f"drawdown within {LABEL_A_HORIZON_MONTHS} months. "
        f"Prioritisation tool for analyst attention — not a trading signal."
    )


# =============================================================================
# Alert threshold — read from session first so bands are computed before the
# popover that edits it is rendered (Streamlit reruns on any change).
# =============================================================================

ops = scoring.load_operating_points()
op_labels = list(ops["label"]) + ["Custom"]
_choice = st.session_state.get("op_choice", op_labels[0])
if _choice == "Custom":
    thr = float(st.session_state.get("custom_thr", 0.163))
else:
    _row = ops[ops["label"] == _choice]
    thr = float(_row["thr"].iloc[0]) if len(_row) else float(ops["thr"].iloc[0])


# =============================================================================
# Assemble the watchlist view (one row per firm, latest month)
# =============================================================================

scores = scores.sort_values(["ticker", "date"])
latest = scores.groupby("ticker").tail(1).set_index("ticker")
prev_pd = (
    scores.groupby("ticker").tail(2).groupby("ticker").head(1)
    .set_index("ticker")["pd_score"]
)
spark = scores.groupby("ticker")["pd_score"].apply(lambda s: list(s.tail(12)))
as_of = latest["date"].max()

FINANCIAL_TAGS = ("financ", "bank", "insur")


def _notes(t, row) -> str:
    tags = []
    if t in custom:
        tags.append("➕ added live")
        sector = str(custom[t].get("sector", "")).lower()
        if any(k in sector for k in FINANCIAL_TAGS):
            tags.append("⚠️ outside training universe")
    if bool(row.get("fundamentals_imputed")):
        tags.append("filings unavailable — fundamentals imputed")
    if row["date"] < as_of - pd.Timedelta(days=45):
        tags.append(f"⏸ stale (data to {row['date']:%b %Y})")
    return " · ".join(tags)


view_rows = []
for t, row in latest.iterrows():
    p, p_prev = float(row["pd_score"]), float(prev_pd.get(t, np.nan))
    decile = int(scoring.pd_decile(p, ref)[0])
    is_new = p >= thr and (np.isnan(p_prev) or p_prev < thr)
    band = BAND_ALERT if p >= thr else (BAND_ELEVATED if decile == 10 else BAND_NORMAL)
    info = meta.loc[t] if t in meta.index else None
    view_rows.append({
        "Band": band + (" · NEW" if is_new else ""),
        "Ticker": t,
        "Firm": info["firm_name"] if info is not None else t,
        "Sector": info["industry"] if info is not None else "—",
        "PD": p,
        "Decile": decile,
        "Δ 1m (pp)": None if np.isnan(p_prev) else (p - p_prev) * 100,
        "12m trend": spark.get(t, []),
        "Why (top drivers)": scoring.top_drivers(row, coefs, ref, compact=True),
        "Notes": _notes(t, row),
        "_new": is_new,
    })

watch = pd.DataFrame(view_rows).sort_values("PD", ascending=False).reset_index(drop=True)
watch.insert(0, "#", watch.index + 1)

n_alert = int((watch["PD"] >= thr).sum())
n_elevated = int(watch["Band"].str.startswith("🟠").sum())
new_tickers = list(watch.loc[watch["_new"], "Ticker"])


# =============================================================================
# Header + warning banner + KPI strip (with the threshold control in a popover)
# =============================================================================

data_through = status.get("fresh_through") or f"{as_of:%Y-%m-%d}"
st.title("🚨 Credit Risk Early-Warning Monitor")
st.caption(
    f"**{as_of:%B %Y}** monitoring run · "
    f"prices {'🔴 offline — cached through' if status.get('offline') else '🟢 live through'} "
    f"{data_through} · "
    f"fundamentals 🟡 latest SEC filings (EDGAR) · "
    f"macro 🟠 held at {panel_end:%b %Y} model values"
)

if new_tickers:
    st.markdown(
        f"""<div style="background:rgba(227,73,72,.07);border-left:4px solid {RED};
        border-radius:8px;padding:.8rem 1rem;margin:.3rem 0 1rem 0;">
        <b>⚠️ {len(new_tickers)} new warning{'s' if len(new_tickers) > 1 else ''} this
        month:</b> {', '.join(new_tickers)} — crossed the alert threshold for the
        first time. A new, rising alert is the highest-priority review.</div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""<div style="background:rgba(137,135,129,.07);border-left:4px solid {GRAY};
        border-radius:8px;padding:.8rem 1rem;margin:.3rem 0 1rem 0;">
        <b>No new warnings this month.</b> {n_alert} firm{'s' if n_alert != 1 else ''}
        remain above the alert threshold — review for changes in drivers.</div>""",
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1], vertical_alignment="center")
k1.metric("Firms monitored", len(watch))
k2.metric("🔴 Above threshold", n_alert)
k3.metric("🟠 Top decile, below threshold", n_elevated)
k4.metric("⚡ New warnings", len(new_tickers))
with k5.popover("⚙️ Alert threshold", width="stretch"):
    pick = st.radio("Operating point (from the Phase-3 cost analysis)",
                    op_labels, key="op_choice")
    st.slider("Custom threshold (PD)", 0.02, 0.60, 0.163, 0.001,
              key="custom_thr", format="%.3f",
              disabled=(pick != "Custom"))
    if pick != "Custom":
        op = ops[ops["label"] == pick].iloc[0]
        st.caption(
            f"Threshold **{op['thr']:.3f}** — on held-out 2024 data this catches "
            f"**{op['recall']:.0%}** of deteriorations at **{op['precision']:.0%}** "
            f"precision (flags ~{op['flag_budget']:.0%} of firm-months)."
        )


# =============================================================================
# Add a firm — live scoring of any US-listed ticker
# =============================================================================

with st.form("add_firm", clear_on_submit=True, border=True):
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    new_ticker = c1.text_input(
        "➕ **Add any US-listed ticker to the watchlist** — fetched, scored by the "
        "model, and ranked in seconds",
        placeholder="e.g. COIN, PLTR, DELL",
    )
    submitted = c2.form_submit_button("Add & score", width="stretch")

if submitted and new_ticker:
    try:
        with st.spinner(f"Scoring {new_ticker.upper()}: fetching prices + SEC filings…"):
            info = scoring.add_custom_firm(new_ticker)
        st.session_state["firm_picker"] = new_ticker.strip().upper()
        st.toast(f"{new_ticker.upper()} ({info['name']}) added and scored ✅")
        _rerun_with_fresh_data()
    except ValueError as e:
        st.error(str(e))


# =============================================================================
# The ranked watchlist — with a find-a-firm filter row
# =============================================================================

st.subheader("Watchlist — ranked by model risk score")

f1, f2, f3 = st.columns([2.4, 2.4, 1.6], vertical_alignment="center")
query = f1.text_input("Search", placeholder="🔎 Search ticker or firm…",
                      label_visibility="collapsed")
band_pick = f2.segmented_control(
    "Status", ["All", "🔴 Alerts", "🟠 Elevated", "🟢 Normal"],
    default="All", label_visibility="collapsed")
sector_pick = f3.selectbox(
    "Sector", ["All sectors"] + sorted(watch["Sector"].astype(str).unique()),
    label_visibility="collapsed")

filtered = watch
if query:
    q = query.strip().lower()
    filtered = filtered[
        filtered["Ticker"].str.lower().str.contains(q, regex=False)
        | filtered["Firm"].str.lower().str.contains(q, regex=False)
    ]
if band_pick and band_pick != "All":
    filtered = filtered[filtered["Band"].str.startswith(band_pick[0])]
if sector_pick != "All sectors":
    filtered = filtered[filtered["Sector"] == sector_pick]

# A new widget key per filter state resets stale row-selections when the
# visible rows change.
table_key = f"watch_table_{hash((query, band_pick, sector_pick))}"
if st.session_state.get("_table_key") != table_key:
    st.session_state["_table_key"] = table_key
    st.session_state["_last_row_sel"] = None

table_df = filtered.drop(columns="_new").reset_index(drop=True)
if len(table_df) == 0:
    st.info("No firms match the current filters.")
    event = None
else:
    event = st.dataframe(
        table_df,
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(table_df), 563),
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Band": st.column_config.TextColumn("Status", width="medium",
                                                help="🔴 above alert threshold · 🟠 riskiest tenth of history · 🟢 normal"),
            "Ticker": st.column_config.TextColumn(width="small"),
            "PD": st.column_config.ProgressColumn(
                "PD (12m)", format="percent", min_value=0.0, max_value=1.0,
                help="Model probability of a ≥40% equity drawdown within 12 months"),
            "Decile": st.column_config.NumberColumn(
                "Decile", width="small",
                help="Rank vs all 2010–2020 training firm-months; 10 = riskiest tenth"),
            "Δ 1m (pp)": st.column_config.NumberColumn("Δ 1m (pp)", format="%+.1f",
                                                       help="Change vs prior month, percentage points"),
            "12m trend": st.column_config.LineChartColumn("12m trend", y_min=0.0, y_max=0.6),
            "Why (top drivers)": st.column_config.TextColumn("Why (top drivers)", width="large",
                                                             help="▲ pushes risk up · ▼ pulls risk down — read from model coefficients"),
            "Notes": st.column_config.TextColumn(width="medium"),
        },
    )

with st.expander("📖 How to read this screen (analyst guide)"):
    st.markdown(f"""
**What PD means.** The model's estimate of the probability that this firm suffers a
**≥40% peak-to-trough equity drawdown at some point in the next 12 months** — the
market-implied credit-deterioration event the model was trained on. *"PD 32%"* reads:
of all historical firm-months that looked statistically like this firm today, roughly
one in three went on to a 40% collapse within a year.

**Read it in three steps:**
1. **Band before number** — calibration is imperfect, ranking is what's validated. 🔴 = above
   the chosen alert threshold ({thr:.3f}); 🟠 = in the riskiest tenth of training history but
   below the threshold; 🟢 = normal. Anchor: the training base rate is ~{ref['base_rate']:.0%}
   — decile 10 concentrates ~2.4× that event rate.
2. **Direction next** — a fresh 🔴 **NEW** crossing outranks a chronically high score. Rising
   three months in a row = deteriorating; use the sparkline and Δ column.
3. **Then the why** — the drivers show *which* features create the warning. A market-driven
   warning (drawdown, volatility) and a balance-sheet warning (leverage, profitability)
   prompt different reviews. Click a row for the full picture.

**Honest limits.** Strength is **top-decile triage** — the model reliably beats the Altman-Z
benchmark there. Ranking *within* stable/cyclical/growth firms is close to random, so treat
mid-table ordering with caution. Not a trading signal, not a default-timing predictor.
""")


# =============================================================================
# Drill-down — the model's full reasoning for one firm (ECharts visuals)
# =============================================================================

# Two ways to drive the drill-down: tick a row in the table, or pick from the
# dropdown. Only a *change* in table selection wins, so the dropdown still works
# while a row stays ticked.
options = list(watch["Ticker"])
row_sel = event.selection.rows[0] if (event and event.selection.rows) else None
if row_sel is not None and st.session_state.get("_last_row_sel") != row_sel:
    st.session_state["firm_picker"] = table_df.iloc[row_sel]["Ticker"]
st.session_state["_last_row_sel"] = row_sel
if st.session_state.get("firm_picker") not in options:
    st.session_state["firm_picker"] = options[0]

st.divider()
firm_labels = dict(zip(watch["Ticker"], watch["Firm"]))
pick_col, _ = st.columns([1.4, 3.6])
sel = pick_col.selectbox(
    "🔍 Inspect a firm (or tick a row above)",
    options,
    format_func=lambda t: f"{t} — {firm_labels.get(t, t)}",
    key="firm_picker",
)

sel_row = latest.loc[sel]
sel_view = watch[watch["Ticker"] == sel].iloc[0]
sel_pd = float(sel_row["pd_score"])
sel_pct = float(scoring.pd_percentile(sel_pd, ref)[0])
sel_pct_str = "99%+" if sel_pct >= 99.5 else f"{sel_pct:.0f}%"
sel_asof = data_through if bool(sel_row["is_live"]) else f"{sel_row['date']:%d %b %Y}"
thr_pct = thr * 100

st.subheader(f"{sel} — {sel_view['Firm']}")
st.caption(f"{sel_view['Sector']} · {sel_view['Band']}"
           + (f" · {sel_view['Notes']}" if sel_view["Notes"] else ""))

g0, g1, g2, g3 = st.columns([1.6, 1, 1, 1], vertical_alignment="center")
with g0:
    st_echarts({
        "series": [{
            "type": "gauge",
            "startAngle": 210, "endAngle": -30,
            "min": 0, "max": 100,
            "axisLine": {"lineStyle": {"width": 16, "color": [
                [thr_pct / 100, "#e3ecdf"], [1.0, "#f7dbda"]]}},
            "pointer": {"length": "60%", "width": 5,
                        "itemStyle": {"color": INK}},
            "anchor": {"show": True, "size": 7, "itemStyle": {"color": INK}},
            "axisTick": {"show": False},
            "splitLine": {"show": False},
            "axisLabel": {"show": False},
            "title": {"offsetCenter": [0, "80%"], "fontSize": 12, "color": INK2},
            "detail": {"valueAnimation": True, "formatter": "{value}%",
                       "fontSize": 26, "fontWeight": 600,
                       "offsetCenter": [0, "44%"], "color": INK},
            "data": [{"value": round(sel_pd * 100, 1), "name": "PD (next 12m)"}],
        }],
    }, height="210px", key="pd_gauge")
g1.metric("Risk decile", f"{sel_view['Decile']} / 10",
          help="vs all 2010–2020 training firm-months; 10 = riskiest tenth")
g2.metric("Riskier than", sel_pct_str, help="of all training firm-months")
g3.metric("Δ vs last month",
          "—" if pd.isna(sel_view["Δ 1m (pp)"]) else f"{sel_view['Δ 1m (pp)']:+.1f} pp",
          delta=None if pd.isna(sel_view["Δ 1m (pp)"]) else f"{sel_view['Δ 1m (pp)']:+.1f} pp",
          delta_color="inverse", label_visibility="visible")

st.markdown(
    f"> The model estimates a **{sel_pd:.0%}** probability of a ≥40% equity drawdown "
    f"within the next 12 months — riskier than **{sel_pct_str}** of everything it saw "
    f"in training. Every input was available by {sel_asof}; no look-ahead."
)

c_left, c_right = st.columns([3, 2])

with c_left:
    st.markdown("**Risk trajectory**")
    traj = scores[scores["ticker"] == sel][["date", "pd_score", "label_a", "is_live"]].copy()
    traj = traj.sort_values("date")
    pairs = [[d.strftime("%Y-%m-%d"), round(p * 100, 2)]
             for d, p in zip(traj["date"], traj["pd_score"])]

    # Merge consecutive realised-distress months into shaded episodes.
    episodes, ep_start, ep_prev = [], None, None
    for d in traj.loc[traj["label_a"] == 1, "date"]:
        if ep_start is None:
            ep_start = ep_prev = d
        elif (d - ep_prev).days <= 35:
            ep_prev = d
        else:
            episodes.append((ep_start, ep_prev))
            ep_start = ep_prev = d
    if ep_start is not None:
        episodes.append((ep_start, ep_prev))
    mark_areas = [
        [{"xAxis": s.strftime("%Y-%m-%d")},
         {"xAxis": (e + pd.DateOffset(months=1)).strftime("%Y-%m-%d")}]
        for s, e in episodes
    ]

    mark_lines = [{"yAxis": round(thr_pct, 1),
                   "label": {"formatter": "alert threshold", "color": INK2,
                             "fontSize": 10}}]
    if bool(traj["is_live"].any()):
        mark_lines.append({"xAxis": panel_end.strftime("%Y-%m-%d"),
                           "lineStyle": {"type": "dotted"},
                           "label": {"formatter": "live →", "color": INK2,
                                     "fontSize": 10}})

    st_echarts({
        "animationDuration": 600,
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 44, "right": 18, "top": 18, "bottom": 36},
        "xAxis": {"type": "time",
                  "axisLine": {"lineStyle": {"color": BASELINE}},
                  "axisLabel": {"color": INK2}},
        "yAxis": {"type": "value",
                  "axisLabel": {"formatter": "{value}%", "color": INK2},
                  "splitLine": {"lineStyle": {"color": GRID}}},
        "series": [{
            "name": "PD (%)", "type": "line", "showSymbol": False,
            "data": pairs,
            "lineStyle": {"width": 2, "color": BLUE},
            "itemStyle": {"color": BLUE},
            "areaStyle": {"color": {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [
                    {"offset": 0, "color": "rgba(42,120,214,0.20)"},
                    {"offset": 1, "color": "rgba(42,120,214,0.02)"}]}},
            "markLine": {"silent": True, "symbol": "none",
                         "lineStyle": {"type": "dashed", "color": GRAY},
                         "data": mark_lines},
            "markArea": {"silent": True,
                         "itemStyle": {"color": "rgba(227,73,72,0.10)"},
                         "data": mark_areas},
        }],
    }, height="300px", key="traj_chart")
    st.caption("Shaded: months where a ≥40% drawdown followed within 12m · "
               "dashed: alert threshold · dotted: live scoring starts")

    px = scoring.price_history(sel)
    px = px[px["date"] >= as_of - pd.DateOffset(years=5)]
    if len(px):
        st.markdown("**Price context** — adjusted close, last 5 years")
        st_echarts({
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 52, "right": 18, "top": 10, "bottom": 32},
            "xAxis": {"type": "time",
                      "axisLine": {"lineStyle": {"color": BASELINE}},
                      "axisLabel": {"color": INK2}},
            "yAxis": {"type": "value", "scale": True,
                      "axisLabel": {"formatter": "${value}", "color": INK2},
                      "splitLine": {"lineStyle": {"color": GRID}}},
            "series": [{
                "name": "Adj. close", "type": "line", "showSymbol": False,
                "data": [[d.strftime("%Y-%m-%d"), round(v, 2)]
                         for d, v in zip(px["date"], px["adj_close"])],
                "lineStyle": {"width": 1.5, "color": GRAY},
                "itemStyle": {"color": GRAY},
                "areaStyle": {"color": "rgba(137,135,129,0.10)"},
            }],
        }, height="170px", key="price_chart")

with c_right:
    st.markdown("**Why? — what pushes this firm's risk score**")
    drivers = scoring.driver_table(sel_row, coefs, ref)
    drivers = drivers[drivers["contribution"].abs() > 1e-6].head(9)
    bar_data = [{
        "value": round(float(r.contribution), 2),
        "itemStyle": {
            "color": RED if r.contribution > 0 else BLUE,
            "opacity": 0.45 if r.systemic else 1.0,
            "borderRadius": [0, 3, 3, 0] if r.contribution > 0 else [3, 0, 0, 3],
        },
        "label": {"show": True,
                  "position": "right" if r.contribution > 0 else "left",
                  "color": INK2, "fontSize": 10},
    } for r in drivers.itertuples()]

    st_echarts({
        "animationDuration": 600,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 150, "right": 44, "top": 8, "bottom": 28},
        "xAxis": {"type": "value",
                  "axisLabel": {"color": INK2},
                  "splitLine": {"lineStyle": {"color": GRID}}},
        "yAxis": {"type": "category", "inverse": True,
                  "data": list(drivers["label"]),
                  "axisTick": {"show": False}, "axisLine": {"show": False},
                  "axisLabel": {"color": INK, "fontSize": 11}},
        "series": [{"name": "push on log-odds", "type": "bar",
                    "barWidth": 14, "data": bar_data}],
    }, height="300px", key="driver_chart")
    st.caption("Red pushes risk up · blue pulls it down · faded = market-wide "
               "conditions · read directly from the regression coefficients")

    with st.expander("Feature values vs training history"):
        detail = drivers[["label", "value", "train_median"]].copy()
        detail["top-decile median"] = [
            float(ref["top_decile_medians"][f]) for f in drivers["feature"]]
        st.dataframe(
            detail.rename(columns={"label": "Feature", "value": "This firm",
                                   "train_median": "Typical firm"}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.3f")
                           for c in ["This firm", "Typical firm", "top-decile median"]},
        )

st.divider()
st.caption(
    f"Alert threshold {thr:.3f} · deployed pooled logistic regression · open data only "
    f"(yfinance / SEC EDGAR / panel macro) · Known limitation: within-sector ranking for "
    f"stable/cyclical/growth firms is near-random — this screen is a top-decile triage "
    f"tool, not a universal ranker. Not a trading signal."
)
