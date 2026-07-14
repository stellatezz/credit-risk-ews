"""
Live scoring engine for the analyst watchlist (pages/0_Live_Watchlist.py).

Scores any US-listed ticker with the DEPLOYED pooled logistic regression —
the committed coefficients in outputs/full_model_coefficients_pooled.csv
(fit on train <= TRAIN_END_YEAR) — using the same feature construction as
the pipeline:

  market        yfinance daily prices -> features.build_market_features
  fundamentals  SEC EDGAR companyfacts -> same ratio math as loaders.py
  macro         panel macro series, held at its last value beyond panel end
                (the panel macro block is synthetic; a live FRED feed would
                mix regimes the model never saw — held-constant is honest)

No look-ahead: every feature at month t uses only data available at t.
No silent synthetic fallback: a firm whose EDGAR fetch fails is scored with
train-median fundamentals and flagged `fundamentals_imputed=True` so the UI
can badge it.

Disk caches (all under data/interim/live/ — never touches data/raw/, so the
pipeline's SHA-baselined inputs stay untouched):
  prices_cache.csv + prices_meta.json    merged daily prices, same-day TTL
  live_scores.csv + live_meta.json       scored live firm-months
  custom_watchlist.json                  analyst-added tickers + metadata
  triage_log.csv                         escalate/monitor/dismiss decisions
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import datetime as dt

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, PATHS, TRAIN_END_YEAR, FIRMS
from .features import build_market_features
from . import loaders

LIVE_DIR = os.path.join(PATHS.INTERIM, "live")
COEF_PATH = os.path.join(PATHS.OUTPUTS, "full_model_coefficients_pooled.csv")
PANEL_PATH = os.path.join(PATHS.PROCESSED, "panel_phase2.csv")
OPERATING_POINTS_PATH = os.path.join(PATHS.OUTPUTS, "fp_fn_operating_points.csv")
CUSTOM_WATCHLIST_PATH = os.path.join(LIVE_DIR, "custom_watchlist.json")
TRIAGE_LOG_PATH = os.path.join(LIVE_DIR, "triage_log.csv")

# Fundamental ratios the model consumes (late_filing handled separately).
FUND_COLS = ["leverage", "liquidity_buffer", "wc_ratio", "profitability"]
MACRO_COLS = ["vix", "term_spread", "credit_spread"]
MARKET_COLS = ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"]

# How far back to fetch prices for an analyst-added firm (drives its PD
# trajectory depth; features need 252 trading days before the first month).
CUSTOM_HISTORY_START = "2017-01-01"

# Short labels for dense UI spots (watchlist driver chips).
FEATURE_LABELS_COMPACT = {
    "ret_1m": "1m return", "ret_3m": "3m return", "ret_6m": "6m return",
    "vol_3m": "3m volatility", "vol_6m": "6m volatility",
    "drawdown_12m": "12m drawdown",
    "leverage": "leverage", "liquidity_buffer": "liquidity",
    "wc_ratio": "working capital", "wc_ratio_missing": "WC unreported",
    "profitability": "profitability", "late_filing": "late filing",
    "vix": "VIX", "term_spread": "term spread", "credit_spread": "credit spread",
}

# Human-readable feature names for the UI.
FEATURE_LABELS = {
    "ret_1m": "1-month return",
    "ret_3m": "3-month return",
    "ret_6m": "6-month return",
    "vol_3m": "3-month volatility",
    "vol_6m": "6-month volatility",
    "drawdown_12m": "12-month drawdown",
    "leverage": "Leverage (Liab/Assets)",
    "liquidity_buffer": "Liquidity (Cash/Assets)",
    "wc_ratio": "Working-capital ratio",
    "wc_ratio_missing": "WC ratio unreported",
    "profitability": "Profitability (NI/Assets)",
    "late_filing": "Late SEC filing",
    "vix": "VIX (systemic)",
    "term_spread": "Term spread (systemic)",
    "credit_spread": "Credit spread (systemic)",
}


def _ensure_live_dir() -> None:
    os.makedirs(LIVE_DIR, exist_ok=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# =============================================================================
# Model: coefficients + scoring
# =============================================================================

def load_coefficients(path: str = COEF_PATH) -> pd.Series:
    """Deployed pooled-logit coefficients, indexed by feature (incl. 'const')."""
    coefs = pd.read_csv(path).set_index("feature")["coef"]
    missing = [c for c in FEATURE_COLS if c not in coefs.index]
    if missing or "const" not in coefs.index:
        raise ValueError(f"coefficient file {path} missing terms: {missing}")
    return coefs


def score_features(df: pd.DataFrame, coefs: pd.Series) -> pd.Series:
    """PD for each row of a frame that carries all FEATURE_COLS."""
    beta = coefs.reindex(FEATURE_COLS).to_numpy(dtype=float)
    logit = coefs["const"] + df[FEATURE_COLS].to_numpy(dtype=float) @ beta
    return pd.Series(_sigmoid(logit), index=df.index, name="pd_score")


def load_scored_panel() -> pd.DataFrame:
    """The committed Phase-2 panel with a pd_score column added."""
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    panel["pd_score"] = score_features(panel, load_coefficients())
    return panel


# =============================================================================
# Reference distribution (train window) — percentiles, deciles, driver medians
# =============================================================================

def build_reference(panel_scored: pd.DataFrame) -> dict:
    """Training-window reference stats used to contextualise a live PD."""
    train = panel_scored[panel_scored["year"] <= TRAIN_END_YEAR]
    pds = np.sort(train["pd_score"].to_numpy())
    decile_edges = np.quantile(pds, np.arange(0.1, 1.0, 0.1))
    top_decile = train[train["pd_score"] >= decile_edges[-1]]
    return {
        "train_pds": pds,
        "decile_edges": decile_edges,
        "feature_medians": train[FEATURE_COLS].median(),
        "top_decile_medians": top_decile[FEATURE_COLS].median(),
        "base_rate": float(train["label_a"].mean()),
    }


def pd_percentile(values, ref: dict) -> np.ndarray:
    """Percentile of PD value(s) within the training distribution (0-100)."""
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    pct = np.searchsorted(ref["train_pds"], arr, side="right") / len(ref["train_pds"])
    return pct * 100


def pd_decile(values, ref: dict) -> np.ndarray:
    """Risk decile 1 (safest) .. 10 (riskiest) vs the training distribution."""
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    return np.searchsorted(ref["decile_edges"], arr, side="right") + 1


def driver_table(row: pd.Series, coefs: pd.Series, ref: dict) -> pd.DataFrame:
    """Per-feature log-odds contributions vs the typical training firm-month.

    contribution_i = coef_i * (x_i - train_median_i): how much this feature
    pushes the firm's log-odds away from a typical firm-month. Macro terms
    are tagged systemic — identical for every firm in the same month.
    """
    med = ref["feature_medians"]
    rows = []
    for f in FEATURE_COLS:
        rows.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "value": float(row[f]),
            "train_median": float(med[f]),
            "contribution": float(coefs[f] * (row[f] - med[f])),
            "systemic": f in MACRO_COLS,
        })
    out = pd.DataFrame(rows)
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index)


def top_drivers(row: pd.Series, coefs: pd.Series, ref: dict, n: int = 3,
                compact: bool = False) -> str:
    """Compact firm-specific driver chips for the watchlist table.

    Skips systemic (macro) terms — they are identical across firms in a
    month, so they never explain a firm's rank.
    """
    tbl = driver_table(row, coefs, ref)
    tbl = tbl[(~tbl["systemic"]) & (tbl["contribution"].abs() > 1e-9)]
    chips = [
        f"{'▲' if r.contribution > 0 else '▼'} "
        + (FEATURE_LABELS_COMPACT.get(r.feature, r.label) if compact else r.label)
        for r in tbl.head(n).itertuples()
    ]
    return " · ".join(chips)


# =============================================================================
# Operating points (thresholds) from the Phase-3 FP/FN analysis
# =============================================================================

def load_operating_points() -> pd.DataFrame:
    """Named thresholds with held-out 2024 test-set stats, for the sidebar."""
    ops = pd.read_csv(OPERATING_POINTS_PATH).set_index("operating_point")
    friendly = {
        "top_10pct": "Top-decile watchlist (default)",
        "cost_opt_2:1": "Precision-first (flag few, FN:FP = 2:1)",
        "cost_opt_10:1": "Recall-first (flag many, FN:FP = 10:1)",
    }
    ops = ops.loc[[k for k in friendly if k in ops.index]].copy()
    ops["label"] = [friendly[k] for k in ops.index]
    return ops


# =============================================================================
# Live prices (yfinance) — reads repo cache, fetches only the gap
# =============================================================================

def _yf_download_long(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """One batched yfinance call -> long [ticker, date, close, adj_close]."""
    import yfinance as yf

    data = yf.download(
        tickers, start=start, end=end, progress=False,
        auto_adjust=False, group_by="column", threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["ticker", "date", "close", "adj_close"])

    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                sub = pd.DataFrame({
                    "date": data.index,
                    "close": data[("Close", t)].to_numpy(),
                    "adj_close": data[("Adj Close", t)].to_numpy(),
                })
            except KeyError:
                continue
            sub["ticker"] = t
            frames.append(sub.dropna(subset=["adj_close"]))
    else:
        sub = pd.DataFrame({
            "date": data.index,
            "close": data["Close"].to_numpy(),
            "adj_close": data["Adj Close"].to_numpy(),
        })
        sub["ticker"] = tickers[0]
        frames.append(sub.dropna(subset=["adj_close"]))

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", "adj_close"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out[["ticker", "date", "close", "adj_close"]]


def _read_repo_price_cache(ticker: str) -> pd.DataFrame | None:
    """Per-ticker pipeline cache in data/raw/ (read-only — never written here)."""
    path = os.path.join(PATHS.RAW, f"yfinance_{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["ticker"] = ticker
    return df[["ticker", "date", "close", "adj_close"]]


def fetch_prices(
    tickers: list[str],
    history_start: str,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Daily prices for `tickers` from history_start to today.

    Sources, in order: same-day live cache -> repo data/raw cache -> one
    batched yfinance fetch for whatever is missing. Network failure degrades
    to cached data with status['offline']=True rather than raising, so the
    dashboard still renders on a dead conference wifi.
    """
    _ensure_live_dir()
    cache_csv = os.path.join(LIVE_DIR, "prices_cache.csv")
    cache_meta = os.path.join(LIVE_DIR, "prices_meta.json")
    today = dt.date.today().isoformat()

    if not force and os.path.exists(cache_csv) and os.path.exists(cache_meta):
        with open(cache_meta) as f:
            meta = json.load(f)
        if meta.get("fetched_on") == today and set(tickers) <= set(meta.get("tickers", [])):
            prices = pd.read_csv(cache_csv, parse_dates=["date"])
            prices = prices[prices["ticker"].isin(tickers)]
            status = {"offline": meta.get("offline", False),
                      "fresh_through": meta.get("fresh_through")}
            return prices, status

    base_frames, gap_tickers, full_tickers = [], [], []
    gap_start = None
    for t in tickers:
        cached = _read_repo_price_cache(t)
        if cached is None or cached.empty:
            full_tickers.append(t)
            continue
        base_frames.append(cached[cached["date"] >= pd.Timestamp(history_start)])
        last = cached["date"].max()
        gap_tickers.append(t)
        gap_start = last if gap_start is None else min(gap_start, last)

    offline = False
    fresh: list[pd.DataFrame] = []
    try:
        if gap_tickers and gap_start is not None:
            fresh.append(_yf_download_long(
                gap_tickers, start=(gap_start + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        if full_tickers:
            fresh.append(_yf_download_long(full_tickers, start=history_start))
    except Exception:
        offline = True

    prices = pd.concat(base_frames + fresh, ignore_index=True) if (base_frames or fresh) \
        else pd.DataFrame(columns=["ticker", "date", "close", "adj_close"])
    prices = (
        prices.dropna(subset=["adj_close"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    fresh_through = str(prices["date"].max().date()) if len(prices) else None
    prices.to_csv(cache_csv, index=False)
    with open(cache_meta, "w") as f:
        json.dump({"fetched_on": today, "tickers": sorted(set(tickers)),
                   "offline": offline, "fresh_through": fresh_through}, f)
    return prices, {"offline": offline, "fresh_through": fresh_through}


# =============================================================================
# Live fundamentals (SEC EDGAR) — single-ticker, no synthetic fallback
# =============================================================================

def fetch_fundamentals_monthly(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Monthly fundamentals for one ticker via EDGAR companyfacts.

    Same ratio math, clipping, and quarterly->monthly forward-fill as the
    pipeline's `_fundamentals_sec_impl`, but for a single ticker and with NO
    placeholder fallback: any failure returns None so the caller imputes
    train medians and flags the row instead of silently inventing data.
    """
    try:
        cik = loaders._load_cik_map().get(ticker.upper())
        if cik is None:
            return None
        facts = loaders._load_company_facts(cik, ticker)

        concepts = {
            key: loaders._extract_concept_series(facts, primary, fallbacks)
            for key, (primary, fallbacks) in loaders._SEC_CONCEPTS.items()
        }
        all_dates: set[pd.Timestamp] = set()
        for s in concepts.values():
            if s is not None:
                all_dates.update(s.index)
        if not all_dates:
            return None

        q_index = pd.DatetimeIndex(sorted(all_dates))
        snap = pd.DataFrame(index=q_index)
        for key, s in concepts.items():
            snap[key] = s.reindex(q_index) if s is not None else np.nan

        ta, tl = snap["TotalAssets"], snap["TotalLiabilities"]
        with np.errstate(divide="ignore", invalid="ignore"):
            quarterly = pd.DataFrame({
                "leverage": np.where(ta != 0, tl / ta, np.nan),
                "liquidity_buffer": np.where(ta != 0, snap["Cash"] / ta, np.nan),
                "wc_ratio": np.where(
                    ta != 0, (snap["CurrentAssets"] - snap["CurrentLiabilities"]) / ta, np.nan),
                "profitability": np.where(ta != 0, snap["NetIncome"] / ta, np.nan),
            }, index=q_index)

        combined = q_index.union(dates).sort_values()
        monthly = quarterly.reindex(combined).ffill().reindex(dates)
        if monthly[["leverage"]].dropna().empty:
            return None

        clip_map = {"leverage": (0.0, 1.0), "liquidity_buffer": (0.0, 1.0),
                    "wc_ratio": (-1.0, 1.0), "profitability": (-0.5, 0.5)}
        for col, (lo, hi) in clip_map.items():
            monthly[col] = monthly[col].clip(lo, hi)

        monthly["late_filing"] = loaders._compute_late_filing_flag(facts, dates).values
        monthly["ticker"] = ticker
        return monthly.reset_index(names="date")
    except Exception:
        return None


# =============================================================================
# Live panel assembly + scoring
# =============================================================================

def macro_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel macro block by month; consumers ffill past the panel's end."""
    return (
        panel[["date"] + MACRO_COLS]
        .drop_duplicates(subset="date")
        .set_index("date")
        .sort_index()
    )


def _build_live_rows(
    tickers: list[str],
    prices: pd.DataFrame,
    months_after: dict[str, pd.Timestamp | None],
    macro: pd.DataFrame,
    ref: dict,
    coefs: pd.Series,
) -> pd.DataFrame:
    """Feature rows + PD for each ticker's months after months_after[ticker].

    months_after[t] = last panel date for panel firms (only newer months are
    live-built), or None for analyst-added firms (build the full history the
    price data supports, capped at the panel's first macro month).
    """
    with contextlib.redirect_stdout(io.StringIO()):  # silence pipeline prints
        mkt = build_market_features(prices[prices["ticker"].isin(tickers)])
    if mkt.empty:
        return pd.DataFrame()

    macro_start = macro.index.min()
    rows = []
    for t in tickers:
        m = mkt[mkt["ticker"] == t].sort_values("date")
        cutoff = months_after.get(t)
        m = m[m["date"] > cutoff] if cutoff is not None else m[m["date"] >= macro_start]
        if m.empty:
            continue

        dates = pd.DatetimeIndex(m["date"])
        fund = fetch_fundamentals_monthly(t, dates)
        imputed = fund is None
        if imputed:
            fund = pd.DataFrame({"date": dates})
            for c in FUND_COLS:
                fund[c] = ref["feature_medians"][c]
            fund["late_filing"] = 0
        m = m.merge(fund[["date"] + FUND_COLS + ["late_filing"]], on="date", how="left")

        # Same treatment as panel assembly: unreported working capital gets a
        # neutral 0 plus its missingness indicator; other missing fundamentals
        # fall back to train medians and mark the firm as imputed.
        m["wc_ratio_missing"] = m["wc_ratio"].isna().astype(int)
        m["wc_ratio"] = m["wc_ratio"].fillna(0.0)
        for c in ["leverage", "liquidity_buffer", "profitability"]:
            if m[c].isna().any():
                imputed = True
                m[c] = m[c].fillna(ref["feature_medians"][c])
        m["late_filing"] = m["late_filing"].fillna(0).astype(int)

        macro_held = macro.reindex(macro.index.union(dates)).ffill().reindex(dates)
        for c in MACRO_COLS:
            m[c] = macro_held[c].to_numpy()

        m["fundamentals_imputed"] = imputed
        rows.append(m)

    if not rows:
        return pd.DataFrame()
    live = pd.concat(rows, ignore_index=True).dropna(subset=FEATURE_COLS)
    live["pd_score"] = score_features(live, coefs)
    live["is_live"] = True
    return live


def load_custom_watchlist() -> dict[str, dict]:
    if not os.path.exists(CUSTOM_WATCHLIST_PATH):
        return {}
    with open(CUSTOM_WATCHLIST_PATH) as f:
        return json.load(f)


def _save_custom_watchlist(watchlist: dict[str, dict]) -> None:
    _ensure_live_dir()
    with open(CUSTOM_WATCHLIST_PATH, "w") as f:
        json.dump(watchlist, f, indent=2)


def build_live_scores(force: bool = False) -> tuple[pd.DataFrame, dict]:
    """The dashboard's single data source: scored panel history + live months.

    Returns (scores, status). `scores` has one row per firm-month with
    FEATURE_COLS, pd_score, label_a (NaN for live months), is_live and
    fundamentals_imputed flags. `status` reports data freshness for the UI.
    Live rows are cached on disk (same-day TTL) so the on-stage load is
    instant after running scripts/warm_live_cache.py.
    """
    _ensure_live_dir()
    panel = load_scored_panel()
    coefs = load_coefficients()
    ref = build_reference(panel)
    macro = macro_series(panel)
    custom = load_custom_watchlist()

    panel_tickers = sorted(panel["ticker"].unique())
    custom_tickers = [t for t in sorted(custom) if t not in panel_tickers]
    panel_end = panel["date"].max()

    cache_csv = os.path.join(LIVE_DIR, "live_scores.csv")
    cache_meta = os.path.join(LIVE_DIR, "live_meta.json")
    today = dt.date.today().isoformat()

    live, status = None, {"offline": False, "fresh_through": None, "built_on": today}
    if not force and os.path.exists(cache_csv) and os.path.exists(cache_meta):
        with open(cache_meta) as f:
            meta = json.load(f)
        if meta.get("built_on") == today and meta.get("tickers") == panel_tickers + custom_tickers:
            live = pd.read_csv(cache_csv, parse_dates=["date"])
            status.update({k: meta.get(k) for k in ("offline", "fresh_through")})

    if live is None:
        # Panel firms only need the gap after panel_end; the ~14-month price
        # runway before the first live month feeds the 252-day feature windows.
        runway_start = (panel_end - pd.Timedelta(days=430)).strftime("%Y-%m-%d")
        prices_panel, st1 = fetch_prices(panel_tickers, history_start=runway_start, force=force)
        cutoff_map: dict[str, pd.Timestamp | None] = {t: panel_end for t in panel_tickers}

        prices_all, st2 = prices_panel, {"offline": False}
        if custom_tickers:
            prices_custom, st2 = fetch_prices(
                panel_tickers + custom_tickers, history_start=CUSTOM_HISTORY_START, force=force)
            # Panel firms only need the runway window — trimming keeps the
            # O(months x tickers) feature loop off their 2017+ history.
            prices_all = pd.concat([
                prices_custom[prices_custom["ticker"].isin(custom_tickers)],
                prices_custom[prices_custom["ticker"].isin(panel_tickers)
                              & (prices_custom["date"] >= pd.Timestamp(runway_start))],
            ], ignore_index=True)
            cutoff_map.update({t: None for t in custom_tickers})

        live = _build_live_rows(
            panel_tickers + custom_tickers, prices_all, cutoff_map, macro, ref, coefs)
        status["offline"] = bool(st1.get("offline") or st2.get("offline"))
        status["fresh_through"] = st1.get("fresh_through") or st2.get("fresh_through")

        live.to_csv(cache_csv, index=False)
        with open(cache_meta, "w") as f:
            json.dump({"built_on": today, "tickers": panel_tickers + custom_tickers,
                       **{k: status[k] for k in ("offline", "fresh_through")}}, f)

    hist = panel[["ticker", "date"] + FEATURE_COLS + ["pd_score", "label_a"]].copy()
    hist["is_live"] = False
    hist["fundamentals_imputed"] = False
    if len(live):
        live["label_a"] = np.nan
        live = live[hist.columns]
    scores = (
        pd.concat([hist, live], ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return scores, status


def _expected_ticker_list() -> list[str]:
    """The ticker list build_live_scores() stamps into its cache meta."""
    panel_tickers = sorted(pd.read_csv(PANEL_PATH, usecols=["ticker"])["ticker"].unique())
    custom_tickers = [t for t in sorted(load_custom_watchlist()) if t not in panel_tickers]
    return panel_tickers + custom_tickers


def _append_live_cache(ticker: str, prices: pd.DataFrame) -> bool:
    """Incrementally add ONE ticker's live rows to today's disk caches.

    Keeps the on-stage add-a-firm moment at seconds instead of a full 78-firm
    rebuild. Returns False when today's caches don't exist yet — the caller
    then falls back to a full build_live_scores().
    """
    cache_csv = os.path.join(LIVE_DIR, "live_scores.csv")
    cache_meta = os.path.join(LIVE_DIR, "live_meta.json")
    today = dt.date.today().isoformat()
    if not (os.path.exists(cache_csv) and os.path.exists(cache_meta)):
        return False
    with open(cache_meta) as f:
        meta = json.load(f)
    if meta.get("built_on") != today:
        return False

    panel = load_scored_panel()
    rows = _build_live_rows(
        [ticker], prices, {ticker: None},
        macro_series(panel), build_reference(panel), load_coefficients(),
    )
    if rows.empty:
        raise ValueError(
            f"{ticker}: could not build model features from its price history.")

    live = pd.read_csv(cache_csv, parse_dates=["date"])
    if len(live):
        rows = rows.reindex(columns=live.columns)
        live = pd.concat([live[live["ticker"] != ticker], rows], ignore_index=True)
    else:
        live = rows
    live.to_csv(cache_csv, index=False)
    meta["tickers"] = _expected_ticker_list()
    with open(cache_meta, "w") as f:
        json.dump(meta, f)

    # Keep the prices cache consistent so price_history() and same-day cache
    # checks see the new firm too.
    p_csv = os.path.join(LIVE_DIR, "prices_cache.csv")
    p_meta = os.path.join(LIVE_DIR, "prices_meta.json")
    if os.path.exists(p_csv) and os.path.exists(p_meta):
        cached = pd.read_csv(p_csv, parse_dates=["date"])
        merged = (
            pd.concat([cached[cached["ticker"] != ticker], prices], ignore_index=True)
            .drop_duplicates(subset=["ticker", "date"], keep="last")
            .sort_values(["ticker", "date"])
        )
        merged.to_csv(p_csv, index=False)
        with open(p_meta) as f:
            pm = json.load(f)
        pm["tickers"] = sorted(set(pm.get("tickers", [])) | {ticker})
        with open(p_meta, "w") as f:
            json.dump(pm, f)
    return True


def add_custom_firm(ticker: str) -> dict:
    """Validate + register an analyst-added ticker and score it live.

    Returns the firm's metadata dict. Raises ValueError with an
    analyst-readable message if the ticker can't be scored.
    """
    import yfinance as yf

    ticker = ticker.strip().upper()
    if not ticker or not ticker.replace("-", "").replace(".", "").isalnum():
        raise ValueError(f"'{ticker}' is not a valid ticker symbol.")
    panel_tickers = set(pd.read_csv(PANEL_PATH, usecols=["ticker"])["ticker"].unique())
    if ticker in panel_tickers or ticker in load_custom_watchlist():
        raise ValueError(f"{ticker} is already on the watchlist.")

    probe = _yf_download_long([ticker], start=CUSTOM_HISTORY_START)
    if len(probe) < 252:
        raise ValueError(
            f"{ticker}: found {len(probe)} daily prices — the model needs at "
            f"least a year of trading history (252 days) to build features."
        )

    name, sector = ticker, "—"
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector") or "—"
    except Exception:
        pass

    watchlist = load_custom_watchlist()
    watchlist[ticker] = {"name": name, "sector": sector,
                         "added_on": dt.date.today().isoformat()}
    _save_custom_watchlist(watchlist)
    try:
        if not _append_live_cache(ticker, probe):
            build_live_scores(force=False)
    except Exception:
        # Don't leave a half-registered firm behind a failed scoring attempt.
        del watchlist[ticker]
        _save_custom_watchlist(watchlist)
        raise
    return watchlist[ticker]


def remove_custom_firm(ticker: str) -> None:
    watchlist = load_custom_watchlist()
    if ticker not in watchlist:
        return
    del watchlist[ticker]
    _save_custom_watchlist(watchlist)

    # Incremental removal from today's caches; full rebuild only if absent.
    cache_csv = os.path.join(LIVE_DIR, "live_scores.csv")
    cache_meta = os.path.join(LIVE_DIR, "live_meta.json")
    if os.path.exists(cache_csv) and os.path.exists(cache_meta):
        live = pd.read_csv(cache_csv, parse_dates=["date"])
        live[live["ticker"] != ticker].to_csv(cache_csv, index=False)
        with open(cache_meta) as f:
            meta = json.load(f)
        meta["tickers"] = _expected_ticker_list()
        with open(cache_meta, "w") as f:
            json.dump(meta, f)
    else:
        build_live_scores(force=False)


# =============================================================================
# Firm metadata + triage log
# =============================================================================

def price_history(ticker: str) -> pd.DataFrame:
    """Daily [date, adj_close] for one firm: repo cache + live cache merged."""
    frames = []
    repo = _read_repo_price_cache(ticker)
    if repo is not None:
        frames.append(repo)
    live_csv = os.path.join(LIVE_DIR, "prices_cache.csv")
    if os.path.exists(live_csv):
        live = pd.read_csv(live_csv, parse_dates=["date"])
        frames.append(live[live["ticker"] == ticker])
    if not frames:
        return pd.DataFrame(columns=["date", "adj_close"])
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
    )
    return out[["date", "adj_close"]].dropna()


def firm_directory(panel: pd.DataFrame) -> pd.DataFrame:
    """ticker -> name / sector / archetype for panel + analyst-added firms."""
    meta = (
        panel[["ticker", "firm_name", "industry", "archetype"]]
        .drop_duplicates(subset="ticker")
        .set_index("ticker")
    )
    for t, info in load_custom_watchlist().items():
        if t not in meta.index:
            meta.loc[t] = [info.get("name", t), info.get("sector", "—"), "Added live"]
    for t, info in FIRMS.items():  # panel drops delisted firms (e.g. CHK)
        if t not in meta.index:
            meta.loc[t] = [info["name"], info["industry"], "—"]
    return meta


def log_triage(entry: dict) -> None:
    """Append one escalate/monitor/dismiss decision to the governance log."""
    _ensure_live_dir()
    df = pd.DataFrame([entry])
    header = not os.path.exists(TRIAGE_LOG_PATH)
    df.to_csv(TRIAGE_LOG_PATH, mode="a", header=header, index=False)


def load_triage_log() -> pd.DataFrame:
    if not os.path.exists(TRIAGE_LOG_PATH):
        return pd.DataFrame(
            columns=["timestamp", "as_of", "ticker", "pd_score", "decile", "action", "note"])
    return pd.read_csv(TRIAGE_LOG_PATH)
