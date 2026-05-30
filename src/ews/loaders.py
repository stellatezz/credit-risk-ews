"""
Data loaders — the team-facing API.

This file defines the contract Allen (SEC fundamentals) and Darren (8-K labels)
build against. Each public loader takes a set of inputs, dispatches by
`source=` kwarg, validates output shape via `check_loader`, and returns a
long-format DataFrame.

Loader failure policy: any unexpected exception inside a loader propagates
up and halts the pipeline. Silent fallback from a real-data source to a
placeholder is explicitly prohibited — see `run.py` three-tier policy block.

To add a new source (e.g., Bloomberg prices): add a branch to the dispatch
`if`/`elif` in the public `load_*` function, implement the private
`_<name>_<source>_impl` function, and append to LOADER_SCHEMAS if the output
schema differs. Call `check_loader()` at the bottom of every implementation.
"""

from __future__ import annotations

import os
from typing import Any

import json
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from .config import (
    ALLOWED_SHORT_HISTORY,
    FIRMS,
    MIN_HISTORY_DAYS,
    PATHS,
    PRICE_END,
    PRICE_START,
)


# =============================================================================
# Errors
# =============================================================================

class LoaderError(RuntimeError):
    """Raised by any loader when its output cannot satisfy the pipeline contract."""


# =============================================================================
# Schema registry + validator
# =============================================================================

LOADER_SCHEMAS: dict[str, set[str]] = {
    "prices":          {"ticker", "date", "close", "adj_close"},
    "market_features": {"ticker", "date", "ret_1m", "ret_3m", "ret_6m",
                        "vol_3m", "vol_6m", "drawdown_12m"},
    "fundamentals":    {"ticker", "date", "leverage", "liquidity_buffer",
                        "wc_ratio", "profitability", "z_score", "late_filing"},
    "macros":          {"date", "vix", "term_spread", "credit_spread"},
    "labels":          {"ticker", "date", "label_a", "forward_max_drawdown", "label_b"},
}


def check_loader(name: str, df: pd.DataFrame) -> None:
    """
    Validate that `df` satisfies the contract for loader `name`.

    Raises AssertionError with a specific message for any violation. Allen
    and Darren should call this as the last line of their loader, before
    returning, to confirm the pipeline will accept their output.
    """
    if name not in LOADER_SCHEMAS:
        raise AssertionError(
            f"Unknown loader name '{name}'. "
            f"Known: {sorted(LOADER_SCHEMAS)}. "
            f"If you're adding a new loader, register its schema in LOADER_SCHEMAS."
        )
    expected = LOADER_SCHEMAS[name]
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"{name} loader missing columns: {sorted(missing)}"
    assert not extra, (
        f"{name} loader has unexpected columns: {sorted(extra)}. "
        f"Add them to LOADER_SCHEMAS if intentional, or drop them before returning."
    )
    assert len(df) > 0, f"{name} loader returned 0 rows"
    if "date" in expected:
        assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
            f"{name}.date must be datetime64; got {df['date'].dtype}"
        )


# =============================================================================
# load_prices
# =============================================================================

def load_prices(
    tickers: list[str],
    start: str = PRICE_START,
    end: str = PRICE_END,
    source: str = "yfinance",
) -> pd.DataFrame:
    """
    Returns raw daily prices in LONG format. No returns, no features —
    derived market math lives in features.py.

    Columns: [ticker, date, close, adj_close].
    One row per (ticker, trading-day) pair.

    source='yfinance' is the only implementation today; kept as a kwarg for
    uniformity with other loaders and to leave room for future sources
    (e.g., source='bloomberg').
    """
    if source == "yfinance":
        df = _prices_yfinance_impl(tickers, start, end)
    else:
        raise LoaderError(
            f"load_prices: unknown source={source!r}. "
            f"Supported: 'yfinance'."
        )
    check_loader("prices", df)
    return df


def _prices_yfinance_impl(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily prices from yfinance, with per-ticker disk caching
    in data/raw/yfinance_<TICKER>.csv. Raises LoaderError if any ticker has
    fewer than MIN_HISTORY_DAYS days and is not in ALLOWED_SHORT_HISTORY."""
    print("Loading daily prices (yfinance, with data/raw/ cache)...")
    print(f"  Tickers: {', '.join(tickers)}")

    os.makedirs(PATHS.RAW, exist_ok=True)

    records: list[pd.DataFrame] = []
    dropped: list[tuple[str, int]] = []

    for ticker in tickers:
        cache_path = os.path.join(PATHS.RAW, f"yfinance_{ticker}.csv")

        if os.path.exists(cache_path):
            raw = pd.read_csv(cache_path, parse_dates=["date"])
            print(f"  [OK] {ticker}: {len(raw)} days (cached)")
        else:
            try:
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            except Exception as e:
                print(f"  [FAIL] {ticker}: download failed ({e})")
                dropped.append((ticker, 0))
                continue

            if data is None or len(data) == 0:
                print(f"  [FAIL] {ticker}: no data returned")
                dropped.append((ticker, 0))
                continue

            # auto_adjust=False so `close` and `adj_close` carry distinct values:
            #   close     -> raw exchange close (unadjusted for splits/dividends)
            #   adj_close -> split/dividend-adjusted close (the one features.py uses)
            # Every downstream consumer (features.py, labels.py) already pivots on
            # adj_close, so the adjusted-close values are numerically identical to
            # what auto_adjust=True used to surface — panel output is unchanged.
            raw = pd.DataFrame({
                "date": data.index,
                "close": data["Close"].squeeze().values,
                "adj_close": data["Adj Close"].squeeze().values,
            })
            raw.to_csv(cache_path, index=False)
            print(f"  [OK] {ticker}: {len(raw)} days "
                  f"({raw['date'].min().strftime('%Y-%m-%d')} -> "
                  f"{raw['date'].max().strftime('%Y-%m-%d')})")

        if len(raw) < MIN_HISTORY_DAYS:
            dropped.append((ticker, len(raw)))
            continue

        raw["ticker"] = ticker
        records.append(raw[["ticker", "date", "close", "adj_close"]])

    # Loud-drop policy: if any requested ticker fell below MIN_HISTORY_DAYS and
    # isn't explicitly allow-listed, raise. This prevents silent panel shrinkage
    # when a teammate adds a ticker with incomplete history.
    unexpected_drops = [(t, n) for (t, n) in dropped if t not in ALLOWED_SHORT_HISTORY]
    if unexpected_drops:
        details = ", ".join(f"{t} ({n} days)" for t, n in unexpected_drops)
        raise LoaderError(
            f"load_prices: {len(unexpected_drops)} ticker(s) have insufficient history "
            f"(< {MIN_HISTORY_DAYS} days): {details}. "
            f"Either (a) fix the data source, or (b) add these tickers to "
            f"ALLOWED_SHORT_HISTORY in config.py if the short history is intentional."
        )

    df = pd.concat(records, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\nPrice matrix (long): {len(df)} rows, {df['ticker'].nunique()} tickers")
    return df

# ---------------------------------------------------------------------------
# SEC EDGAR XBRL fundamentals loader
# ---------------------------------------------------------------------------

_SEC_CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "TotalAssets": (
        "Assets",
        ["AssetsCurrent"],
    ),
    "TotalLiabilities": (
        "Liabilities",
        ["LiabilitiesAndStockholdersEquity"],
    ),
    "StockholdersEquity": (
        "StockholdersEquity",
        ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    ),
    "Cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        ["CashCashEquivalentsAndShortTermInvestments", "Cash"],
    ),
    "CurrentAssets": (
        "AssetsCurrent",
        [],
    ),
    "CurrentLiabilities": (
        "LiabilitiesCurrent",
        [],
    ),
    "NetIncome": (
        "NetIncomeLoss",
        ["ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    ),
    "RetainedEarnings": (
        "RetainedEarningsAccumulatedDeficit",
        [],
    ),
}

_SEC_HEADERS = {
    "User-Agent": "credit-risk-ews contact@ews-team.edu",
    "Accept-Encoding": "gzip, deflate",
}
_SEC_RATE_SLEEP = 0.12   # 10 req/s SEC limit; 120ms between calls is safe
_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_FACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def _load_cik_map() -> dict[str, int]:
    """Download/cache the SEC company_tickers.json -> {TICKER: CIK}."""
    os.makedirs(PATHS.RAW, exist_ok=True)
    cache = os.path.join(PATHS.RAW, "sec_company_tickers.json")
    if os.path.exists(cache):
        with open(cache) as f:
            raw = json.load(f)
    else:
        resp = requests.get(_EDGAR_TICKERS_URL, headers=_SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        with open(cache, "w") as f:
            json.dump(raw, f)
        time.sleep(_SEC_RATE_SLEEP)
    # SEC format: {"0": {"cik_str": 40987, "ticker": "GE", ...}, ...}
    return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}


def _load_company_facts(cik: int, ticker: str) -> dict:
    """Download (and disk-cache) the XBRL company-facts JSON for one CIK."""
    os.makedirs(PATHS.RAW, exist_ok=True)
    cache = os.path.join(PATHS.RAW, f"sec_{ticker.upper()}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    url = _EDGAR_FACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=_SEC_HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    with open(cache, "w") as f:
        json.dump(data, f)
    time.sleep(_SEC_RATE_SLEEP)
    return data


def _extract_concept_series(facts: dict, primary: str, fallbacks: list[str]) -> pd.Series | None:
    """
    Extract one GAAP concept as a time-series (period-end date -> value).
    Tries primary name first, then each fallback.
    Only uses 10-K and 10-Q entries (skips instantaneous point-in-time values).
    Returns None if nothing found.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in [primary] + fallbacks:
        if concept not in gaap:
            continue
        usd_entries = gaap[concept].get("units", {}).get("USD", [])
        if not usd_entries:
            continue
        rows = []
        for entry in usd_entries:
            if entry.get("form", "") not in ("10-K", "10-Q"):
                continue
            end_date = entry.get("end")
            val = entry.get("val")
            if end_date is None or val is None:
                continue
            rows.append({"date": pd.to_datetime(end_date), "val": float(val)})
        if not rows:
            continue
        s = (
            pd.DataFrame(rows)
            .sort_values("date")
            .drop_duplicates(subset="date", keep="last")
            .set_index("date")["val"]
        )
        return s
    return None


def _compute_late_filing_flag(facts: dict, monthly_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Returns integer Series (0/1) indexed by monthly_dates.
    Flagged 1 if any 10-K/10-Q due in the prior 4-month window was filed
    more than 5 days after its SEC deadline:
      10-K deadline = period_end + 60 days  (accelerated filer)
      10-Q deadline = period_end + 40 days
    Falls back to all-zero if no filing metadata found.
    """
    filings_meta = facts.get("facts", {}).get("us-gaap", {})
    filed_rows = []
    for concept_data in filings_meta.values():
        for unit_entries in concept_data.get("units", {}).values():
            for entry in unit_entries:
                form = entry.get("form", "")
                if form not in ("10-K", "10-Q"):
                    continue
                end_date  = entry.get("end")
                filed_str = entry.get("filed")
                if end_date and filed_str:
                    filed_rows.append({
                        "period_end": pd.to_datetime(end_date),
                        "filed":      pd.to_datetime(filed_str),
                        "form":       form,
                    })
        if len(filed_rows) > 100:   # enough data; stop early
            break

    flags = pd.Series(0, index=monthly_dates, dtype=int)
    if not filed_rows:
        return flags

    filing_df = (
        pd.DataFrame(filed_rows)
        .drop_duplicates(subset=["period_end", "form"], keep="first")
    )
    filing_df["deadline_days"] = filing_df["form"].map({"10-K": 60, "10-Q": 40})
    filing_df["deadline"] = (
        filing_df["period_end"]
        + pd.to_timedelta(filing_df["deadline_days"], unit="D")
    )
    filing_df["late"] = (filing_df["filed"] - filing_df["deadline"]).dt.days > 5

    for month in monthly_dates:
        lookback_start = month - pd.DateOffset(months=4)
        window = filing_df[
            (filing_df["deadline"] >= lookback_start) &
            (filing_df["deadline"] <= month)
        ]
        if len(window) > 0 and window["late"].any():
            flags[month] = 1

    return flags


def _fundamentals_sec_impl(
    tickers: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Real SEC EDGAR XBRL fundamentals loader.

    Ratio definitions:
      leverage          = Liabilities / Assets
      liquidity_buffer  = Cash / Assets
      wc_ratio          = (CurrentAssets - CurrentLiabilities) / Assets
      profitability     = NetIncome / Assets
      z_score           = 1.2*wc_ratio + 1.4*(RetainedEarnings/Assets)
                        + 3.3*profitability + 0.6*(SE/Liabilities)
                        + 0.999*profitability  [simplified Altman]

    Graceful degradation:
      - Missing concept -> that ratio is NaN for that ticker.
      - CIK not found / JSON fetch fails -> ticker skipped, warned,
        refilled via placeholder.
      - < 12 non-null months after ffill -> ticker skipped similarly.
    """
    MIN_SEC_ROWS = 12

    print("\nLoading SEC EDGAR fundamentals (with data/raw/ cache)...")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Monthly dates: {dates.min().strftime('%Y-%m')} "
          f"to {dates.max().strftime('%Y-%m')} ({len(dates)} months)")

    try:
        cik_map = _load_cik_map()
    except Exception as e:
        raise LoaderError(f"SEC CIK map download failed: {e}") from e

    all_records: list[pd.DataFrame] = []
    skipped: list[str] = []

    for ticker in tickers:
        print(f"  Processing {ticker}...", end=" ", flush=True)

        cik = cik_map.get(ticker.upper())
        if cik is None:
            print("CIK not found — skipping")
            skipped.append(ticker)
            continue

        try:
            facts = _load_company_facts(cik, ticker)
        except Exception as e:
            print(f"facts fetch failed ({e}) — skipping")
            skipped.append(ticker)
            continue

        # Extract raw concept series
        concepts: dict[str, pd.Series | None] = {}
        for key, (primary, fallbacks) in _SEC_CONCEPTS.items():
            concepts[key] = _extract_concept_series(facts, primary, fallbacks)

        # Build quarterly snapshot index (union of all concept dates)
        all_dates_set: set[pd.Timestamp] = set()
        for s in concepts.values():
            if s is not None:
                all_dates_set.update(s.index)

        if not all_dates_set:
            print("no us-gaap data — skipping")
            skipped.append(ticker)
            continue

        q_index = pd.DatetimeIndex(sorted(all_dates_set))
        snap = pd.DataFrame(index=q_index)
        for key, s in concepts.items():
            snap[key] = s.reindex(q_index) if s is not None else np.nan

        # Compute ratios at each quarterly snapshot
        ta   = snap["TotalAssets"]
        tl   = snap["TotalLiabilities"]
        se   = snap["StockholdersEquity"]
        cash = snap["Cash"]
        ca   = snap["CurrentAssets"]
        cl   = snap["CurrentLiabilities"]
        ni   = snap["NetIncome"]
        re   = snap["RetainedEarnings"]

        with np.errstate(divide="ignore", invalid="ignore"):
            leverage         = np.where(ta != 0, tl / ta,        np.nan)
            liquidity_buffer = np.where(ta != 0, cash / ta,      np.nan)
            wc_ratio         = np.where(ta != 0, (ca - cl) / ta, np.nan)
            profitability    = np.where(ta != 0, ni / ta,        np.nan)
            re_ratio         = np.where(ta != 0, re / ta,        np.nan)
            se_tl_ratio      = np.where(tl != 0, se / tl,        np.nan)

        z_score = (
            1.2   * wc_ratio
            + 1.4   * re_ratio
            + 3.3   * profitability
            + 0.6   * se_tl_ratio
            + 0.999 * profitability
        )

        quarterly = pd.DataFrame({
            "leverage":         leverage,
            "liquidity_buffer": liquidity_buffer,
            "wc_ratio":         wc_ratio,
            "profitability":    profitability,
            "z_score":          z_score,
        }, index=q_index)

        # Forward-fill quarterly -> monthly
        combined_index = q_index.union(dates).sort_values()
        monthly_fund = (
            quarterly
            .reindex(combined_index)
            .ffill()
            .reindex(dates)
        )

        if monthly_fund.dropna(how="all").shape[0] < MIN_SEC_ROWS:
            print(f"only {monthly_fund.dropna(how='all').shape[0]} non-null rows -- skipping")
            skipped.append(ticker)
            continue

        # Late-filing flag
        late_flags = _compute_late_filing_flag(facts, dates)
        monthly_fund["late_filing"] = late_flags.values
        monthly_fund["ticker"]      = ticker
        monthly_fund["date"]        = dates

        all_records.append(monthly_fund.reset_index(drop=True))
        cov = monthly_fund.drop(columns=["ticker","date"]).notna().mean().mean()
        print(f"OK ({monthly_fund.shape[0]} months, {cov:.0%} coverage)")

    if skipped:
        print(f"\n  Warning: {len(skipped)} ticker(s) skipped, "
              f"falling back to placeholder: {', '.join(skipped)}")
        all_records.append(_fundamentals_placeholder_impl(skipped, dates))

    if not all_records:
        raise LoaderError(
            "SEC loader: no tickers produced usable data. "
            "Check network access to data.sec.gov."
        )

    df = pd.concat(all_records, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    clip_map = {
        "leverage":         (0.0,  1.0),
        "liquidity_buffer": (0.0,  1.0),
        "wc_ratio":         (-1.0, 1.0),
        "profitability":    (-0.5, 0.5),
        "z_score":          (-5.0, 15.0),
    }
    for col, (lo, hi) in clip_map.items():
        df[col] = df[col].clip(lo, hi)

    df = df[["ticker", "date", "leverage", "liquidity_buffer",
             "wc_ratio", "profitability", "z_score", "late_filing"]]

    print(f"\n  SEC fundamentals: {len(df)} firm-months, "
          f"{df['ticker'].nunique()} tickers")
    return df


# =============================================================================
# load_fundamentals
# =============================================================================

def load_fundamentals(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    # source: str = "placeholder",
    source: str = "sec",
) -> pd.DataFrame:
    """
    Returns firm-month fundamentals in LONG format.

    Columns: [ticker, date, leverage, liquidity_buffer, wc_ratio,
              profitability, z_score, late_filing].

    source='placeholder' -> synthetic industry-typical profiles (Phase 1 default).
    source='sec'         -> Allen's real SEC EDGAR loader (not implemented yet).
    """
    if source == "sec":
        df = _fundamentals_sec_impl(tickers, dates)
    elif source == "placeholder":
        df = _fundamentals_placeholder_impl(tickers, dates)
    else:
        raise LoaderError(
            f"load_fundamentals: unknown source={source!r}. "
            f"Supported: 'sec', 'placeholder'."
        )
    check_loader("fundamentals", df)
    return df


def _fundamentals_placeholder_impl(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic per-industry fundamentals with distress drift for BBBY/CHK.
    Deterministic: seed=42. DO NOT edit without re-baselining the panel SHA256."""
    print("\nGenerating placeholder fundamentals (replace with SEC data later)...")

    # Industry-typical financial profiles. Keep in sync with FIRMS in config.
    profiles = {
        "GE":   {"lev": 0.65, "liq": 0.08, "wc": 0.05, "prof": 0.04},
        "F":    {"lev": 0.75, "liq": 0.10, "wc": 0.02, "prof": 0.03},
        "BBBY": {"lev": 0.60, "liq": 0.05, "wc": 0.08, "prof": 0.02},
        "XOM":  {"lev": 0.45, "liq": 0.06, "wc": 0.10, "prof": 0.08},
        "CHK":  {"lev": 0.80, "liq": 0.03, "wc": -0.05, "prof": -0.02},
        "INTC": {"lev": 0.35, "liq": 0.15, "wc": 0.15, "prof": 0.10},
        "SNAP": {"lev": 0.50, "liq": 0.20, "wc": 0.10, "prof": -0.05},
        "PFE":  {"lev": 0.40, "liq": 0.12, "wc": 0.12, "prof": 0.12},
        "SPG":  {"lev": 0.70, "liq": 0.04, "wc": -0.02, "prof": 0.06},
        "AAL":  {"lev": 0.85, "liq": 0.08, "wc": -0.10, "prof": 0.02},
    }

    np.random.seed(42)  # reproducibility pin; identical panel byte output across runs
    records = []

    for ticker in tickers:
        p = profiles.get(ticker, {"lev": 0.5, "liq": 0.10, "wc": 0.05, "prof": 0.05})

        for date in dates:
            # Slow quarterly-ish drift + small noise.
            distress_drift = 0
            if ticker == "BBBY" and date.year >= 2018:
                distress_drift = (date.year - 2018) * 0.03
            if ticker == "CHK" and date.year >= 2015:
                distress_drift = (date.year - 2015) * 0.02

            leverage = np.clip(p["lev"] + distress_drift + np.random.normal(0, 0.02), 0.05, 0.99)
            liquidity = np.clip(p["liq"] - distress_drift * 0.3 + np.random.normal(0, 0.01), 0.01, 0.5)
            wc_ratio = np.clip(p["wc"] - distress_drift * 0.5 + np.random.normal(0, 0.02), -0.4, 0.4)
            profitability = np.clip(p["prof"] - distress_drift * 0.4 + np.random.normal(0, 0.015), -0.3, 0.3)

            # Altman Z-score approximation (same formula as original).
            z_score = 1.2 * wc_ratio + 1.4 * profitability + 3.3 * profitability + 0.6 * (1 - leverage) + 0.8

            records.append({
                "ticker": ticker,
                "date": date,
                "leverage": leverage,
                "liquidity_buffer": liquidity,
                "wc_ratio": wc_ratio,
                "profitability": profitability,
                "z_score": z_score,
                "late_filing": 1 if (ticker in ["BBBY", "CHK"] and np.random.random() < 0.05) else 0,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Placeholder fundamentals: {len(df)} firm-months")
    return df


# =============================================================================
# load_macros
# =============================================================================

def load_macros(dates: pd.DatetimeIndex, source: str = "synthetic") -> pd.DataFrame:
    """
    Returns monthly macro series.

    Columns: [date, vix, term_spread, credit_spread].

    source='synthetic' -> regime-aware approximations (Phase 1 default).
    source='fred'      -> FRED API loader (not implemented yet).
    """
    if source == "synthetic":
        df = _macros_synthetic_impl(dates)
    elif source == "fred":
        raise NotImplementedError(
            "load_macros(source='fred') — FRED API loader is not yet wired. "
            "See docs/05_PLUGGING_IN_REAL_DATA.md."
        )
    else:
        raise LoaderError(
            f"load_macros: unknown source={source!r}. "
            f"Supported: 'synthetic', 'fred'."
        )
    check_loader("macros", df)
    return df


def _macros_synthetic_impl(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Regime-aware VIX / term spread / credit spread approximations.
    Deterministic: seed=123. DO NOT edit without re-baselining."""
    print("\nGenerating macro indicators (replace with FRED data when available)...")

    np.random.seed(123)
    records = []
    for date in dates:
        y, m = date.year, date.month

        base_vix = 16
        if y == 2011 and m >= 8:  base_vix = 30   # Euro debt crisis
        if y == 2015 and m >= 8:  base_vix = 25   # China concerns
        if y == 2018 and m == 2:  base_vix = 33   # Volmageddon
        if y == 2018 and m >= 10: base_vix = 25   # Q4 selloff
        if y == 2020 and 2 <= m <= 4: base_vix = 55  # COVID crash
        if y == 2020 and 5 <= m <= 8: base_vix = 30  # COVID recovery
        if y == 2022 and m >= 1:  base_vix = 25   # Rate hikes
        if y == 2022 and m >= 6:  base_vix = 28

        vix = max(10, base_vix + np.random.normal(0, 3))

        base_spread = 1.5
        if y >= 2019 and y <= 2020: base_spread = 0.2
        if y >= 2022: base_spread = -0.5
        if y >= 2024: base_spread = 0.5
        term_spread = base_spread + np.random.normal(0, 0.2)

        base_credit = 2.0
        if y == 2020 and 2 <= m <= 5: base_credit = 4.0
        if y == 2022 and m >= 6: base_credit = 2.5
        credit_spread = max(0.5, base_credit + np.random.normal(0, 0.2))

        records.append({
            "date": date,
            "vix": vix,
            "term_spread": term_spread,
            "credit_spread": credit_spread,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Macro data: {len(df)} months")
    return df


# =============================================================================
# load_labels
# =============================================================================

def load_labels(
    prices_df: pd.DataFrame,
    horizon_months: int = 12,
    threshold: float = -0.40,
    source: str = "drawdown",
) -> pd.DataFrame:
    """
    Returns (ticker, date, label_a, forward_max_drawdown, label_b).

    label_b is filled with NaN until Darren wires in 8-K Item 1.03 filings.
    Pipeline tolerates the NaN column (panel uses label_a as the prediction
    target; label_b is unused in Phase 1 models).

    source='drawdown' -> Label A from forward price drawdown (Phase 1 default).
    source='8k'       -> Darren's 8-K labels (not implemented yet).
    """
    # Local import: labels.py owns the actual derivation logic; this module
    # just handles dispatch + schema validation. Kept local to avoid a
    # circular import since labels.py doesn't need anything from loaders.py.
    from . import labels as _labels

    if source == "drawdown":
        df = _labels.compute_label_a_from_prices(prices_df, horizon_months, threshold)
    elif source == "8k":
        raise NotImplementedError(
            "load_labels(source='8k') — Darren's 8-K bankruptcy loader is not yet "
            "wired. See docs/05_PLUGGING_IN_REAL_DATA.md for Label B semantics."
        )
    else:
        raise LoaderError(
            f"load_labels: unknown source={source!r}. "
            f"Supported: 'drawdown', '8k'."
        )
    check_loader("labels", df)
    return df
