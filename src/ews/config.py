"""
Central configuration for the EWS pipeline.

Single source of truth for: firms, features, time splits, thresholds, and paths.
Anyone adding a firm edits FIRMS here; anyone changing the train/val cutoff
edits TRAIN_END_YEAR here. No magic numbers elsewhere.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Repo root: src/ews/config.py -> ../../.. -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(REPO_ROOT, "data")


class PATHS:
    RAW = os.path.join(DATA_DIR, "raw")
    INTERIM = os.path.join(DATA_DIR, "interim")
    PROCESSED = os.path.join(DATA_DIR, "processed")
    FIGURES = os.path.join(REPO_ROOT, "outputs", "figures")


# -----------------------------------------------------------------------------
# Firm universe — Phase 2 (80 firms)
# -----------------------------------------------------------------------------
# Expanded from Phase 1 (10 firms) to cover 80 US-listed non-financial firms
# across Technology, Consumer, Industrials, Energy, Healthcare, REITs, and Telecom.
FIRMS = {
    # =========================================================================
    # Phase 1 Core (10 firms)
    # =========================================================================
    "GE":   {"name": "General Electric",     "industry": "Industrial"},
    "F":    {"name": "Ford Motor",            "industry": "Auto"},
    "BBBY": {"name": "Bed Bath & Beyond",     "industry": "Retail"},
    "XOM":  {"name": "Exxon Mobil",           "industry": "Energy"},
    "CHK":  {"name": "Chesapeake Energy",     "industry": "Energy"},
    "INTC": {"name": "Intel",                 "industry": "Technology"},
    "SNAP": {"name": "Snap Inc",              "industry": "Technology"},
    "PFE":  {"name": "Pfizer",                "industry": "Healthcare"},
    "SPG":  {"name": "Simon Property Group",  "industry": "Real Estate"},
    "AAL":  {"name": "American Airlines",     "industry": "Airlines"},

    # =========================================================================
    # Phase 2 Expansion A: Technology and Communication Services (15 firms)
    # =========================================================================
    "AAPL": {"name": "Apple",                 "industry": "Technology"},
    "MSFT": {"name": "Microsoft",             "industry": "Technology"},
    "NVDA": {"name": "Nvidia",                "industry": "Semiconductors"},
    "AMD":  {"name": "AMD",                   "industry": "Semiconductors"},
    "MU":   {"name": "Micron",                "industry": "Semiconductors"},
    "CSCO": {"name": "Cisco",                 "industry": "Technology"},
    "ORCL": {"name": "Oracle",                "industry": "Technology"},
    "CRM":  {"name": "Salesforce",            "industry": "Software"},
    "IBM":  {"name": "IBM",                   "industry": "Technology"},
    "META": {"name": "Meta",                  "industry": "Communication"},
    "GOOGL":{"name": "Alphabet",              "industry": "Communication"},
    "NFLX": {"name": "Netflix",               "industry": "Media"},
    "PYPL": {"name": "PayPal",                "industry": "Fintech"},
    "SQ":   {"name": "Block",                 "industry": "Fintech"},
    "UBER": {"name": "Uber",                  "industry": "Platform"},

    # =========================================================================
    # Phase 2 Expansion B: Consumer Discretionary, Retail, and Autos (15 firms)
    # =========================================================================
    "AMZN": {"name": "Amazon",                "industry": "Retail"},
    "TSLA": {"name": "Tesla",                 "industry": "Auto"},
    "GM":   {"name": "General Motors",        "industry": "Auto"},
    "HD":   {"name": "Home Depot",            "industry": "Retail"},
    "LOW":  {"name": "Lowe's",                "industry": "Retail"},
    "NKE":  {"name": "Nike",                  "industry": "Consumer"},
    "SBUX": {"name": "Starbucks",             "industry": "Consumer"},
    "MCD":  {"name": "McDonald's",            "industry": "Consumer"},
    "TGT":  {"name": "Target",                "industry": "Retail"},
    "WMT":  {"name": "Walmart",               "industry": "Retail"},
    "COST": {"name": "Costco",                "industry": "Retail"},
    "M":    {"name": "Macy's",                "industry": "Retail"},
    "KSS":  {"name": "Kohl's",                "industry": "Retail"},
    "GPS":  {"name": "Gap",                   "industry": "Retail"},
    "ROST": {"name": "Ross Stores",           "industry": "Retail"},

    # =========================================================================
    # Phase 2 Expansion C: Industrials, Transport, and Aerospace (10 firms)
    # =========================================================================
    "BA":   {"name": "Boeing",                "industry": "Aerospace"},
    "CAT":  {"name": "Caterpillar",           "industry": "Industrials"},
    "DE":   {"name": "Deere",                 "industry": "Industrials"},
    "MMM":  {"name": "3M",                    "industry": "Industrials"},
    "HON":  {"name": "Honeywell",             "industry": "Industrials"},
    "UPS":  {"name": "UPS",                   "industry": "Logistics"},
    "FDX":  {"name": "FedEx",                 "industry": "Logistics"},
    "DAL":  {"name": "Delta",                 "industry": "Airlines"},
    "UAL":  {"name": "United Airlines",       "industry": "Airlines"},
    "LUV":  {"name": "Southwest",             "industry": "Airlines"},

    # =========================================================================
    # Phase 2 Expansion D: Energy and Materials (10 firms)
    # =========================================================================
    "CVX":  {"name": "Chevron",               "industry": "Energy"},
    "COP":  {"name": "ConocoPhillips",        "industry": "Energy"},
    "OXY":  {"name": "Occidental Petroleum",  "industry": "Energy"},
    "SLB":  {"name": "Schlumberger",          "industry": "Energy"},
    "HAL":  {"name": "Halliburton",           "industry": "Energy"},
    "FCX":  {"name": "Freeport-McMoRan",      "industry": "Materials"},
    "NUE":  {"name": "Nucor",                 "industry": "Materials"},
    "DOW":  {"name": "Dow",                   "industry": "Chemicals"},
    "ALB":  {"name": "Albemarle",             "industry": "Materials"},
    "MOS":  {"name": "Mosaic",                "industry": "Materials"},

    # =========================================================================
    # Phase 2 Expansion E: Healthcare and Defensive Firms (15 firms)
    # =========================================================================
    "JNJ":  {"name": "Johnson & Johnson",     "industry": "Healthcare"},
    "MRK":  {"name": "Merck",                 "industry": "Healthcare"},
    "ABBV": {"name": "AbbVie",                "industry": "Healthcare"},
    "BMY":  {"name": "Bristol Myers Squibb",  "industry": "Healthcare"},
    "GILD": {"name": "Gilead",                "industry": "Healthcare"},
    "AMGN": {"name": "Amgen",                 "industry": "Healthcare"},
    "MDT":  {"name": "Medtronic",             "industry": "Healthcare"},
    "CVS":  {"name": "CVS Health",            "industry": "Healthcare"},
    "UNH":  {"name": "UnitedHealth",          "industry": "Healthcare"},
    "TMO":  {"name": "Thermo Fisher",         "industry": "Healthcare"},

    # =========================================================================
    # Phase 2 Expansion F: REITs, Telecom, and Staples (5 firms)
    # =========================================================================
    "AMT":  {"name": "American Tower",        "industry": "Real Estate"},
    "PLD":  {"name": "Prologis",              "industry": "Real Estate"},
    "O":    {"name": "Realty Income",         "industry": "Real Estate"},
    "VTR":  {"name": "Ventas",                "industry": "Real Estate"},
    "T":    {"name": "AT&T",                  "industry": "Telecom"},
    "VZ":   {"name": "Verizon",               "industry": "Telecom"},
    "KO":   {"name": "Coca-Cola",             "industry": "Staples"},
    "PEP":  {"name": "PepsiCo",               "industry": "Staples"},
    "PG":   {"name": "Procter & Gamble",      "industry": "Staples"},
    "KHC":  {"name": "Kraft Heinz",           "industry": "Staples"},
}

TICKERS = list(FIRMS.keys())

# Tickers allowed to have < MIN_HISTORY_DAYS of data without raising LoaderError.
# Empty by default — add a ticker here only after deciding its short history
# is acceptable for this panel.
#
# Phase 1: CHK (Chesapeake Energy): delisted post-bankruptcy (June 2020); yfinance
# returns no data. Intentional inclusion for historical context.
#
# Phase 2: Some tickers may be newer IPOs or have incomplete historical pricing.
# Review yfinance errors during first run and add tickers here as needed.
ALLOWED_SHORT_HISTORY: set[str] = {"CHK","SQ","GPS"}  # Expand if needed during Phase 2 run

# -----------------------------------------------------------------------------
# Price-download window (daily prices; used by load_prices)
# -----------------------------------------------------------------------------
PRICE_START = "2009-06-01"   # 6mo buffer before 2010 for rolling features
PRICE_END = "2025-12-31"
MIN_HISTORY_DAYS = 100       # min per-ticker history; below this -> LoaderError

# -----------------------------------------------------------------------------
# Feature set
# -----------------------------------------------------------------------------
# Column order matters: models.py adds these as a constant list; changing the
# order here would change the coefficient output order and break stdout diff.
FEATURE_COLS = [
    "leverage", "liquidity_buffer", "wc_ratio", "profitability",
    "ret_1m", "ret_3m", "ret_6m",
    "vol_3m", "vol_6m", "drawdown_12m",
    "late_filing",
    "vix", "term_spread", "credit_spread",
]
LABEL_COL = "label_a"

# -----------------------------------------------------------------------------
# Time split
# -----------------------------------------------------------------------------
TRAIN_END_YEAR = 2020    # train <= 2020
VAL_END_YEAR = 2023      # val 2021-2023; test 2024+
PANEL_START_YEAR = 2010  # drop anything before this in panel assembly

# -----------------------------------------------------------------------------
# Label A definition (forward drawdown)
# -----------------------------------------------------------------------------
LABEL_A_THRESHOLD = -0.40         # 40% peak-to-trough drawdown
LABEL_A_HORIZON_MONTHS = 12       # over the next 12 months

# -----------------------------------------------------------------------------
# Evaluation thresholds
# -----------------------------------------------------------------------------
TOP_K_FRACTION = 0.10             # decile chart / top-K lift
LEAD_TIME_THRESHOLD = 0.3         # prob above this = "flagged"
