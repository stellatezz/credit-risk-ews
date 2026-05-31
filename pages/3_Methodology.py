import streamlit as st

st.set_page_config(page_title="Methodology", layout="wide")

st.title("🔍 Methodology & Feature Definitions")
st.markdown("Technical details on data sources, feature engineering, and model specifications")

st.markdown("---")

st.header("1️⃣ Data Pipeline")

st.write("""
```
1. Data Ingestion
   ├─ Equity prices: yfinance (daily, 2009-2025)
   ├─ Fundamentals: SEC EDGAR XBRL (real SEC data integrated in Phase 2)
   ├─ Macros: FRED API (real macro series integrated in Phase 2)
   └─ Labels: Forward drawdown computation from prices

2. Feature Engineering
   ├─ Market features: rolling returns/vol/drawdowns
   ├─ Accounting ratios: leverage, liquidity, working capital, profitability
   └─ Filing signals: late filing indicators (12b-25 forms)

3. Panel Assembly
   ├─ Merge all data by (ticker, month-end date)
   └─ Forward-fill accounting data (latest available at each month-end)

4. Time-Based Train/Val/Test Split
   ├─ Train: 2010-2020 (1,058 firm-months)
   ├─ Val: 2021-2023 (324 firm-months)
   └─ Test: 2024+ (108 firm-months)

5. Model Fitting
   ├─ Pooled logistic regression
   ├─ Fixed-effects panel logit (firm intercepts)
   └─ Discrete-time hazard logit (time-to-event)

6. Evaluation & Visualization
   └─ AUROC, AUPRC, calibration, top-K lift, lead-time analysis
```
""")

st.markdown("---")

st.header("2️⃣ Feature Definitions")

st.subheader("Market Features (6 total)")

market_features = """
| Feature | Calculation | Window | Frequency | Interpretation |
|---------|-------------|--------|-----------|-----------------|
| **ret_1m** | (Price_t / Price_t-21) - 1 | 1 month | Monthly | Very recent momentum; often negative before distress |
| **ret_3m** | (Price_t / Price_t-63) - 1 | 3 months | Monthly | Medium-term trend; captures developing slowdown |
| **ret_6m** | (Price_t / Price_t-126) - 1 | 6 months | Monthly | Longer-term trajectory; statistically predictive |
| **vol_3m** | Std(daily returns) × √252 | 3 months | Monthly | Annualized volatility; increases before distress episodes |
| **vol_6m** | Std(daily returns) × √252 | 6 months | Monthly | Sustained volatility; reflects uncertainty |
| **drawdown_12m** | (max(price) - price) / max(price) | 12 months | Monthly | Peak-to-trough; direct measure of investor losses |
"""
st.markdown(market_features)

st.write("""
**Key insight:** Market features are the most predictive group (ablation AUROC 0.733).
Why? Because equity prices incorporate forward-looking information (market efficiency).
A stock down 30% signals deterioration better than backward-looking accounting.
""")

st.markdown("---")

st.subheader("Accounting Features (4 total)")

accounting_features = """
| Feature | Calculation | Data Source | Update Frequency | Interpretation |
|---------|-------------|-------------|------------------|-----------------|
| **leverage** | Total Liabilities / Total Assets | SEC XBRL | Quarterly/Annual | Higher = more financial risk; declining buffer |
| **liquidity_buffer** | Cash & Equivalents / Total Assets | SEC XBRL | Quarterly/Annual | Lower = less ability to absorb shocks |
| **wc_ratio** | (Current Assets - Current Liabilities) / TA | SEC XBRL | Quarterly/Annual | Negative = liquidity stress; working capital strain |
| **profitability** | EBIT / Total Assets (or TTM estimate) | SEC XBRL | Quarterly/Annual | Negative/declining = fundamentals weakening |
"""
st.markdown(accounting_features)

st.write("""
**Data integrity notes:**
- All accounting metrics are **carried forward** by filing date at month-end
- No lookahead: we use only data filed/available by month t
- Phase 1 used **synthetic data** calibrated to industry baselines (GE ~65% leverage, BBBY distress drift)
- Phase 2 integrates real SEC XBRL extraction (Allen's component) for production accounting metrics
""")

st.markdown("---")

st.subheader("Macro Features (3 total)")

macro_features = """
| Feature | Source | Update Frequency | Time Series | Interpretation |
|---------|--------|-------------------|------------|-----------------|
| **vix** | CBOE (FRED VIXCLS) | Daily | 1989-present | Market fear gauge; spikes during crises |
| **term_spread** | FRED (10Y - 2Y yields) | Daily | 1990-present | Recession indicator; inversion predicts downturns |
| **credit_spread** | FRED (BAA corporate - risk-free) | Daily/Monthly | 2003-present | Credit stress; widens during financial distress |
"""
st.markdown(macro_features)

st.write("""
**Limitation in Phase 1:**
Macro features performed poorly (AUROC 0.399) because:
- Sample was only 10 firms → idiosyncratic firm risk dominated macro regime effects
- Period was mostly post-2009 recovery with few severe macro shocks

**Phase 2 findings (≈80 firms):**
Macro indicators became more informative as cross-sector correlations emerged, although market features remained the dominant predictive group.
""")

st.markdown("---")

st.subheader("Filing Signals (1 total)")

st.write("""
| Feature | Definition | Data Source | Frequency | Interpretation |
|---------|-----------|------------|-----------|-----------------|
| **late_filing** | Indicator = 1 if Form 12b-25 filed in last 6 months | SEC EDGAR | Event-based | Firms unable to file on time are red flags |

**Phase 1 Status:** Placeholder (synthetic; ~5% for BBBY/CHK).
**Phase 2:** Darren + Ivan will map 8-K Item 1.03 bankruptcy filings.
""")

st.markdown("---")

st.header("3️⃣ Label Definition: Forward Drawdown")

st.write("""
**Label A (Primary):**
```
Y_t = 1 if max_drawdown(returns from t+1 to t+12 months) ≤ -40%
Y_t = 0 otherwise
```

**Interpretation:**
- Binary: did the firm's equity fall ≥40% peak-to-trough at any point in the next 12 months?
- 40% threshold chosen because:
  - Economically meaningful (clear investor pain)
  - Empirically motivated (historical distress studies)
  - Balances label frequency (20% event rate = feasible modeling)

**Why 12-month horizon?**
- Credit analysts typically have 6-12 month decision windows
- Longer horizons (24m) are too noisy; shorter (3m) are too late for intervention

**Validation:**
- BBBY labels spike 2018-2022 (retail crisis) → matches known distress
- Most events cluster around financial crisis periods (2008-09, 2020) and firm-specific shocks
- Label is **forward-looking**: no leakage (features at t, outcomes at t+1 to t+12)
""")

st.markdown("---")

st.header("4️⃣ Model Specifications")

st.subheader("Model 1: Pooled Logistic Regression (Baseline)")

st.write("""
**Equation:**
```
P(Y_t = 1 | X_t) = 1 / (1 + exp(-(β₀ + β'X_t)))
```

**Features:** All 14 (4 accounting, 6 market, 3 macro, 1 filing)

**Advantages:**
- Simple, interpretable coefficients
- Fast to train and deploy
- Standard errors + t-tests for variable selection

**Disadvantages:**
- Ignores firm-level heterogeneity (some firms structurally riskier)
- No time fixed effects (macro regime shifts)

**Validation Performance:**
- AUROC: 0.603
- AUPRC: 0.527
- Brier Score: 0.257
""")

st.markdown("---")

st.subheader("Model 2: Fixed-Effects Panel Logit (Best Performer)")

st.write("""
**Equation:**
```
P(Y_it = 1 | X_it, α_i) = 1 / (1 + exp(-(α_i + β'X_it)))

where α_i = firm intercept (captured via conditional likelihood)
      β = common slope parameters across all firms
```

**Motivation:**
- GE has structurally higher leverage baseline than PFE (different industries)
- Model should learn firm-specific risk "gravity" (α_i) separate from feature slopes (β)
- Conditional logit eliminates incidental parameters problem (Chamberlain, 1980)

**Advantages:**
- Accounts for firm heterogeneity
- More flexible risk capture
- Better calibration than pooled

**Disadvantages:**
- Eliminates time-invariant features (industry, firm size proxies)
- Convergence issues with tied outcomes (all 0s or all 1s for a firm)

**Validation Performance:**
- AUROC: 0.629 ← **Best**
- AUPRC: 0.557
- Brier Score: 0.198
""")

st.markdown("---")

st.subheader("Model 3: Discrete-Time Hazard Logit (Shumway-style)")

st.write("""
**Motivation:**
- Time-to-distress is inherently dynamic (duration until event)
- Hazard models condition on not having failed yet (captures "survival time")
- Theoretically grounded in contingent claims (Merton model)

**Equation:**
```
h_it = P(Y_it = 1 | Y_i,t-1 = 0, X_it) = 1 / (1 + exp(-(α + β'X_it + γ*t)))

Fitted on "risk set": only months where firm has not failed yet
(firms that already defaulted drop out of sample)
```

**Challenges observed initially:**
- Early (Phase 1) runs had few events and numerical instability for flexible hazard specifications
- Singular matrix issues and near-collinearity in small risk-sets caused occasional optimization failures
- **Fallback decision:** Use pooled logit predictions when hazard fails to converge

**Phase 2 outcome:**
With the enlarged Phase 2 sample and more events, the discrete-time hazard specification stabilized in most runs and provided a useful complementary perspective when it converged.
""")

st.markdown("---")

st.header("5️⃣ No-Lookahead & Data Leakage Prevention")

st.write("""
**Critical principle:** Features at month t must use only data available at or before month t.

**Implementation:**

✅ **Market data:** Prices are observed at month-end → use for returns/vol/drawdowns
✅ **Accounting:** Balanced sheet at fiscal year-end Q4 (filed ~60 days later) → carry forward to next month-end
✅ **Macros:** VIX, yields, spreads are published daily → month-end snapshots
✅ **Label (forward):** Computed 12 months ahead of month t → no leakage into training

❌ **Prevented:** 
- Using Q1 earnings to predict Q1 label (requires ~2 month filing window)
- Using stock price at month t+6 to predict events at month t+1 (lookahead bias)

**Validation:**
Time-based split (train ≤2020, val 2021-23, test 2024+) ensures no temporal leakage.
""")

st.info("""
💡 **Phase 2 real-data integration (completed):**
- SEC XBRL pipeline includes filing dates (not just period-end dates) used for carry-forward logic
- 8-K bankruptcy events are time-stamped and mapped to labels
- Macro data from FRED is ingested with revision-awareness
""")

st.markdown("---")

st.header("6️⃣ Evaluation Metrics")

st.write("""
**AUROC (Area Under ROC Curve):**
- Range: 0.5 (random) to 1.0 (perfect)
- Interpretation: Probability that ranking of distressed firm ahead of non-distressed firm is correct
- Our 0.60: Modest; ~60% chance a randomly selected distressed firm will have higher predicted prob than random non-distressed firm

**AUPRC (Area Under Precision-Recall Curve):**
- More relevant for imbalanced settings (only 20% events)
- AUROC can be misleading when base rate is low
- Our 0.53: Slightly above random (0.20 = base rate); indicates useful signal

**Brier Score:**
- Mean squared prediction error
- Range: 0 (perfect) to 0.25 (for 20% base rate, if always predict 20%)
- Our 0.26: Slightly worse than baseline, indicating loose calibration
- FE Logit 0.20: Better calibrated

**Top-10% Capture & Lift:**
- Practical metric: in top decile, what fraction of events are we catching?
- Our 3.06x lift: Among top 10% flagged, we catch 30.6% of all events
- **Use case:** Focus credit team on 100 firms → catch ~30% of crises → ROI on analyst hours

**Lead Time:**
- Average months between prediction and actual event
- Measure of **early warning** capability
- Initial runs observed ~1 month lead time; Phase 2 improvements aim to increase actionable lead time (target: 3-6 months)
""")

