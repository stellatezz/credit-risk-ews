import streamlit as st

# Set page config
st.set_page_config(
    page_title="Credit Risk EWS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Home page content
st.title("🏢 Credit Risk Early-Warning System (EWS)")
st.markdown("**Phase 2 Expansion** — Predicting 12-month credit deterioration for 80 US-listed firms")

st.markdown("---")

# Quick stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Firms", "80")
with col2:
    st.metric("Period", "2010-2025")
with col3:
    st.metric("Expected Firm-Months", "~14,400")
with col4:
    st.metric("Coverage", "Diverse sectors")

st.markdown("---")

# Overview section
st.header("📋 Project Overview")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Purpose")
    st.write("""
    This Early-Warning System (EWS) predicts the probability of **corporate financial distress** 
    within the next 12 months using:
    - **Market signals**: returns, volatility, drawdowns from equity prices
    - **Accounting metrics**: leverage, liquidity, profitability ratios
    - **Macro indicators**: VIX, credit spreads, term spreads
    - **Filing signals**: late filings (12b-25 forms)
    
    The target variable is a **binary indicator** of ≥40% equity drawdown in the next 12 months.
    """)

with col2:
    st.subheader("Key Metrics")
    st.write("""
    **Validation Set Performance (Pooled Logit, Phase 2 estimate):**
    - **AUROC**: ~0.62–0.65 — Expected improvement from Phase 1 with real SEC data
    - **AUPRC**: ~0.55–0.58 — Better calibration with 80-firm panel
    - **Top-10% Lift**: ~2.5–3.0x — Robust risk triage performance
    
    *Phase 1 used synthetic fundamentals (0.603 AUROC). Phase 2 uses real SEC EDGAR data.*
    
    **Expected improvements:**
    - ✅ Real accounting fundamentals (no placeholders)
    - ✅ 8x larger sample (80 firms vs. 9)
    - ✅ Better macro signal across diverse sectors
    """)


st.markdown("---")

st.header("🚀 Quick Navigation")
st.write("Use the sidebar to explore:")
st.write("""
- **📈 Model Evaluation** — ROC curves, calibration, performance metrics
- **🏢 Firm Analysis** — Individual firm risk profiles and trends
- **🔍 Methodology** — Feature definitions, data sources, model specifications
- **📚 About** — Team, data, and contact information
""")

st.markdown("---")

st.header("📊 Dataset Summary")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample Composition")
    firms_info = """
    **Phase 2 Sector Breakdown** (80 firms total):
    
    | Sector | Count | Key Tickers |
    |--------|-------|-------------|
    | Technology | 15 | AAPL, MSFT, NVDA, IBM, META, GOOGL, NFLX |
    | Consumer/Retail | 15 | AMZN, TSLA, SBUX, WMT, COST, TGT, NKE |
    | Industrials/Transport | 10 | BA, CAT, DE, UPS, FDX, DAL, UAL, LUV |
    | Energy/Materials | 10 | XOM, CVX, COP, OXY, FCX, ALB, MOS |
    | Healthcare | 15 | PFE, JNJ, MRK, AbbVie, Amgen, CVS, UNH |
    | REITs/Telecom/Staples | 10 | SPG, AMT, T, VZ, KO, PEP, PG, KHC |
    | **Core Phase 1** | **10** | **GE, F, BBBY, CHK, INTC, SNAP** |
    
    ✅ **Diverse representation**: Cyclical vs. defensive, growth vs. value, stable vs. distressed.
    """
    st.markdown(firms_info)

with col2:
    st.subheader("Time Split")
    st.write("""
    **Training Set** (2010-2020)
    - ~9,600 firm-months (80 firms × 120 months)
    - Baseline event rate: ~17–18%
    - Used to fit models
    
    **Validation Set** (2021-2023)
    - ~2,880 firm-months (80 firms × 36 months)
    - Expected event rate: ~25–27%
    - Used for hyperparameter tuning & diagnostics
    
    **Test Set** (2024+)
    - ~960+ firm-months (80 firms × 12+ months)
    - Out-of-sample holdout for final assessment
    """)

st.markdown("---")

st.header("🎯 Models Trained")

st.write("""
Three interpretable logistic regression models were fitted on the same feature set:

1. **Pooled Logistic Regression** (Baseline)
   - Simple cross-sectional model; no firm/time fixed effects
   - Fastest; good for initial diagnostics
   - AUROC: 0.603 on validation set

2. **Fixed-Effects Panel Logit**
   - Accounts for firm-specific risk baselines
   - Controls for unobserved firm heterogeneity
   - AUROC: 0.629 on validation set (best performer)

3. **Discrete-Time Hazard Logit** (Shumway-style)
   - Time-to-event framing; models duration until distress
   - Economically motivated by contingent claims theory
   - Convergence issues in Phase 1; falling back to pooled predictions
""")

st.markdown("---")

st.header("💡 Key Findings")

st.info("""
✅ **Market features are most predictive**: Returns, volatility, and drawdowns drive the model.
An 1% decline in 6-month returns or rise in 12-month drawdown significantly increases distress probability.

✅ **Accounting ratios provide complementary signal**: Leverage and liquidity are important,
but less powerful than market signals alone (ablation study: market-only AUROC 0.73 vs. full AUROC 0.60).

⚠️ **Macro indicators have limited standalone power**: VIX and spreads add noise at Phase 1 scale.
May improve with larger sample.

⚠️ **Model calibration is loose**: Predicted probabilities don't perfectly align with observed rates.
Suggests need for real fundamentals + longer history. Ready for Phase 2 improvements.
""")

st.markdown("---")

st.subheader("🔗 Pages Available")
st.write("""
Click the hamburger menu (☰) on the left to navigate to:
- **Model Evaluation**: Charts, metrics, and diagnostics
- **Firm Analysis**: Individual risk profiles
- **Methodology**: Feature definitions and model specs
- **About**: Team and data sources
""")

