import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("📚 About This Project")
st.markdown("Credit Risk Early-Warning System (EWS) — Phase 2 Results and Dashboard")

st.markdown("---")

st.header("👥 Team")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Shrey")
    st.write("""
    **Responsibility:** Market Features
    - yfinance extraction & caching
    - Rolling returns/volatility/drawdowns
    - Monthly panel aggregation
    - Feature engineering validation
    """)

    st.subheader("Allen")
    st.write("""
    **Responsibility:** SEC Fundamentals
    - SEC EDGAR XBRL extraction
    - Accounting ratio construction
    - Filing-date carry-forward logic
    - Phase 2: Real data integration (completed)
    """)

with col2:
    st.subheader("Darren")
    st.write("""
    **Responsibility:** Labels & Filing Signals
    - Label A (forward drawdown) construction
    - 12b-25 late filing flags
    - Phase 2: 8-K bankruptcy mapping (Label B) (completed)
    """)

    st.subheader("Ivan")
    st.write("""
    **Responsibility:** Econometrics & Evaluation
    - Pooled/FE/hazard logit training
    - Metrics computation & calibration
    - Ablation studies & robustness
    - Model selection & diagnostics
    """)

st.markdown("---")

st.subheader("Vincent Hui Fai Wong")
st.write("""
**Responsibility:** Integration, Scope Control, Writing
- Project proposal & narrative consistency
- Oral defense preparation
- Technical documentation & reproducibility
- Phase 1→Phase 2 roadmap
""")

st.markdown("---")

st.header("📊 Project Phases")

phases = """
| Phase | Duration | Scope | Status |
|-------|----------|-------|--------|
| **Phase 1** | Apr 15 - May 13 | 9 firms, toy pipeline, Label A, pooled logit | ✅ Complete |
| **Phase 2** | May 13 - Jun 01 | ~80 firms, full pipeline, real SEC data, ablation & evaluation | ✅ Complete |
| **Phase 3** | Jun 01 - Jun 15| Robustness testing, threshold sensitivity, final model selection | ⏳ In Progress |
| **Phase 4** | Jun - Jul | Report writing, presentation, webpage, oral exam | ⏳ Upcoming |
"""
st.markdown(phases)

st.markdown("---")

st.header("💾 Data Sources & Attribution")

st.write("""
**Market Data (Equity Prices):**
- Source: yfinance (Yahoo Finance API wrapper)
- Period: 2009-06-01 to 2025-12-31
- Frequency: Daily adjusted close prices
- License: Public data; no attribution required
- Caching: data/raw/yfinance_<TICKER>.csv (local disk)

**Accounting Data (Fundamentals) — Phase 1:**
- Source: **Placeholder (synthetic)**
- Simulated based on industry-typical baselines + distress drift
- Purpose: Proof-of-concept; will be replaced with real data in Phase 2
- Reproducibility: Seed=42; deterministic output

**Accounting Data (Fundamentals) — Phase 2:**
- Source: SEC EDGAR (companyfacts endpoint)
- CIK lookups + XBRL extraction
- Allen's component integrated into pipeline

**Macro Data — Phase 1:**
- Source: **Placeholder (synthetic)**
- Simulated VIX, term spreads, credit spreads aware of historical crisis regimes
- Purpose: Proof-of-concept; will be replaced with FRED data in Phase 2
- Reproducibility: Seed=123; deterministic output

**Macro Data — Phase 2:**
- Source: FRED API (Federal Reserve Economic Data)
- Series: VIXCLS, T10Y2Y, BAA spreads
- License: Public domain; no attribution required

**Label Data:**
- Computed in-house from yfinance price data
- Forward drawdown detection: max decline in next 12 months ≥ 40%
- No external dependency

**8-K Filings (Label B) — Phase 2:**
- Source: SEC EDGAR (Form 8-K, Item 1.03)
- Darren's mapping component integrated into pipeline
""")

st.markdown("---")

st.header("🔧 Technical Stack")

tech_stack = """
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.12 | Primary implementation |
| **Data Processing** | pandas | 3.0.2 | Panel assembly, feature engineering |
| **Numerical Computing** | numpy | 2.4.4 | Matrix operations, calculations |
| **Econometrics** | statsmodels | 0.14+ | Logistic regression, panel models |
| **Visualization** | matplotlib, seaborn | Latest | Publication-quality charts |
| **Data Download** | yfinance | Latest | Yahoo Finance API |
| **Web Dashboard** | Streamlit | 1.57+ | Interactive dashboard (this app) |
| **Version Control** | Git | 2.x | Repository management |
| **Deployment** | Streamlit Cloud | - | Free hosting (Phase 2 goal) |
"""
st.markdown(tech_stack)

st.markdown("---")

st.header("📁 Repository Structure")

repo_structure = """
```
credit-risk-ews/
├── README.md                          # Project overview
├── CONTRIBUTING.md                    # Contribution guidelines
├── spec.md                            # Detailed specification
├── TODOS.md                           # Deferred work items
├── requirements.txt                   # Python dependencies
│
├── src/
│   ├── run.py                         # Entry point: orchestrates pipeline
│   └── ews/
│       ├── __init__.py
│       ├── config.py                  # Central config (firms, paths, splits)
│       ├── loaders.py                 # Data loaders with schema validation
│       ├── features.py                # Market feature engineering
│       ├── labels.py                  # Label A construction
│       ├── panel.py                   # Panel assembly & merging
│       ├── models.py                  # Model training (pooled, FE, hazard)
│       ├── eval.py                    # Evaluation metrics & ablation
│       └── viz.py                     # Visualization generation
│
├── data/
│   ├── raw/                           # Cached price data (yfinance_*.csv)
│   ├── interim/                       # Intermediate outputs (features, labels)
│   └── processed/                     # Final panel (panel_phase1.csv)
│
├── outputs/
│   ├── figures/                       # Evaluation charts
│   │   └── phase1_{roc_pr,calibration,deciles,trajectories}.png
│   └── pipeline_overview.html         # Pipeline diagram
│
├── docs/
│   ├── 01_PIPELINE.md                 # 8-stage pipeline documentation
│   ├── 02_OUTPUTS.md                  # Output interpretation guide
│   ├── 03_USAGE.md                    # Analyst user guide
│   ├── 04_PRESENTATION.md             # Presentation slides notes
│   └── 05_PLUGGING_IN_REAL_DATA.md    # Integration guide for Phase 2 loaders
│
├── tests/
│   └── smoke_test.py                  # Basic pipeline validation
│
├── pages/                             # Streamlit multi-page app
│   ├── 1_Model_Evaluation.py          # Charts & metrics
│   ├── 2_Firm_Analysis.py             # Individual firm deep-dives
│   └── 3_Methodology.py               # Detailed technical specs
│
├── streamlit_app.py                   # Streamlit home page
├── app.py                             # Legacy single-page version
└── .gitignore
```
"""
st.markdown(repo_structure)

st.markdown("---")

st.header("📖 Documentation")

st.write("""
**For Analysts (Non-Technical):**
- Start with: `docs/03_USAGE.md` — How to interpret risk scores and act on alerts
- Then: This dashboard (Firm Analysis page) — Drill into specific firms

**For Developers (Technical):**
- Start with: `docs/01_PIPELINE.md` — End-to-end pipeline architecture
- Then: `src/ews/config.py` — Configuration parameters
- Then: Individual module docstrings (loaders.py, features.py, models.py)

**For Extending (Phase 2 Integration):**
- Read: `docs/05_PLUGGING_IN_REAL_DATA.md` — Contract for Allen's SEC loader, Darren's 8-K mapper, FRED macros
- Reference: `src/ews/loaders.py` — Loader schema validation (`check_loader`)
- Example: `src/ews/loaders.py::load_prices` — Template for new data sources
""")

st.markdown("---")

st.header("🎯 Key Decisions & Assumptions")

decisions = """
1. **Logistic Regression Only (No Neural Networks)**
   - Reason: Interpretability for credit analysts; well-understood theory
   - Trade-off: Potentially lower predictive power than deep learning
   - Phase 2: Consider gradient boosting (XGBoost) for robustness comparison

2. **40% Drawdown Threshold**
   - Reason: Economically meaningful; balances label frequency (20% events)
   - Sensitivity: Phase 3 will test 30%/50% alternatives

3. **12-Month Horizon**
   - Reason: Aligns with credit analyst decision windows
   - Trade-off: Longer = noisier; shorter = less time for intervention

4. **Time-Based Not Cross-Validated Split**
   - Reason: Prevents lookahead bias; realistic out-of-time testing
   - Trade-off: Smaller validation set (fewer samples for evaluation)

5. **Forward-Fill Accounting Data**
   - Reason: SEC filings are sparse (quarterly/annual); monthly panel requires interpolation
   - Assumption: Balance sheet metrics don't change materially month-to-month
   - Validation: Phase 2 with real data will check this holds

6. **No Industry Dummies in Pooled Logit**
   - Reason: Comparing coefficients across models is cleaner without fixed effects at this stage
   - Trade-off: Loses industry heterogeneity signal in pooled model
   - Addressed: FE Logit captures via firm intercepts
"""
st.markdown(decisions)

st.markdown("---")

st.header("🚀 Next Steps & Roadmap")

next_steps = """
**Immediate (By May 24, 2026 → Supervisor Demo):**
1. ✅ Validate Phase 1 pipeline on 10 firms
2. ✅ Build interactive dashboard (multi-page Streamlit app)
3. ✅ Deploy to free host (Streamlit Cloud) or demo locally

**Phase 2 (May 24 - Jun 1, 2026):**
1. Expand to 60-80 firms (cross-sector sample)
2. Integrate Allen's SEC fundamentals loader (replace synthetic)
3. Test Darren's 8-K bankruptcy mapping (Label B)
4. Re-train models on fuller dataset
5. Ablation on threshold sensitivity (30%, 40%, 50% drawdown)

**Phase 3 (Early Jun):**
1. Robustness: Rolling window validation
2. Calibration improvements: Platt scaling or isotonic regression
3. Feature engineering: Interaction terms, sector rotations (macros)
4. Final model selection (pooled vs. FE vs. hazard)

**Phase 4 (Jun - Jul):**
1. Write final report + appendices
2. Create webpage with reproducibility instructions
3. Prepare oral exam slides
4. Document limitations & assumptions

**Future (Post-Capstone):**
- Real-time prediction API (REST endpoint)
- Integration with Bloomberg/Refinitiv data
- Causal inference analysis (what drives risk changes?)
- Interpretability: SHAP values for individual predictions
"""
st.markdown(next_steps)

st.markdown("---")

st.header("📞 Contact & Support")

st.info("""
**Questions about the dashboard?**
- Shrey: Market features & data extraction
- Ivan: Model evaluation & metrics interpretation
- Vincent: Overall project coordination

**Code issues?**
- Check `TODOS.md` for known limitations
- Review `docs/05_PLUGGING_IN_REAL_DATA.md` for integration contracts

**Want to contribute or extend?**
- See `CONTRIBUTING.md` for guidelines
- Fork the repo on GitHub: `credit-risk-ews`
- Submit PRs for Phase 2 enhancements
""")

st.markdown("---")

st.success("Dashboard powered by **Streamlit**. Data pipeline in **Python 3.12**. Models via **statsmodels**. 🚀")

