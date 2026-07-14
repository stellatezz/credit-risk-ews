# Credit Risk Early-Warning System (EWS)

An interpretable, open-data credit risk early-warning system for US-listed non-financial firms, 2010–2024. Each month, for each firm in the panel, the pipeline outputs a probability of **credit deterioration** over the next 12 months — together with a risk-trajectory view for analyst watchlist and escalation triage.

Not for trading. For credit-risk oversight.

Every model is a standard interpretable regression (logistic, fixed-effects, discrete-time hazard) whose coefficients a practitioner can read and argue with. No deep learning, no black boxes. Every data source is open and free (SEC EDGAR, yfinance, FRED) — no Bloomberg, no paid feeds.

## Team

| Role | Name |
| :--- | :--- |
| Mentor | Yi Chen |
| Leader | Hui Fai Wong |
| Contributor (market features) | Anchalwar Shrey Sanjay |
| Contributor (labels & filing signals) | Fung Tat Ki (Darren) |
| Contributor (econometrics & evaluation) | Chow Pak Ho (Ivan) |
| Contributor (SEC fundamentals) | Yu Yuk Lam Allen |

## Where we are

**Phase 2 + Phase 3 diagnostics — current as of June 2026.** The live panel is `data/processed/panel_phase2.csv`: **77 firms, 12,473 firm-months, 2010–2024**, with **real SEC EDGAR fundamentals** (Allen) and **real FRED macros** now integrated. The earlier 10-firm toy panel (`panel_phase1.csv`) is retained — it validated the pipeline end-to-end. The one data source still outstanding is **8-K bankruptcy labels** (`label_b`, Darren); the panel currently labels on **`label_a`** alone (≥40% peak-to-trough drawdown over the next 12 months, ~8.5% base rate).

**Diagnostics delivered (Phase 3).** Sector-relative ("within-industry") market features; probability **calibration** (Platt + isotonic); **per-slice evaluation** by sector and firm archetype with firm-clustered CIs; **feature-group ablation**; and a **false-positive / false-negative cost analysis** (cost frontier, cost-weighted thresholds, evaluation on a held-out 2024 test set, Altman-Z benchmark). Results are browsable in the **Streamlit app** (`project_home.py` + `pages/`), hosted from GitHub off committed artifacts in `outputs/`.

**Models.** The committed, deployed model is the **pooled logistic regression**; the deployed *slice* model adds the **sector-relative features** (raw market features plus their within-industry z-scores). A **fixed-effects panel logit** and a **Shumway-style discrete-time hazard logit** are also fit (the hazard duration bug is fixed). In the current ablation FE scores highest, but its industry/year dummies don't transfer forward cleanly, so **pooled remains the deployed choice** — treat FE/hazard metrics as provisional.

> **Honest limitation.** The model works as a top-decile watchlist for already-distressed firms and beats Altman Z, but it cannot rank distress within Stable/Cyclical/Growth firms (within-sector AUROC ≈ 0.5). That gap is a *ranking* failure, not a thresholding one, and is not fixed by the SEC fundamentals in the panel (they don't help against a price-drawdown label) — see `outputs/fp_fn_findings.md`.

## Repo map

```
.
├── README.md                     ← you are here
├── CONTRIBUTING.md               ← how to plug work in
├── TODOS.md                      ← backlog
├── requirements.txt              ← pinned deps
├── project_home.py               ← Streamlit app home (multipage; see pages/)
├── docs/
│   ├── 01_PIPELINE.md            ← data flow: inputs → outputs (ASCII diagram + module map)
│   ├── 02_OUTPUTS.md             ← what comes out + how to read the charts
│   ├── 03_USAGE.md               ← how an analyst uses the outputs (1-page workflow)
│   ├── 04_PRESENTATION.md        ← how to present to markers / committee
│   ├── 05_PLUGGING_IN_REAL_DATA.md  ← HOWTO for Allen + Darren (loader contract)
│   └── superpowers/              ← design specs + plans (e.g. the FP/FN analysis design)
├── src/
│   ├── ews/                      ← the pipeline package (see docs/01_PIPELINE.md)
│   │   ├── config.py             ← firms, features, thresholds, paths
│   │   ├── loaders.py            ← team-facing data-source contract
│   │   ├── features.py           ← market-feature engineering
│   │   ├── labels.py             ← Label A construction (Label B goes here too)
│   │   ├── panel.py              ← merge + sector-relative features + time split
│   │   ├── models.py             ← pooled / FE / hazard logit
│   │   ├── eval.py               ← metrics + ablation + per-slice + calibration
│   │   ├── viz.py                ← evaluation charts
│   │   └── pipeline.py           ← the orchestrator (main)
│   └── run.py                    ← entry point (thin wrapper; sets sys.path, calls ews.pipeline.main)
├── pages/                        ← Streamlit pages (0 = LIVE analyst watchlist — the deployed
│                                    model scoring live yfinance/EDGAR data, add-any-ticker;
│                                    1–9 = Model Eval, Firm, Methodology, About, Feature-Group,
│                                    Sector/Category, FP/FN, Horizon, Threshold diagnostics)
├── scripts/                      ← standalone analyses
│   ├── extract_firm_categories.py   ← archetype parser
│   ├── fp_fn_analysis.py            ← FP/FN cost frontier + held-out-test operating points
│   └── fundamentals_slice_test.py   ← do fundamentals rescue the failing slices?
├── tests/
│   ├── smoke_test.py
│   ├── ablation_test.py
│   └── category_sector_test.py
├── data/
│   ├── raw/                      ← yfinance cache (committed for reproducibility)
│   ├── interim/                  ← one CSV per loader (inspectable)
│   └── processed/
│       ├── panel_phase1.csv      ← 10-firm toy panel (1,490 × 23) — legacy
│       └── panel_phase2.csv      ← live panel (12,473 firm-months × 32, 77 firms)
├── outputs/                      ← committed so the Streamlit app can be hosted from GitHub
│   ├── figures/                  ← phase1 / phase2 / phase3 charts + FP/FN frontier
│   ├── *_results.csv, *_errors.csv, *_findings.md   ← ablation, per-slice, FP/FN
│   └── pipeline_overview.html
└── reference/                    ← canonical source of truth
    ├── Detailed Proposal v1.docx.md
    └── Team Catch v1 Apr 16.docx.md
```

## Quickstart

```bash
# One-time setup — isolates deps from system Python (avoids PEP 668 errors on macOS)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline (first run hits yfinance ~30–60s; subsequent runs use data/raw/ cache)
python src/run.py
# or equivalently (PYTHONPATH=src is needed because there's no pyproject.toml yet):
PYTHONPATH=src python -m ews.pipeline

# Regenerate the FP/FN cost analysis (standalone)
MPLBACKEND=Agg python scripts/fp_fn_analysis.py

# Browse all results in the dashboard
streamlit run project_home.py

# BEFORE A LIVE DEMO: pre-warm the Live Watchlist caches (prices + EDGAR + scores)
# so the analyst monitoring page loads instantly on stage
python scripts/warm_live_cache.py

# Verify your setup
python tests/smoke_test.py
```

Outputs are written automatically — no manual copying required:

- `data/processed/panel_phase2.csv` — the live modeling panel (12,473 firm-months × 32 columns, 77 firms)
- `data/interim/{prices,market_features,fundamentals,macros,labels}.csv` — per-source intermediate tables (Excel-friendly, with a provenance header)
- `outputs/figures/phase2_{roc_pr,calibration,deciles,trajectories}.png` — core evaluation charts
- `outputs/figures/phase3_{calibration_compare,fp_fn_frontier}.png` + `outputs/{ablation,sector,category,fp_fn}_*.{csv,md}` — Phase 3 diagnostics

## Data sources

Four families feed the panel. Full schema contract and real-loader HOWTO lives in `docs/05_PLUGGING_IN_REAL_DATA.md`.

| Family | Source | Owner | Status |
| :--- | :--- | :--- | :--- |
| Equity prices | yfinance (real, cached) | Shrey | integrated |
| Firm fundamentals | SEC EDGAR (real) | Allen | integrated |
| Macro stress | FRED (real) | — | integrated |
| Labels | Label A drawdown (real); 8-K `label_b` | Darren | `label_a` done; `label_b` pending |

No-look-ahead is a hard rule: fundamentals align to **filing date**, not period-end; market and macro aggregate to month-end using only data available at time t.

## Where to read next

- **How it works?** → `docs/01_PIPELINE.md` (data flow + ASCII diagram + module map).
- **What the charts mean?** → `docs/02_OUTPUTS.md`.
- **How an analyst uses it?** → `docs/03_USAGE.md`.
- **Preparing slides / oral exam?** → `docs/04_PRESENTATION.md`.
- **Contributing real data (Allen / Darren)?** → `docs/05_PLUGGING_IN_REAL_DATA.md`.
- **Coding conventions / first-time setup?** → `CONTRIBUTING.md`.
- **Phase 2 backlog?** → `TODOS.md`.
- **Canonical source of truth?** → `reference/Detailed Proposal v1.docx.md`.

## One-paragraph project summary for outsiders

Each month, for each of ~80 US-listed non-financial firms, the pipeline reads SEC filings, stock prices, and macro stress indicators, and outputs a probability between 0 and 1 that the firm will experience **market-implied credit deterioration** — a peak-to-trough equity drawdown of at least 40% — at some point over the next 12 months. A credit analyst can then focus their review time on the highest-probability firms (a "monitoring scorecard" for watchlist triage). Everything uses open, reproducible data, and every model is an interpretable regression whose coefficients can be read and argued with.
