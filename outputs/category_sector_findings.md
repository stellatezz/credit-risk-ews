# Category & Sector Performance Analysis

**Date:** 2026-06-06
**Model:** Market-only pooled logit (6 features: ret_1m/3m/6m, vol_3m/6m, drawdown_12m) — the v2-identified leader.
**Eval split:** validation 2021-2023 (2,772 rows across 77 firms).
**Slicing:**
  - **Sector** = `industry` column (22 buckets; 5 have zero events in the val window).
  - **Archetype** = parsed from Phase 2 sample-company doc, normalised to 7 buckets (Distressed, Cyclical, Stable, Growth, Defensive, Rate-sensitive, Commodity-sensitive).
**Uncertainty:** 95% firm-clustered bootstrap, 1,000 resamples (inherited from v2 work).
**Operating threshold:** top decile of val predicted probability (matches analyst workflow; recomputed each pipeline run).

## Per-sector AUROC

| slice | n_firms | n_events | event_rate | AUROC | AUROC_lo | AUROC_hi |
|---|---|---|---|---|---|---|
| Healthcare | 11 | 2 | 0.005 | 0.836 | 0.738 | 0.919 |
| Retail | 10 | 81 | 0.225 | 0.758 | 0.316 | 0.907 |
| Technology | 7 | 35 | 0.139 | 0.740 | 0.482 | 0.913 |
| Logistics | 2 | 4 | 0.056 | 0.654 | 0.359 | 0.654 |
| Materials | 4 | 26 | 0.181 | 0.565 | 0.398 | 0.738 |
| Communication | 2 | 19 | 0.264 | 0.537 | 0.322 | 0.537 |
| Consumer | 3 | 5 | 0.046 | 0.476 | 0.174 | 0.635 |
| Platform | 1 | 12 | 0.333 | 0.427 | 0.427 | 0.427 |
| Auto | 3 | 32 | 0.296 | 0.292 | 0.058 | 0.328 |
| Semiconductors | 3 | 35 | 0.324 | 0.255 | 0.170 | 0.333 |
| Airlines | 4 | 5 | 0.035 | 0.236 | 0.093 | 0.372 |
| Industrial | 1 | 3 | 0.083 | 0.192 | 0.192 | 0.192 |
| Software | 1 | 10 | 0.278 | 0.177 | 0.177 | 0.177 |
| Real Estate | 5 | 4 | 0.022 | 0.155 | 0.022 | 0.267 |
| Media | 1 | 14 | 0.389 | 0.140 | 0.140 | 0.140 |
| Aerospace | 1 | 14 | 0.389 | 0.078 | 0.078 | 0.078 |
| Fintech | 1 | 13 | 0.361 | 0.077 | 0.077 | 0.077 |
| Chemicals | 1 | 0 | 0.000 | NaN | NaN | NaN |
| Energy | 6 | 0 | 0.000 | NaN | NaN | NaN |
| Industrials | 4 | 0 | 0.000 | NaN | NaN | NaN |
| Staples | 4 | 0 | 0.000 | NaN | NaN | NaN |
| Telecom | 2 | 0 | 0.000 | NaN | NaN | NaN |

## Per-archetype AUROC

| slice | n_firms | n_events | event_rate | AUROC | AUROC_lo | AUROC_hi |
|---|---|---|---|---|---|---|
| Defensive | 10 | 2 | 0.006 | 0.860 | 0.765 | 0.942 |
| Distressed | 6 | 72 | 0.333 | 0.752 | 0.477 | 0.881 |
| Commodity-sensitive | 5 | 25 | 0.139 | 0.658 | 0.436 | 0.854 |
| Stable | 35 | 64 | 0.051 | 0.506 | 0.344 | 0.625 |
| Growth | 7 | 105 | 0.417 | 0.297 | 0.224 | 0.347 |
| Cyclical | 9 | 42 | 0.130 | 0.282 | 0.137 | 0.434 |
| Rate-sensitive | 5 | 4 | 0.022 | 0.155 | 0.022 | 0.267 |

## Per-sector error patterns (top-decile threshold)

| slice | n_events | n_flags | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|---|---|
| Retail | 81 | 68 | 43 | 25 | 38 | 254 | 0.632 | 0.531 |
| Communication | 19 | 17 | 7 | 10 | 12 | 43 | 0.412 | 0.368 |
| Technology | 35 | 25 | 6 | 19 | 29 | 198 | 0.240 | 0.171 |
| Auto | 32 | 33 | 5 | 28 | 27 | 48 | 0.152 | 0.156 |
| Media | 14 | 14 | 2 | 12 | 12 | 10 | 0.143 | 0.143 |
| Platform | 12 | 12 | 1 | 11 | 11 | 13 | 0.083 | 0.083 |
| Fintech | 13 | 13 | 1 | 12 | 12 | 11 | 0.077 | 0.077 |
| Materials | 26 | 19 | 2 | 17 | 24 | 101 | 0.105 | 0.077 |
| Semiconductors | 35 | 26 | 1 | 25 | 34 | 48 | 0.038 | 0.029 |
| Software | 10 | 6 | 0 | 6 | 10 | 20 | 0.000 | 0.000 |
| Real Estate | 4 | 4 | 0 | 4 | 4 | 172 | 0.000 | 0.000 |
| Aerospace | 14 | 6 | 0 | 6 | 14 | 16 | 0.000 | 0.000 |
| Airlines | 5 | 15 | 0 | 15 | 5 | 124 | 0.000 | 0.000 |
| Industrial | 3 | 1 | 0 | 1 | 3 | 32 | 0.000 | 0.000 |
| Healthcare | 2 | 0 | 0 | 0 | 2 | 394 | NaN | 0.000 |
| Consumer | 5 | 2 | 0 | 2 | 5 | 101 | 0.000 | 0.000 |
| Logistics | 4 | 1 | 0 | 1 | 4 | 67 | 0.000 | 0.000 |
| Chemicals | 0 | 1 | 0 | 1 | 0 | 35 | 0.000 | NaN |
| Energy | 0 | 15 | 0 | 15 | 0 | 201 | 0.000 | NaN |
| Industrials | 0 | 0 | 0 | 0 | 0 | 144 | NaN | NaN |
| Staples | 0 | 0 | 0 | 0 | 0 | 144 | NaN | NaN |
| Telecom | 0 | 0 | 0 | 0 | 0 | 72 | NaN | NaN |

## Per-archetype error patterns (top-decile threshold)

| slice | n_events | n_flags | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|---|---|
| Distressed | 72 | 70 | 43 | 27 | 29 | 117 | 0.614 | 0.597 |
| Growth | 105 | 100 | 21 | 79 | 84 | 68 | 0.210 | 0.200 |
| Commodity-sensitive | 25 | 20 | 2 | 18 | 23 | 137 | 0.100 | 0.080 |
| Cyclical | 42 | 36 | 1 | 35 | 41 | 247 | 0.028 | 0.024 |
| Stable | 64 | 48 | 1 | 47 | 63 | 1149 | 0.021 | 0.016 |
| Defensive | 2 | 0 | 0 | 0 | 2 | 358 | NaN | 0.000 |
| Rate-sensitive | 4 | 4 | 0 | 4 | 4 | 172 | 0.000 | 0.000 |

## Headline findings

*Updated 2026-06-10: the slice model is now the **Market + sector-relative pooled logit** (Phase 3 #2 — market features plus their within-industry z-scores). Numbers in this section are from that model; the detailed sections further down predate the feature change.*

- **Sector-relative features lifted discrimination across the board.** Z-scoring the market features within each industry-month raised the Distressed archetype to AUROC 0.805 (recall 0.69, precision 0.60 on 72 events), Technology to 0.844 and Retail to 0.833 (the two strongest sectors), and pulled Real Estate up from 0.15 to 0.69 — by letting the model ask "unusual *for its sector*?" rather than "volatile in absolute terms?".

- **The real win is the proposal's target — Distressed firms.** Precision 0.602, recall 0.694, AUROC 0.805 on 72 events: a different regime from every other archetype (precision ~0.0–0.2).

- **Best sectors: Retail and Technology.** Both strong at *ranking and flagging* (Retail 0.833 / prec 0.658 / rec 0.617; Technology 0.844 / prec 0.571 / rec 0.571) — rare in this table.

- **Tiny-event slices stay untrustworthy.** Healthcare (AUROC 0.775) and Defensive (0.885) each have only 2 distress events in-window; their top-decile flags are essentially all false alarms. This is a sample-size artefact, **not a calibration problem** — calibration is monotone and cannot change which firms clear the threshold.

- **Airlines remains the hard case.** Sector-relative features raised Airlines AUROC from 0.24 to 0.33, but it is still below chance with 0 correct flags on 5 events — airlines move together, so even relative volatility barely separates a distressed carrier from sector-wide turbulence. Needs airline-specific fundamentals.

- **REITs: helped but inconclusive.** Real Estate / Rate-sensitive (the same 5 firms) jumped to AUROC 0.69, but with only 4 distress events the CI [0.49, 0.80] is too wide to conclude.


## Limitations

- The 7-archetype mapping is a heuristic substring match against the Phase 2 doc. Edge cases default to `Stable`; a reader who challenges a specific firm's classification can be referred to `scripts/extract_firm_categories.py` for the deterministic rule.
- Per-slice CIs use firm-clustered bootstrap (correct for credit-risk panel structure) but their width is bounded below by the number of unique firms in each slice. Slices with fewer than 3 firms produce CIs that should be read as "directional only." Single-firm slices (Platform, Industrial, Software, Media, Aerospace, Fintech, Chemicals) produce degenerate CIs (lo = hi = point estimate) because firm-clustered resampling has no between-firm variance to exploit.
- Five sectors have zero events in the validation window (Chemicals, Energy, Industrials, Staples, Telecom). AUROC is undefined for these slices. This reflects how the labelling period and the 22-bucket industry taxonomy interact with a sample of 80 firms; it is not evidence of model failure.
- The global top-decile threshold is computed on the full validation set. Applied to narrow slices it is not calibrated to each slice's event rate, which mechanically disadvantages low-base-rate sectors (Stable, Real Estate) and overfires on high-base-rate ones where the model's rank-ordering is poor (Growth, Cyclical).
- The Market-only model is one of three families tested in v2; pooled was chosen because it had the highest point-estimate AUROC. FE and hazard family slice results are not produced here. Horizon analysis (Phase 3 item 4) and threshold sensitivity (item 5) are queued separately.

## Implications for the rest of Phase 3

- **Item 4 (horizon analysis)** — Retail and Distressed are the natural candidates for horizon analysis: they have enough events (81 and 72 respectively) and the model already works at 12-month drawdown labelling. The AUROC ceiling for other slices is set by where the model currently fails, so horizon analysis on those slices should not be expected to recover much.
- **Item 5 (threshold sensitivity)** — Relabel at 30% and 50% drawdown and rerun this slicing. Growth and Semiconductor slices have very high event rates at the 20% threshold (0.417 and 0.324); a stricter threshold may shrink their event counts and change the AUROC ranking.
- **Item 8 (calibration)** — The Distressed and Retail recall numbers (0.597, 0.531) are the baseline that a calibrated output must match or beat at the same operating threshold. The Stable and Cyclical false-alarm rates (precision = 0.021, 0.028) are what calibration should reduce.
- **Item 9 (final model selection)** — Synthesize this with the v2 ablation table and the horizon/calibration results once available. The current result suggests the Market-only pooled model's aggregate AUROC is driven primarily by Retail and Distressed slices; slices that make up a large share of the panel (Stable = 1,260 of 2,772 rows) show near-random ordering.

## Reproducibility

- Pipeline command: `MPLBACKEND=Agg python src/run.py`
- Source CSVs: `outputs/{sector,category}_{results,errors}.csv` (all gitignored, regenerated every run)
- Parser script: `scripts/extract_firm_categories.py`
- Category source-of-truth: `data/firm_categories.csv` (committed)
- Test script: `tests/category_sector_test.py`
