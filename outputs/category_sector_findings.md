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

- **Where the AUROC ranking misleads:** Healthcare leads all sectors at AUROC = 0.836 (95% CI 0.738-0.919, 11 firms, 2 events), but the model produces zero flags for this sector at the top-decile threshold — both events are missed, recall = 0.000. The high AUROC reflects that when the model does assign scores, it rank-orders correctly, but at the operating threshold it never commits to flagging a Healthcare firm. The same pattern holds for Defensive archetype (AUROC = 0.860, CI 0.765-0.942, 10 firms, 2 events): zero flags, recall = 0.000. These are the best-ranked slices by AUROC and the worst by actual detection at the chosen threshold.

- **Where the model acts as well as it ranks:** Retail is the only sector where AUROC (0.758, CI 0.316-0.907) and operational performance align. It produces 68 flags, catches 43 of 81 events (recall = 0.531), and achieves the highest sector precision (0.632). Distressed is the archetype equivalent: AUROC = 0.752 (CI 0.477-0.881), 70 flags, 43 of 72 events caught (recall = 0.597), precision = 0.614. These two slices together account for the majority of the model's true-positive output.

- **REIT result (paying off v2's recovery):** Real Estate sector (5 firms, 4 events in val) has AUROC = 0.155 (CI 0.022-0.267). At the top-decile threshold it produces 4 flags, all false positives — recall = 0.000, precision = 0.000. The Rate-sensitive archetype (which overlaps heavily with REITs; 5 firms, 4 events) has AUROC = 0.155 (CI 0.022-0.267), 4 flags, all false positives — recall = 0.000, precision = 0.000. The v2 REIT recovery expanded the panel from 0 to 5 Real Estate firms and confirmed they carry a distinct market footprint; however, with only 4 distress events in the validation window the sample is too small to draw a reliable conclusion. The AUROC is near-random and the CI almost spans zero to one-quarter, so the honest reading is: directional only, not conclusive.

- **Where false alarms concentrate:** Airlines is the most consequential false-alarm sector among those with actual events: 15 flags, 0 TP, 5 events missed — precision = 0.000, recall = 0.000. That combination of the worst recall and the worst flag-volume-to-event-count ratio (15 flags for 5 events) indicates the model is structurally unable to flag the right Airlines firms at this threshold; it fires on airline-like market features but never on the firms that actually distress. Energy adds 15 more pure false alarms in a sector with 0 val events (AUROC undefined). Semiconductors has a higher flag volume (26 flags) and catches 1 of 35 events (precision = 0.038, recall = 0.029), making it the lowest non-zero precision sector — but it is not the headline false-alarm case because some of its flags are true positives. At the archetype level, Stable has 48 flags across 1,260 rows but only 1 TP (precision = 0.021, recall = 0.016); Cyclical has 36 flags and 1 TP (precision = 0.028, recall = 0.024). These are the slices where the model over-fires: the base event rate is low (Stable 0.051, Cyclical 0.130) but the threshold is global, so it transfers false alarms onto firms whose market-feature profile superficially resembles distress.

- **Where events are most missed:** Growth has the largest FN count: 84 of 105 events missed, recall = 0.200. Semiconductors misses 34 of 35 events, recall = 0.029. At the sector level, five industries — Chemicals, Energy, Industrials, Staples, Telecom — have zero events in the validation window; AUROC is undefined and no conclusion about model fit is possible for these sectors.

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
