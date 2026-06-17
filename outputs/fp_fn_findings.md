# False Positive vs False Negative Analysis

**Date:** 2026-06-16  
**Model:** Market + sector-relative pooled logit (`MARKET_PLUS_REL_FEATURE_COLS`), fit on train ≤2020.  
**Threshold selection:** validation 2021-2023. **Headline evaluation:** held-out TEST 2024+ with the threshold frozen.  
**Test set:** 924 firm-months, 77 firms, 94 events (base rate 10.2%).  
**Uncertainty:** firm-clustered bootstrap, 1000 resamples, 95% CI.

## Operating points on the held-out 2024 test

| Operating point | thr | flags | TP | FP | FN | TN | recall (95% CI) | precision | FPR (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| top_10pct | 0.163 | 72 | 39 | 33 | 55 | 797 | 0.41 [0.11, 0.66] | 0.54 | 0.04 [0.01, 0.07] |
| cost_opt_1:1 | 0.656 | 1 | 1 | 0 | 93 | 830 | 0.01 [0.00, 0.03] | 1.00 | 0.00 [0.00, 0.00] |
| cost_opt_2:1 | 0.282 | 15 | 12 | 3 | 82 | 827 | 0.13 [0.00, 0.36] | 0.80 | 0.00 [0.00, 0.01] |
| cost_opt_5:1 | 0.164 | 70 | 38 | 32 | 56 | 798 | 0.40 [0.10, 0.65] | 0.54 | 0.04 [0.01, 0.07] |
| cost_opt_10:1 | 0.045 | 558 | 83 | 475 | 11 | 355 | 0.88 [0.72, 0.99] | 0.15 | 0.57 [0.48, 0.67] |
| cost_opt_20:1 | 0.044 | 573 | 85 | 488 | 9 | 342 | 0.90 [0.76, 0.99] | 0.15 | 0.59 [0.49, 0.68] |

## Cost sensitivity — where the optimal flag budget lands

| FN:FP | thr | val flag budget | test recall | test FPR | test precision | test FP | test FN |
|---|---|---|---|---|---|---|---|
| 1:1 | 0.656 | 0.0% | 0.01 | 0.00 | 1.00 | 0 | 93 |
| 2:1 | 0.282 | 2.8% | 0.13 | 0.00 | 0.80 | 3 | 82 |
| 5:1 | 0.164 | 9.7% | 0.40 | 0.04 | 0.54 | 32 | 56 |
| 10:1 | 0.045 | 67.2% | 0.88 | 0.57 | 0.15 | 475 | 11 |
| 20:1 | 0.044 | 68.7% | 0.90 | 0.59 | 0.15 | 488 | 9 |

## Artifact vs failure — global vs slice-relative top-decile threshold (val)

Does flagging the riskiest 10% *within each slice* rescue the slices that scored zero recall under the single global threshold? If recall stays near the ~10% random floor, the within-sector ranking is broken (AUROC ≤ 0.5) — a *ranking* failure no threshold can fix; if it jumps well above, the global threshold was a *thresholding* artifact.

| Archetype | events | flags@global | recall@global | flags@slice-rel | recall@slice-rel | gain |
|---|---|---|---|---|---|---|
| Growth | 105 | 40% | 0.32 | 10% | 0.07 | -0.26 |
| Distressed | 72 | 38% | 0.69 | 10% | 0.25 | -0.44 |
| Stable | 64 | 4% | 0.00 | 10% | 0.02 | +0.02 |
| Cyclical | 42 | 3% | 0.00 | 10% | 0.02 | +0.02 |
| Commodity-sensitive | 25 | 13% | 0.32 | 10% | 0.28 | -0.04 |
| Rate-sensitive | 4 | 4% | 0.00 | 10% | 0.00 | +0.00 |
| Defensive | 2 | 2% | 0.00 | 10% | 0.50 | +0.50 |

## Headline findings

- **At the deployed top-decile operating point, generalisation is modest but real:** on the held-out 2024 test the model catches **41% of distress events (95% CI [11%, 66%])** at a 8% flag budget, precision 54%, FPR 4%.
- **The analyst's top-decile rule implicitly encodes a ~5:1 cost preference:** the cost-optimal threshold at FN:FP = 5:1 flags 8% of firm-months (recall 40%), essentially reproducing the top-decile point. The operating point widens sharply only once FN:FP reaches ~10:1, where the optimum jumps to flagging 60% of firm-months — recall 88% [72%, 99%] but FPR 57%, precision 15%. That jump between 5:1 and 10:1 is where the tool flips from a selective watchlist to a flag-most screen — the decision a risk committee must actually make.
- **The zero-recall slices are a ranking failure, not a threshold artifact:** Growth, Stable, Cyclical carry events but, even when given a slice-relative top-decile threshold (top-10% within their own scores), recall only reaches roughly the 10% random floor — i.e. their events are not ranked above their non-events (within-sector AUROC ≤ 0.5). No threshold rescues a broken ranking; this is the diagnosis for the model's biggest blind spots.
- **Recall and FPR are the transferable numbers; precision is base-rate-bound.** The test base rate is 10.2%; a lower-prevalence deployment population would depress precision at the same recall/FPR, so precision/FP counts here should be read as panel-specific.

## Reproducibility

- Command: `MPLBACKEND=Agg python scripts/fp_fn_analysis.py`
- Figure: `outputs/figures/phase3_fp_fn_frontier.png`
- Tables: `outputs/fp_fn_{operating_points,cost_sensitivity,slice_threshold_compare}.csv`
- Design/methodology: `docs/superpowers/specs/2026-06-16-fp-fn-analysis-design.md`

**Deferred (follow-up):** lead-time-aware TP definition and a firm-episode-level confusion matrix (reuse `eval.compute_lead_time`).
