# Feature Group Ablation — Phase 2 Findings

**Date:** 2026-06-03
**Panel:** `data/processed/panel_phase2.csv` (72 firms, 11,496 firm-months, 8.7% event rate)
**Eval split:** validation (2021–2023)
**Model:** pooled logistic regression (statsmodels)
**Uncertainty:** 95% percentile bootstrap on validation AUROC, 200 resamples

## Result table

| Feature set     |  N | AUROC | AUROC_lo | AUROC_hi | AUPRC | Brier |
|-----------------|---:|------:|---------:|---------:|------:|------:|
| Accounting only |  4 | 0.570 |    0.536 |    0.608 | 0.205 | 0.106 |
| Market only     |  6 | 0.678 |    0.651 |    0.704 | 0.206 | 0.103 |
| Macro only      |  3 | 0.344 |    0.313 |    0.375 | 0.088 | 0.116 |
| Filing only     |  1 | 0.473 |    0.450 |    0.502 | 0.115 | 0.107 |
| Acct + Market   | 10 | 0.644 |    0.609 |    0.674 | 0.213 | 0.103 |
| Full model      | 14 | 0.595 |    0.564 |    0.628 | 0.217 | 0.104 |

(Altman Z-score row omitted: model failed at fit time due to inf/NaN in `z_score` on the Phase 2 panel.)

## Headline finding

Market features alone (AUROC 0.678, CI [0.651, 0.704]) are the dominant predictors of 12-month deterioration on the Phase 2 panel, and their lower CI bound (0.651) exceeds the upper CI bound of the Full model (0.628) — meaning the two CIs are entirely disjoint: adding accounting, macro, and filing features to market signals measurably *hurts* out-of-sample discrimination, not merely fails to help.

## What carries the signal

- **Market features (AUROC 0.678, CI [0.651, 0.704]):** The strongest single group by a clear margin. Six market features — including 1-month return, 3-month volatility, and 12-month drawdown — capture the bulk of predictable credit-risk variation on this panel. The CI lower bound (0.651) exceeds the point estimate of every other group, including Acct + Market (0.644), confirming this is not a noise result.

- **Accounting features (AUROC 0.570, CI [0.536, 0.608]):** Meaningful signal above coin-flip. Real SEC fundamentals (leverage, liquidity buffer, working capital ratio, profitability) push AUROC about 7 points above 0.5, but the CI lies entirely below Market only. When combined with market features (Acct + Market: 0.644, CI [0.609, 0.674]), performance improves over accounting alone but still falls short of Market only, suggesting accounting features partially overlap with or add noise relative to the market signal at this sample size.

- **Macro features (AUROC 0.344, CI [0.313, 0.375]):** Below 0.5, meaning the macro features (VIX, term spread, credit spread) are worse than a coin flip on this panel. Their CI is entirely below 0.5. Flipping the sign of the macro predictions would produce a model with AUROC ≈ 0.656 — which would rank second. This is consistent with the Phase 1 prototype observation that macro features may capture risk-on periods (low VIX, tight spreads) as safe when those are exactly the conditions preceding latent distress buildup. At Phase 2 scale the CI lies entirely below 0.5, ruling out coin-flip at the 95% level.

- **Filing features (`late_filing` alone, AUROC 0.473, CI [0.450, 0.502]):** Essentially no signal. The CI straddles 0.5 (upper bound 0.502), so the discriminative value of a single binary indicator for late SEC filings cannot be distinguished from chance on the Phase 2 panel. The filing indicator's predictive content is negligible at this event rate and panel composition.

## Limitations

- Panel excludes 4 of 5 REITs (`O`, `PLD`, `SPG`, `VTR`) and `DE`, dropped by the
  panel `dropna` because `wc_ratio` is structurally NaN for REITs and not
  reported for `DE`. Accounting-only and Full-model rows therefore reflect a
  panel skewed away from rate-sensitive firms — fix tracked separately.
- Bootstrap CIs are computed on the validation split (2,592 rows, 12.0% event rate,
  spanning 2021–2023). At 2,592 observations the bootstrap is well-powered for
  the group-level comparison reported here, but subset CIs reflect the same 2,592
  rows fitted on different feature subsets — not subsets of rows.
- Altman Z-score row is absent: the model failed at fit time with
  `exog contains inf or nans` on the Phase 2 panel. This is a pre-existing
  behavior inherited from Phase 1 (NaN/inf in the synthetic `z_score` column for
  some sectors); fix tracked separately.
- All models are pooled logit; fixed-effects and hazard variants are not
  ablated here (FE and hazard models achieve AUROC 0.662 and 0.669 respectively
  on validation, but feature-group ablation is run only on the pooled model for
  comparability across subsets).

## Implications for Phase 3

- **Item #8 (calibration):** Market-only model (6 features) is the priority target for calibration. The full model's lower AUROC despite more features suggests the extra parameters introduce miscalibration; calibration plots should be produced separately for Market only vs. Full model to confirm. Platt scaling or isotonic regression applied to the market-only predictions is the recommended next step.
- **Item #4 (horizon analysis):** Run horizon sensitivity (6-month, 18-month, 24-month labels) on the Market-only subset first. Market features — particularly 1-month return and 12-month drawdown — are likely most horizon-sensitive, and isolating them makes the signal shift interpretable without accounting/macro noise.
- **Item #7 (error analysis):** Slice false negatives and false positives of the Market-only model by GICS sector. Given that macro features backfire (AUROC < 0.5), the error pattern may cluster in rate-sensitive sectors (REITs, utilities, telecoms) where macro conditions correlate with idiosyncratic risk differently from the rest of the panel. This would motivate sector-conditional macro adjustments in Phase 4.
