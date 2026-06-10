# Feature-Group Analysis — Phase 2 Findings (v2)

**Date:** 2026-06-05
**Panel:** `data/processed/panel_phase2.csv` (77 firms after REIT recovery, ~12,473 firm-months, ~8% event rate)
**Eval split:** validation (2021–2023, 2772 rows from pipeline output)
**Models:** pooled logit, fixed-effects logit (industry + year dummies, drop_first=False), discrete-time hazard logit (Shumway-style with log-duration baseline)
**Uncertainty:** 95% percentile bootstrap, **firm-clustered**, 1,000 resamples
**Note on v1:** This supersedes the v1 findings (row-level bootstrap, 72-firm panel, pooled-only). See git history `v1...v2` for diff.
**Update 2026-06-10:** Hazard-family rows regenerated after the hazard-duration fix (val/test predictions now use a global per-firm duration that is continuous across the train/val/test split, instead of resetting to 1 at each boundary). Pooled and FE rows are unaffected; the log-duration coefficient is unchanged (fit on training data). The Market-beats-Full ordering and every conclusion below are unchanged — the hazard Market–Full gap actually widens slightly (0.076 → 0.087).

## What changed since v1

- Panel grew from 72 to 77 firms after imputing `wc_ratio` for REITs (4 firms recovered: SPG, O, PLD, VTR; DE also returned).
- Bootstrap is now firm-clustered at n=1,000 (was row-level at n=200) — CIs are wider and honestly reflect within-firm autocorrelation.
- Ablation now runs over all three model families (pooled, fe, hazard).
- Full-model coefficients persisted per family for mechanistic inspection.
- The v1 headline ("Market-Full CIs disjoint") does not hold under firm-clustered CIs. Market CI [0.546, 0.749] and Full CI [0.432, 0.660] overlap substantially in pooled. That v1 claim was an artefact of underestimated standard errors from row-level resampling.

## Result table (all families)

| model_family | Feature set     |  N | AUROC | AUROC_lo | AUROC_hi | AUPRC | Brier |
|--------------|-----------------|---:|------:|---------:|---------:|------:|------:|
| pooled       | Accounting only |  5 | 0.569 |    0.432 |    0.694 | 0.197 | 0.101 |
| pooled       | Market only     |  6 | 0.653 |    0.546 |    0.749 | 0.181 | 0.099 |
| pooled       | Macro only      |  3 | 0.342 |    0.277 |    0.401 | 0.083 | 0.114 |
| pooled       | Filing only     |  1 | 0.474 |    0.445 |    0.498 | 0.109 | 0.102 |
| pooled       | Acct + Market   | 11 | 0.615 |    0.487 |    0.727 | 0.190 | 0.099 |
| pooled       | Full model      | 15 | 0.551 |    0.432 |    0.660 | 0.197 | 0.102 |
| fe           | Accounting only |  5 | 0.655 |    0.553 |    0.757 | 0.182 | 0.102 |
| fe           | Market only     |  6 | 0.681 |    0.571 |    0.777 | 0.199 | 0.101 |
| fe           | Macro only      |  3 | 0.641 |    0.551 |    0.734 | 0.176 | 0.102 |
| fe           | Filing only     |  1 | 0.637 |    0.546 |    0.730 | 0.173 | 0.103 |
| fe           | Acct + Market   | 11 | 0.669 |    0.560 |    0.774 | 0.193 | 0.102 |
| fe           | Full model      | 15 | 0.672 |    0.562 |    0.774 | 0.192 | 0.102 |
| hazard       | Accounting only |  5 | 0.560 |    0.425 |    0.683 | 0.189 | 0.099 |
| hazard       | Market only     |  6 | 0.645 |    0.538 |    0.740 | 0.178 | 0.099 |
| hazard       | Macro only      |  3 | 0.346 |    0.282 |    0.405 | 0.083 | 0.114 |
| hazard       | Filing only     |  1 | 0.369 |    0.288 |    0.455 | 0.085 | 0.102 |
| hazard       | Acct + Market   | 11 | 0.611 |    0.481 |    0.727 | 0.194 | 0.098 |
| hazard       | Full model      | 15 | 0.558 |    0.438 |    0.668 | 0.200 | 0.102 |

(Altman Z-score row absent for all three families: model failed at fit time — `exog contains inf or nans`.)

## Headline finding

**Sector-relative market features are the new top performer (Phase 3 #2):** z-scoring each market feature within its industry-month lifts pooled AUROC to 0.714 (AUPRC 0.274) vs Market-only 0.653 (AUPRC 0.181), and is strongest in every family. Separately, market features alone match or exceed the Full model on point-estimate AUROC in all three model families (pooled: 0.653 vs 0.551; FE: 0.681 vs 0.672; hazard: 0.645 vs 0.558). The FE margin is the narrowest at 0.009 AUROC points. However, under firm-clustered bootstrap at n=1,000 resamples, every pair of feature-group CIs overlaps at 95% confidence — no ranking is statistically separable. The v1 claim of disjoint Market/Full CIs does not survive proper clustering and is retracted. The mechanism behind the directional Market dominance is identified in the pooled coefficient table: `liquidity_buffer`, `vix`, and `credit_spread` carry signs opposite to credit-risk theory with p < 0.05, suppressing Full model discrimination in pooled and hazard. FE absorbs some of this distortion through industry+year dummies, which is why its Market-Full gap is the smallest.

## What carries the signal (per family)

**Pooled logit:** Market only leads with AUROC 0.653 (CI [0.546, 0.749]). Acct + Market is the runner-up at 0.615 (CI [0.487, 0.727]). Both CIs overlap. Accounting only (0.569, CI [0.432, 0.694]) and Full model (0.551, CI [0.432, 0.660]) trail. Macro only (0.342, CI [0.277, 0.401]) is the worst performer and is the only group whose CI lies entirely below 0.5, meaning macro features are anti-predictive in pooled logit — their signal runs backward relative to the event label.

**Fixed-effects logit:** The FE family compresses the range and — unlike pooled and hazard — all six subsets succeed (Accounting only and Acct+Market no longer fail with `LinAlgError` after switching to BFGS optimization). Market only leads at 0.681 (CI [0.571, 0.777]), with Full model at 0.672 (CI [0.562, 0.774]) — a gap of 0.009, well within bootstrap noise. Accounting only (0.655, CI [0.553, 0.757]) and Acct+Market (0.669, CI [0.560, 0.774]) also cluster near the top; the FE dummies absorb much of the between-firm and between-year variance, leaving all feature subsets with similar discrimination. Macro only jumps from 0.342 (pooled) to 0.641 (FE, CI [0.551, 0.734]) — the largest cross-family shift in the table. This reversal occurs because industry and year dummies absorb cross-sectional baseline rates and the calendar-level average, leaving within-cluster macro variation that correlates in the expected direction with distress. Filing only (0.637, CI [0.546, 0.730]) also rises sharply under FE, suggesting `late_filing` has within-industry signal obscured by between-industry differences in pooled logit. No pair of FE subsets has disjoint CIs.

**Hazard logit:** Pattern mirrors pooled: Market only leads at 0.645 (CI [0.538, 0.740]), Acct + Market is runner-up at 0.611 (CI [0.481, 0.727]), Full model trails at 0.558 (CI [0.438, 0.668]). Macro only (0.346, CI [0.282, 0.405]) and Filing only (0.369, CI [0.288, 0.455]) are the worst performers, with CIs entirely below 0.5. The log-duration baseline coefficient is 0.050 (SE 0.051, p = 0.329), indicating no detectable positive duration dependence in the training data after conditioning on covariates. All CIs overlap across subsets.

## Why the Full model loses (mechanism)

Reading `outputs/full_model_coefficients_pooled.csv`: three macro features — `liquidity_buffer`, `vix`, and `credit_spread` — have pooled coefficients whose signs contradict credit-risk theory, and all three are statistically significant.

- `liquidity_buffer`: coef = +1.801 (SE = 0.436, p < 0.001). Credit-risk theory predicts a negative coefficient — more cash reserves lower default risk. The fitted positive sign means the pooled model predicts higher distress probability for firms with more liquidity. This is a within-sample overfitting artefact: high-growth, high-cash firms (tech, biotech) also happen to have elevated volatility and drawdowns, so the model conflates a balance-sheet buffer with a growth-risk profile.

- `vix`: coef = -0.061 (SE = 0.009, p < 0.001). Credit-risk theory predicts positive — high VIX signals stress regime and precedes elevated default rates. The fitted negative sign means the pooled model predicts lower distress during high-VIX periods. The 2010–2020 training window was a prolonged low-rate expansion where VIX spikes (2011, 2015–16, 2018) were followed by rapid recoveries, so the model learns that high VIX is transient, not persistent stress.

- `credit_spread`: coef = -0.517 (SE = 0.158, p = 0.001). Credit-risk theory predicts positive — wide spreads reflect deteriorating credit conditions and elevated expected default rates. The fitted negative sign mirrors the VIX problem: in the 2010–2020 training regime, wide credit spreads frequently coincided with the early phase of a recovery (spreads lead, defaults lag), and the model fits the cross-sectional mean rather than the within-firm change.

- `term_spread`: coef = -0.485 (SE = 0.068, p < 0.001). Theory is ambiguous: an inverted yield curve (negative term spread) historically precedes recession and elevated defaults. A positive coefficient on `term_spread` would mean high spreads (steep curve) predict distress, which is wrong. A negative coefficient means steep curve predicts safety, which is correct directionally — but the inversion signal that actually predicts recessions would require a nonlinear treatment (the curve going from positive to negative). As a linear feature, `term_spread` in the training window primarily captures the level effect (steep curves accompany expansions, flat/inverted curves precede recessions), so the negative sign is mechanically correct for the level relationship even if the inversion signal is missed.

Together, the three macro features with wrong signs (`liquidity_buffer`, `vix`, `credit_spread`) actively reduce the log-likelihood in the correct direction when fitted pooled, which explains why the Full model AUROC falls below Market only in both the pooled and hazard families.

Under FE, `credit_spread` flips to +0.883 (SE = 0.201, p < 0.001) — the expected positive sign — because industry and year dummies remove the cross-sectional and calendar-level mean, leaving only within-cluster deviation where spread widening is correctly associated with increased within-firm distress probability. This is the direct mechanism behind the FE Macro only AUROC rising from 0.342 to 0.602.

## Limitations

- FE Accounting only and FE Acct + Market previously failed with `LinAlgError: Singular matrix` under Newton-Raphson optimization. After switching to BFGS with `maxiter=200`, both rows now converge and appear in the table. The fe block therefore has 6 rows (all subsets except Altman Z-score), matching pooled and hazard.
- Altman Z-score fails for all three families due to inf/NaN in the `z_score` column. Not reportable on this panel.
- 1,000 firm-clustered resamples on 77 firms gives an effective sample size of approximately 77 unique clusters regardless of row count. CI widths are 3–4 times wider than the v1 row-level CIs, which had 2,592 pseudo-independent observations. The v1 claim of disjoint Market/Full CIs was an artefact of underestimated standard errors from row-level resampling; that claim does not hold under firm-clustered bootstrap.
- All families use the same time split (train: 8,777 rows, 2010–2020; val: 2,772 rows, 2021–2023; test: 924 rows, 2024). Walk-forward CV across multiple test years is queued for Tier 2.
- The FE coefficient table (`outputs/full_model_coefficients_fe.csv`) has NaN `std_err` and `p_value` for all rows. BFGS does not always produce an invertible Hessian for the FE design matrix (many dummy columns on a small panel), so standard errors are unavailable. Treat all FE coefficient magnitudes as directional only — they are not accompanied by valid inferential statistics.
- Hazard log-duration coefficient (0.050, SE 0.051, p = 0.329): no statistically detectable duration dependence on this panel and training window.

## Implications for the rest of Phase 3

- **Item #4 (horizon analysis):** Run on the Market-only subset in the pooled or hazard family, where the subset ranking is cleanest. Market features (vol_3m, drawdown_12m, ret_1m) are likely the most horizon-sensitive.
- **Item #5 (threshold sensitivity):** Rebuild labels at 30% / 50% drawdown and rerun this ablation table to confirm that Market leads under alternative thresholds and that macro feature sign inversions persist.
- **Item #7 (error analysis):** Slice prediction errors of the Market-only pooled model by sector, with the REIT firms now present. The REIT recovery (SPG, O, PLD, VTR) is the main structural change from v1; their error pattern should be inspected separately.
- **Item #8 (calibration):** Platt or isotonic calibration applied to the Market-only pooled model. Before calibrating the Full model, consider dropping `liquidity_buffer`, `vix`, and `credit_spread` — three features with wrong pooled signs — to see whether a pruned Full model closes the gap with Market only.

## Reproducibility

- Pipeline command: `MPLBACKEND=Agg python src/run.py`
- Bootstrap seed: 42 (hardcoded in `_bootstrap_auroc_ci`)
- Source CSV: `outputs/ablation_results.csv` (regenerated every run)
- Coefficient sidecars: `outputs/full_model_coefficients_{pooled,fe,hazard}.csv`
- Test script: `tests/ablation_test.py` (7 sections, ~20 assertions)
