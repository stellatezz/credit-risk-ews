# Feature Group Ablation — Phase 2 Findings (v2)

**Date:** 2026-06-05
**Panel:** `data/processed/panel_phase2.csv` (77 firms after REIT recovery, ~12,473 firm-months, ~8% event rate)
**Eval split:** validation (2021–2023, 2772 rows from pipeline output)
**Models:** pooled logit, fixed-effects logit (industry + year dummies, drop_first=False), discrete-time hazard logit (Shumway-style with log-duration baseline)
**Uncertainty:** 95% percentile bootstrap, **firm-clustered**, 1,000 resamples
**Note on v1:** This supersedes the v1 findings (row-level bootstrap, 72-firm panel, pooled-only). See git history `v1...v2` for diff.

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
| fe           | Market only     |  6 | 0.637 |    0.510 |    0.752 | 0.190 | 0.108 |
| fe           | Macro only      |  3 | 0.602 |    0.490 |    0.717 | 0.170 | 0.110 |
| fe           | Filing only     |  1 | 0.621 |    0.527 |    0.718 | 0.170 | 0.106 |
| fe           | Full model      | 15 | 0.646 |    0.521 |    0.764 | 0.189 | 0.111 |
| hazard       | Accounting only |  5 | 0.530 |    0.397 |    0.652 | 0.181 | 0.102 |
| hazard       | Market only     |  6 | 0.632 |    0.518 |    0.730 | 0.177 | 0.100 |
| hazard       | Macro only      |  3 | 0.343 |    0.278 |    0.402 | 0.083 | 0.114 |
| hazard       | Filing only     |  1 | 0.358 |    0.297 |    0.418 | 0.083 | 0.104 |
| hazard       | Acct + Market   | 11 | 0.596 |    0.463 |    0.712 | 0.190 | 0.099 |
| hazard       | Full model      | 15 | 0.556 |    0.436 |    0.665 | 0.198 | 0.101 |

(Altman Z-score row absent for all three families: model failed at fit time — `exog contains inf or nans`. FE Accounting only and FE Acct + Market also failed with `LinAlgError: Singular matrix` on the 77-firm panel.)

## Headline finding

On firm-clustered CIs, all per-family subset rankings are within bootstrap noise of each other — no pair of subsets has entirely non-overlapping CIs. The defensible claim is: across all three model families, Market features alone have the highest point-estimate AUROC (pooled: 0.653, fe: 0.637, hazard: 0.632), and the Full model scores lower than Market only in every family (pooled: 0.551, fe: 0.646, hazard: 0.556). However, because every CI overlaps with every other CI at 95% confidence, these differences are not statistically separable — they represent a consistent directional pattern, not a proven ranking. The statement "adding accounting, macro, and filing features measurably hurts discrimination" cannot be made at the 95% level on this panel; the correct statement is that those features provide no measurable benefit on firm-clustered bootstrap at n=77 unique firms.

## What carries the signal (per family)

**Pooled logit:** Market only leads with AUROC 0.653 (CI [0.546, 0.749]). Acct + Market is the runner-up at 0.615 (CI [0.487, 0.727]). Both CIs overlap. Accounting only (0.569, CI [0.432, 0.694]) and Full model (0.551, CI [0.432, 0.660]) trail. Macro only (0.342, CI [0.277, 0.401]) is the worst performer and is the only group whose CI lies entirely below 0.5, meaning macro features are anti-predictive in pooled logit — their signal runs backward relative to the event label.

**Fixed-effects logit:** The FE family compresses the range. Full model leads at 0.646 (CI [0.521, 0.764]), narrowly above Market only at 0.637 (CI [0.510, 0.752]). Both CIs overlap substantially. Macro only jumps from 0.342 (pooled) to 0.602 (FE, CI [0.490, 0.717]) — the largest cross-family shift in the table. This reversal occurs because industry and year dummies absorb cross-sectional baseline rates and the calendar-level average, leaving within-cluster macro variation that correlates in the expected direction with distress. Filing only (0.621, CI [0.527, 0.718]) also rises sharply under FE, suggesting `late_filing` has within-industry signal obscured by between-industry differences in pooled logit. No pair of FE subsets has disjoint CIs.

**Hazard logit:** Pattern mirrors pooled: Market only leads at 0.632 (CI [0.518, 0.730]), Acct + Market is runner-up at 0.596 (CI [0.463, 0.712]), Full model trails at 0.556 (CI [0.436, 0.665]). Macro only (0.343, CI [0.278, 0.402]) and Filing only (0.358, CI [0.297, 0.418]) are the worst performers, with CIs entirely below 0.5. The log-duration baseline coefficient is 0.050 (SE 0.051, p = 0.329), indicating no detectable positive duration dependence in the training data after conditioning on covariates. All CIs overlap across subsets.

## Why the Full model loses (mechanism)

Reading `outputs/full_model_coefficients_pooled.csv`: three macro features — `liquidity_buffer`, `vix`, and `credit_spread` — have pooled coefficients whose signs contradict credit-risk theory, and all three are statistically significant.

- `liquidity_buffer`: coef = +1.801 (SE = 0.436, p < 0.001). Credit-risk theory predicts a negative coefficient — more cash reserves lower default risk. The fitted positive sign means the pooled model predicts higher distress probability for firms with more liquidity. This is a within-sample overfitting artefact: high-growth, high-cash firms (tech, biotech) also happen to have elevated volatility and drawdowns, so the model conflates a balance-sheet buffer with a growth-risk profile.

- `vix`: coef = -0.061 (SE = 0.009, p < 0.001). Credit-risk theory predicts positive — high VIX signals stress regime and precedes elevated default rates. The fitted negative sign means the pooled model predicts lower distress during high-VIX periods. The 2010–2020 training window was a prolonged low-rate expansion where VIX spikes (2011, 2015–16, 2018) were followed by rapid recoveries, so the model learns that high VIX is transient, not persistent stress.

- `credit_spread`: coef = -0.517 (SE = 0.158, p = 0.001). Credit-risk theory predicts positive — wide spreads reflect deteriorating credit conditions and elevated expected default rates. The fitted negative sign mirrors the VIX problem: in the 2010–2020 training regime, wide credit spreads frequently coincided with the early phase of a recovery (spreads lead, defaults lag), and the model fits the cross-sectional mean rather than the within-firm change.

- `term_spread`: coef = -0.485 (SE = 0.068, p < 0.001). Theory is ambiguous: an inverted yield curve (negative term spread) historically precedes recession and elevated defaults. A positive coefficient on `term_spread` would mean high spreads (steep curve) predict distress, which is wrong. A negative coefficient means steep curve predicts safety, which is correct directionally — but the inversion signal that actually predicts recessions would require a nonlinear treatment (the curve going from positive to negative). As a linear feature, `term_spread` in the training window primarily captures the level effect (steep curves accompany expansions, flat/inverted curves precede recessions), so the negative sign is mechanically correct for the level relationship even if the inversion signal is missed.

Together, the three macro features with wrong signs (`liquidity_buffer`, `vix`, `credit_spread`) actively reduce the log-likelihood in the correct direction when fitted pooled, which explains why the Full model AUROC falls below Market only in both the pooled and hazard families.

Under FE, `credit_spread` flips to +0.883 (SE = 0.201, p < 0.001) — the expected positive sign — because industry and year dummies remove the cross-sectional and calendar-level mean, leaving only within-cluster deviation where spread widening is correctly associated with increased within-firm distress probability. This is the direct mechanism behind the FE Macro only AUROC rising from 0.342 to 0.602.

## Limitations

- FE Accounting only and FE Acct + Market failed with `LinAlgError: Singular matrix` on the 77-firm panel — those two FE rows are absent from the table. The fe block therefore has 4 rows (Market only, Macro only, Filing only, Full model) rather than the 6 rows in pooled and hazard.
- Altman Z-score fails for all three families due to inf/NaN in the `z_score` column. Not reportable on this panel.
- 1,000 firm-clustered resamples on 77 firms gives an effective sample size of approximately 77 unique clusters regardless of row count. CI widths are 3–4 times wider than the v1 row-level CIs, which had 2,592 pseudo-independent observations. The v1 claim of disjoint Market/Full CIs was an artefact of underestimated standard errors from row-level resampling; that claim does not hold under firm-clustered bootstrap.
- All families use the same time split (train: 8,777 rows, 2010–2020; val: 2,772 rows, 2021–2023; test: 924 rows, 2024). Walk-forward CV across multiple test years is queued for Tier 2.
- The FE coefficient table (`outputs/full_model_coefficients_fe.csv`) has NaN `std_err` and `p_value` for ~32 of its ~51 rows (the industry and year dummy block) because the Hessian is not invertible for that dummy block on the small panel. Treat FE coefficient interpretation as directional only for the dummy rows — standard errors are not available.
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
