# Ablation v2 — Rigor Pass (Tier 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the v1 ablation finding ("Market features alone beat the Full model on disjoint 95% CIs") against four viva-defense weaknesses: panel composition bias (REIT silent-drop), within-firm autocorrelation (row-level bootstrap), single-model-family fragility (pooled only), and missing mechanistic explanation (no coefficient inspection).

**Architecture:** All changes layer onto the existing `src/ews/eval.py::ablation_analysis` plus its two upstream collaborators (`panel.py` for the REIT-drop fix; `models.py` interfaces re-used as-is for the model-family loop). The ablation result schema gains two columns (`model_family`, and the existing CI columns now reflect firm-clustered uncertainty); a new sidecar artifact (`outputs/full_model_coefficients_<family>.csv`) is written each pipeline run. No new external dependencies. Plan v1's plumbing (PATHS.OUTPUTS, CSV persist, Filing-only group) is left untouched.

**Tech Stack:** pandas, numpy, statsmodels.Logit (existing), no new libraries.

---

## File Structure

- **Modify:** `src/ews/panel.py` — impute `wc_ratio` NaN to 0.0 and emit `wc_ratio_missing` binary flag before `dropna`
- **Modify:** `src/ews/config.py` — add `wc_ratio_missing` to `FEATURE_COLS`
- **Modify:** `src/ews/eval.py` — extend `_bootstrap_auroc_ci` with optional `firm_ids`, bump default `n_boot=1000`; extend `ablation_analysis` to loop over model families; add `persist_full_model_coefficients` helper
- **Modify:** `src/ews/pipeline.py` — call coefficient-persist helper after the existing model fits
- **Modify:** `tests/ablation_test.py` — extend with REIT-back-in-panel + clustered-bootstrap + model-family-column assertions
- **Modify:** `outputs/ablation_findings.md` — replace v1 content with v2 (uses updated CSV numbers)
- **Generated** (gitignored): `outputs/ablation_results.csv` (now with `model_family` column), `outputs/full_model_coefficients_pooled.csv`, `outputs/full_model_coefficients_fe.csv`, `outputs/full_model_coefficients_hazard.csv`

No new files in `src/`. No new test files (`tests/ablation_test.py` already exists from v1).

---

### Task 1: Impute `wc_ratio` NaN + add `wc_ratio_missing` flag

**Why this comes first:** the headline finding is conditional on a panel that excludes 4 of 5 REITs because `wc_ratio` is structurally NaN for REITs (unclassified balance sheets). Tasks 2–4 are all built on top of the panel as-is; fixing the panel first means every later result is computed on the corrected universe.

**Files:**
- Modify: `src/ews/panel.py:88-91` (the `dropna` block)
- Modify: `src/ews/config.py:71-79` (the `FEATURE_COLS` list)
- Modify: `tests/ablation_test.py` (add new test section [4])

- [ ] **Step 1: Write the failing test**

Append to `tests/ablation_test.py` immediately before the `print("\n" + "=" * 60)` summary line (i.e. after section [3]):

```python
print("\n[4] REITs back in panel; wc_ratio_missing column present")
REIT_TICKERS = ["SPG", "O", "PLD", "VTR"]
panel_tickers = set(panel["ticker"].unique())
reits_in_panel = [t for t in REIT_TICKERS if t in panel_tickers]
check(
    "at least 3 of 4 named REITs survive panel dropna",
    len(reits_in_panel) >= 3,
    f"REITs in panel: {reits_in_panel}",
)
check(
    "wc_ratio_missing column present in panel",
    "wc_ratio_missing" in panel.columns,
    f"columns include wc_ratio_missing: {'wc_ratio_missing' in panel.columns}",
)
if "wc_ratio_missing" in panel.columns:
    check(
        "wc_ratio_missing is binary (only 0/1 values)",
        set(panel["wc_ratio_missing"].unique()).issubset({0, 1}),
        f"unique values: {sorted(panel['wc_ratio_missing'].unique())}",
    )
    # For REIT rows, missingness should be 1 (they're the reason we added this feature)
    reit_rows = panel[panel["ticker"].isin(REIT_TICKERS)]
    if len(reit_rows) > 0:
        check(
            "REIT rows have wc_ratio_missing == 1",
            (reit_rows["wc_ratio_missing"] == 1).all(),
            f"REIT wc_ratio_missing values: {reit_rows['wc_ratio_missing'].unique()}",
        )
```

- [ ] **Step 2: Run the test to verify section [4] fails**

Run:

```bash
cd /Users/ivanchow/Documents/projects/hku-final
.venv/bin/python tests/ablation_test.py
```

Expected: sections [1]–[3] PASS; section [4] FAILs with `REITs in panel: []` (none of the 4 REITs are present) and `wc_ratio_missing` column absent. Exit code 1.

- [ ] **Step 3: Edit `src/ews/panel.py` to impute + add the flag**

Find this block in `src/ews/panel.py` (around lines 88–91):

```python
    # 6. Drop rows with any missing feature / label.
    before = len(panel)
    panel = panel.dropna(subset=FEATURE_COLS + [LABEL_COL])
    print(f"  Panel: {len(panel)} firm-months ({before - len(panel)} dropped for NaN)")
```

Replace with:

```python
    # 6. Impute structurally-undefined wc_ratio (REITs file unclassified balance
    #    sheets and don't report current assets/liabilities). We fill with 0.0
    #    as a neutral value and emit a binary missingness indicator so the
    #    model can learn "no working-capital signal" as its own feature.
    panel["wc_ratio_missing"] = panel["wc_ratio"].isna().astype(int)
    panel["wc_ratio"] = panel["wc_ratio"].fillna(0.0)

    # 7. Drop rows with any remaining missing feature / label.
    before = len(panel)
    panel = panel.dropna(subset=FEATURE_COLS + [LABEL_COL])
    print(f"  Panel: {len(panel)} firm-months ({before - len(panel)} dropped for NaN)")
```

- [ ] **Step 4: Edit `src/ews/config.py` to add `wc_ratio_missing` to `FEATURE_COLS`**

Find this in `src/ews/config.py:71-79`:

```python
FEATURE_COLS = [
    "leverage", "liquidity_buffer", "wc_ratio", "profitability",
    "ret_1m", "ret_3m", "ret_6m",
    "vol_3m", "vol_6m", "drawdown_12m",
    "late_filing",
    "vix", "term_spread", "credit_spread",
]
```

Replace with:

```python
FEATURE_COLS = [
    "leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability",
    "ret_1m", "ret_3m", "ret_6m",
    "vol_3m", "vol_6m", "drawdown_12m",
    "late_filing",
    "vix", "term_spread", "credit_spread",
]
```

- [ ] **Step 5: Update the `Accounting only` and `Acct + Market` subsets in `eval.py` to include `wc_ratio_missing`**

In `src/ews/eval.py::ablation_analysis`, find the `subsets` dict and replace the Accounting and Acct + Market entries:

Current:

```python
        "Accounting only":   ["leverage", "liquidity_buffer", "wc_ratio", "profitability"],
        ...
        "Acct + Market":     ["leverage", "liquidity_buffer", "wc_ratio", "profitability",
                              "ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
```

Replace with:

```python
        "Accounting only":   ["leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability"],
        ...
        "Acct + Market":     ["leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability",
                              "ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
```

(`Full model` uses `FEATURE_COLS` directly so it picks up the new feature automatically.)

- [ ] **Step 6: Re-run the pipeline so the panel CSV is refreshed**

```bash
MPLBACKEND=Agg .venv/bin/python src/run.py 2>&1 | grep -E "Panel:|Firms:" | head -5
```

Expected: the printed `Panel:` line shows ~12,100+ firm-months (was 11,496); `Firms: 76` (was 72). The 4 REITs and DE are now present.

- [ ] **Step 7: Run the test to verify it passes**

```bash
.venv/bin/python tests/ablation_test.py
```

Expected: every section `[PASS]`, exit 0. Section `[4]` shows `REITs in panel: ['SPG', 'O', 'PLD', 'VTR']` and the missingness column is binary with REIT rows == 1.

- [ ] **Step 8: Commit**

```bash
git add src/ews/panel.py src/ews/config.py src/ews/eval.py tests/ablation_test.py
git commit -m "panel: impute wc_ratio NaN + add wc_ratio_missing flag; REITs back in panel"
```

---

### Task 2: Firm-clustered bootstrap with 1000 resamples

**Why:** row-level bootstrap treats within-firm autocorrelation as independent. Credit-risk panels have strong within-firm temporal correlation (a firm in distress stays in distress for months), so row-level CIs underestimate uncertainty. Firm-clustered resampling — sample firms with replacement, take all their rows — is the standard fix. 1000 resamples cuts CI-bound variance from ~±0.03 (at 200) to ~±0.014, which is what publication-grade tables want.

**Files:**
- Modify: `src/ews/eval.py` — extend `_bootstrap_auroc_ci`, default `n_boot=1000`, plumb `firm_ids`
- Modify: `tests/ablation_test.py` — add CI-width sanity check + clustered-vs-row-level smoke

- [ ] **Step 1: Write the failing tests**

Append to `tests/ablation_test.py` before the summary line (after section [4]):

```python
print("\n[5] firm-clustered bootstrap widens CIs vs row-level")
# Quick property check: rerunning ablation with row-level bootstrap should give
# strictly narrower CIs than the production (clustered) run, because clustered
# resampling correctly accounts for within-firm autocorrelation.
from ews.eval import ablation_analysis, _bootstrap_auroc_ci  # noqa: E402

# The production call (test passes ticker col via test data → clustered)
rdf_clustered = rdf  # alias the section [2] result for clarity
clustered_widths = (rdf_clustered["AUROC_hi"] - rdf_clustered["AUROC_lo"]).values

# Reference: row-level CIs on the same subsets via a small direct invocation
# of the helper. We only need one subset for the smoke check — Market only.
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score  # noqa: E402

market_cols = ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"]
m = sm.Logit(train["label_a"], sm.add_constant(train[market_cols])).fit(disp=0)
p = m.predict(sm.add_constant(test[market_cols]))
y_true = test["label_a"].values
y_pred = p.values
lo_row, hi_row = _bootstrap_auroc_ci(y_true, y_pred, firm_ids=None, n_boot=1000)
row_level_width = hi_row - lo_row

market_row = rdf_clustered[rdf_clustered["Feature set"] == "Market only"].iloc[0]
clustered_width = market_row["AUROC_hi"] - market_row["AUROC_lo"]

check(
    "Market only clustered CI is at least as wide as row-level CI",
    clustered_width >= row_level_width * 0.9,  # 0.9 buffer for monte-carlo noise
    f"clustered={clustered_width:.4f} vs row-level={row_level_width:.4f}",
)
check(
    "every clustered CI width < 1.0 (CI is actually informative)",
    bool((clustered_widths < 1.0).all()),
    f"widths: {clustered_widths.round(4).tolist()}",
)
```

- [ ] **Step 2: Run the test to verify the new section fails**

```bash
.venv/bin/python tests/ablation_test.py
```

Expected: sections [1]–[4] PASS; section [5] FAILs because (a) `_bootstrap_auroc_ci` doesn't accept a `firm_ids` keyword and (b) the existing CIs are row-level. The traceback will say `unexpected keyword argument 'firm_ids'` or the CI-width comparison fails. Exit 1.

- [ ] **Step 3: Extend `_bootstrap_auroc_ci` in `src/ews/eval.py`**

Replace the existing `_bootstrap_auroc_ci` function with this version (keeping the same module-level position between `evaluate_model` and `compute_lead_time`):

```python
def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    firm_ids: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC.

    If `firm_ids` is provided, performs *firm-clustered* bootstrap: each
    resample draws firm IDs with replacement and includes all rows for each
    drawn firm. This correctly accounts for within-firm autocorrelation that
    row-level bootstrap ignores (a major issue in credit-risk panels where
    consecutive firm-months are highly correlated).

    If `firm_ids` is None, falls back to row-level bootstrap (resample row
    indices with replacement).

    Resamples that yield single-class y_true are skipped (AUROC undefined).
    Returns (nan, nan) if no resample is usable.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if firm_ids is not None:
        firm_ids = np.asarray(firm_ids)
        unique_firms = np.unique(firm_ids)
        # Pre-index rows-per-firm once for speed
        firm_to_rows = {f: np.where(firm_ids == f)[0] for f in unique_firms}
        aurocs: list[float] = []
        for _ in range(n_boot):
            sampled_firms = rng.choice(unique_firms, size=len(unique_firms), replace=True)
            idx = np.concatenate([firm_to_rows[f] for f in sampled_firms])
            if len(np.unique(y_true[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    else:
        aurocs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))

    if not aurocs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aurocs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
```

- [ ] **Step 4: Plumb `firm_ids` through `ablation_analysis`**

Inside the existing `for name, cols in subsets.items()` loop in `ablation_analysis`, find the block that calls `_bootstrap_auroc_ci(y_true, y_pred)` and replace with:

```python
            y_true = test[LABEL_COL].values
            y_pred = p.values if hasattr(p, "values") else np.asarray(p)
            firm_ids = test["ticker"].values
            auroc = roc_auc_score(y_true, y_pred)
            lo, hi = _bootstrap_auroc_ci(y_true, y_pred, firm_ids=firm_ids)
            results.append({
                "Feature set": name,
                "N": len(cols),
                "AUROC": auroc,
                "AUROC_lo": lo,
                "AUROC_hi": hi,
                "AUPRC": average_precision_score(y_true, y_pred),
                "Brier": brier_score_loss(y_true, y_pred),
            })
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python tests/ablation_test.py
```

Expected: every section `[PASS]`, exit 0. Section `[5]` confirms the production (clustered) CI for Market only is at least as wide as a row-level CI on the same data.

- [ ] **Step 6: Commit**

```bash
git add src/ews/eval.py tests/ablation_test.py
git commit -m "eval: firm-clustered bootstrap, n_boot=1000 — proper SE for credit-risk panels"
```

---

### Task 3: Run ablation across all three model families

**Why:** the v1 finding is "Market beats Full *in pooled logit*." The proposal commits to interpretable regression with FE and hazard variants delivered. If the headline holds across all three model families, the claim becomes "Market features dominate regardless of model family," which is structurally stronger. If it doesn't hold (e.g., FE flips the ordering because firm-level intercepts absorb some of what looked like Macro noise), that itself is a sharper finding.

**Files:**
- Modify: `src/ews/eval.py` — add per-family fit wrappers + outer loop
- Modify: `tests/ablation_test.py` — assert `model_family` column + minimum row count

- [ ] **Step 1: Write the failing test**

Append to `tests/ablation_test.py` before the summary line (after section [5]):

```python
print("\n[6] ablation runs over multiple model families")
check(
    "model_family column present",
    "model_family" in rdf.columns,
    f"columns: {list(rdf.columns)}",
)
if "model_family" in rdf.columns:
    families = set(rdf["model_family"].unique())
    check(
        "at least pooled + fe families ran successfully",
        {"pooled", "fe"}.issubset(families),
        f"families: {sorted(families)}",
    )
    # Each successful family should produce all 6 subsets (or fail-and-omit
    # individually; the row count per family must be > 1 for the data to mean
    # anything).
    for fam in families:
        n_subsets = (rdf["model_family"] == fam).sum()
        check(
            f"{fam} family produced multiple subset rows (got {n_subsets})",
            n_subsets >= 4,
            f"{fam}: {n_subsets} rows",
        )
```

- [ ] **Step 2: Run the test to verify the new section fails**

```bash
.venv/bin/python tests/ablation_test.py
```

Expected: sections [1]–[5] PASS; section [6] FAILs because `model_family` column doesn't exist. Exit 1.

- [ ] **Step 3: Add per-family fit wrappers in `src/ews/eval.py`**

Insert these helpers immediately after `_bootstrap_auroc_ci` (before `compute_lead_time`):

```python
# =============================================================================
# Per-family fit wrappers (used by ablation_analysis to loop over model types)
# =============================================================================

def _fit_pooled(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Plain pooled logit: y ~ const + cols. Returns predicted probabilities on test."""
    m = sm.Logit(train[LABEL_COL], sm.add_constant(train[cols])).fit(disp=0)
    return m.predict(sm.add_constant(test[cols]))


def _fit_fe(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Pooled logit with industry + year dummies (lightweight FE for ablation)."""
    train_fe = pd.concat([
        train[cols].reset_index(drop=True),
        pd.get_dummies(train["industry"], prefix="ind", drop_first=True).astype(float).reset_index(drop=True),
        pd.get_dummies(train["year"], prefix="yr", drop_first=True).astype(float).reset_index(drop=True),
    ], axis=1)
    test_fe = pd.concat([
        test[cols].reset_index(drop=True),
        pd.get_dummies(test["industry"], prefix="ind", drop_first=True).astype(float).reset_index(drop=True),
        pd.get_dummies(test["year"], prefix="yr", drop_first=True).astype(float).reset_index(drop=True),
    ], axis=1)
    # Align test columns to train (dummies may differ if a year/industry appears
    # in only one split)
    test_fe = test_fe.reindex(columns=train_fe.columns, fill_value=0.0)
    m = sm.Logit(train[LABEL_COL].reset_index(drop=True), sm.add_constant(train_fe)).fit(disp=0)
    return m.predict(sm.add_constant(test_fe))


def _fit_hazard(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Discrete-time hazard logit: pooled logit + log-duration baseline.

    Duration = months since first observation per firm. Approximates the
    Shumway (2001) form without the full hazard panel restructure.
    """
    def with_log_duration(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["ticker", "date"])
        df["months_obs"] = df.groupby("ticker").cumcount() + 1
        df["log_duration"] = np.log(df["months_obs"])
        return df

    train_h = with_log_duration(train)
    test_h = with_log_duration(test)
    X_train = sm.add_constant(train_h[cols + ["log_duration"]])
    X_test = sm.add_constant(test_h[cols + ["log_duration"]])
    m = sm.Logit(train_h[LABEL_COL], X_train).fit(disp=0)
    # Predictions must align back to the original test row order so the bootstrap
    # firm_ids index matches.
    preds = m.predict(X_test)
    preds.index = test_h.index
    return preds.reindex(test.index)


MODEL_FAMILIES: dict[str, callable] = {
    "pooled": _fit_pooled,
    "fe":     _fit_fe,
    "hazard": _fit_hazard,
}
```

- [ ] **Step 4: Refactor `ablation_analysis` to loop over families**

Replace the body of `ablation_analysis` (everything from the `print("\n" + "=" * 70)` after the function signature down to and including the `return rdf` at the end) with:

```python
    print("\n" + "=" * 70)
    print("ABLATION: FEATURE GROUP COMPARISON")
    print("=" * 70)

    subsets = {
        "Accounting only":   ["leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability"],
        "Market only":       ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Macro only":        ["vix", "term_spread", "credit_spread"],
        "Filing only":       ["late_filing"],
        "Acct + Market":     ["leverage", "liquidity_buffer", "wc_ratio", "wc_ratio_missing", "profitability",
                              "ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Full model":        FEATURE_COLS,
        "Altman Z-score":    ["z_score"],
    }

    results = []
    firm_ids = test["ticker"].values
    y_true = test[LABEL_COL].values

    for family_name, fit_fn in MODEL_FAMILIES.items():
        print(f"\n-- {family_name} --")
        for subset_name, cols in subsets.items():
            try:
                p = fit_fn(train, test, cols)
                y_pred = p.values if hasattr(p, "values") else np.asarray(p)
                auroc = roc_auc_score(y_true, y_pred)
                lo, hi = _bootstrap_auroc_ci(y_true, y_pred, firm_ids=firm_ids)
                results.append({
                    "model_family": family_name,
                    "Feature set": subset_name,
                    "N": len(cols),
                    "AUROC": auroc,
                    "AUROC_lo": lo,
                    "AUROC_hi": hi,
                    "AUPRC": average_precision_score(y_true, y_pred),
                    "Brier": brier_score_loss(y_true, y_pred),
                })
            except Exception as e:
                print(f"  {subset_name}: failed ({type(e).__name__}: {str(e)[:80]})")

    rdf = pd.DataFrame(results)
    print("\n" + rdf.round(4).to_string(index=False))

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    out_path = os.path.join(PATHS.OUTPUTS, "ablation_results.csv")
    rdf.to_csv(out_path, index=False)
    print(f"\n  Saved ablation results to: {out_path}")

    return rdf
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python tests/ablation_test.py
```

Expected: every section `[PASS]`, exit 0. The printed ablation table now has a `model_family` column with at minimum `pooled` and `fe` (and ideally `hazard`) groups; each family contributes ≥ 4 subsets.

If `hazard` family fails on every subset (singular-matrix errors are possible on the larger panel), the test only requires `pooled` and `fe`, so it still passes — that's an acceptable degradation noted in the findings doc.

- [ ] **Step 6: Commit**

```bash
git add src/ews/eval.py tests/ablation_test.py
git commit -m "eval: ablate across pooled / FE / hazard families; CSV gets model_family column"
```

---

### Task 4: Persist Full-model coefficient tables (per family)

**Why:** the proposal's headline selling point is "interpretable regression whose coefficients can be read and argued with." We've been running these models for two phases without ever inspecting their coefficients. Persisting them gives the report a mechanism explanation for *why* the Full model loses to Market alone (most likely: at least one macro feature carries a wrong-signed coefficient relative to credit-risk theory). One artifact per model family.

**Files:**
- Modify: `src/ews/eval.py` — add `persist_full_model_coefficients` helper
- Modify: `src/ews/pipeline.py` — call the helper after the existing main-model fits
- Modify: `tests/ablation_test.py` — assert the coefficient CSV exists with the right columns

- [ ] **Step 1: Write the failing test**

Append to `tests/ablation_test.py` before the summary line (after section [6]):

```python
print("\n[7] full-model coefficient sidecars persisted")
coef_pooled = os.path.join(PATHS.OUTPUTS, "full_model_coefficients_pooled.csv")
check(
    "full_model_coefficients_pooled.csv exists",
    os.path.isfile(coef_pooled),
    coef_pooled,
)
if os.path.isfile(coef_pooled):
    coef_df = pd.read_csv(coef_pooled)
    expected = {"feature", "coef", "std_err", "p_value"}
    check(
        "coefficient CSV has required columns",
        expected.issubset(set(coef_df.columns)),
        f"got {sorted(coef_df.columns)}",
    )
    check(
        "coefficient CSV has one row per feature (≥ 14 features + const)",
        len(coef_df) >= 15,
        f"rows={len(coef_df)}",
    )
```

- [ ] **Step 2: Run the test to verify section [7] fails**

```bash
.venv/bin/python tests/ablation_test.py
```

Expected: section [7] FAILs because the coefficient sidecar doesn't exist yet. Exit 1.

- [ ] **Step 3: Add `persist_full_model_coefficients` in `src/ews/eval.py`**

Insert after `MODEL_FAMILIES` (or wherever fits the surrounding structure best — near the other persist logic):

```python
def persist_full_model_coefficients(
    train: pd.DataFrame,
    family_name: str,
    fit_fn: callable | None = None,
) -> str:
    """Fit the Full model with the given family and write its coefficient table
    to `outputs/full_model_coefficients_<family>.csv`.

    Persists: feature, coef, std_err, p_value.
    Returns the absolute output path written.
    """
    fit_fn = fit_fn or MODEL_FAMILIES[family_name]

    # We need the fitted statsmodels Results object, not just predictions,
    # to read coefficients. Inline the fit here so we can keep the wrappers
    # in MODEL_FAMILIES focused on predictions.
    cols = FEATURE_COLS
    if family_name == "pooled":
        m = sm.Logit(train[LABEL_COL], sm.add_constant(train[cols])).fit(disp=0)
        names = ["const"] + cols
    elif family_name == "fe":
        train_fe = pd.concat([
            train[cols].reset_index(drop=True),
            pd.get_dummies(train["industry"], prefix="ind", drop_first=True).astype(float).reset_index(drop=True),
            pd.get_dummies(train["year"], prefix="yr", drop_first=True).astype(float).reset_index(drop=True),
        ], axis=1)
        m = sm.Logit(train[LABEL_COL].reset_index(drop=True), sm.add_constant(train_fe)).fit(disp=0)
        names = ["const"] + list(train_fe.columns)
    elif family_name == "hazard":
        tr = train.copy().sort_values(["ticker", "date"])
        tr["months_obs"] = tr.groupby("ticker").cumcount() + 1
        tr["log_duration"] = np.log(tr["months_obs"])
        X = sm.add_constant(tr[cols + ["log_duration"]])
        m = sm.Logit(tr[LABEL_COL], X).fit(disp=0)
        names = ["const"] + cols + ["log_duration"]
    else:
        raise ValueError(f"unknown family: {family_name}")

    coef_df = pd.DataFrame({
        "feature": names,
        "coef": m.params.values,
        "std_err": m.bse.values,
        "p_value": m.pvalues.values,
    })

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    out_path = os.path.join(PATHS.OUTPUTS, f"full_model_coefficients_{family_name}.csv")
    coef_df.to_csv(out_path, index=False)
    print(f"  Saved {family_name} full-model coefficients to: {out_path}")
    return out_path
```

- [ ] **Step 4: Call the helper from `src/ews/pipeline.py`**

In `src/ews/pipeline.py`, find the existing ablation call site (around line 161):

```python
    try:
        ablation_analysis(train, eval_data)
    except Exception as e:
        _warn(f"Ablation failed: {e}")
```

Insert *before* the `try` block:

```python
    from .eval import persist_full_model_coefficients  # local import to keep top clean

    for family in ("pooled", "fe", "hazard"):
        try:
            persist_full_model_coefficients(train, family)
        except Exception as e:
            _warn(f"Coefficient persist for {family} failed: {e}")
```

- [ ] **Step 5: Re-run the pipeline**

```bash
MPLBACKEND=Agg .venv/bin/python src/run.py 2>&1 | grep "Saved" | head -10
```

Expected: lines including `Saved pooled full-model coefficients to: …/outputs/full_model_coefficients_pooled.csv` and the same for `fe` and `hazard` (hazard may fail — that's acceptable, logged as a warning).

- [ ] **Step 6: Run the test to verify it passes**

```bash
.venv/bin/python tests/ablation_test.py
```

Expected: every section `[PASS]`, exit 0. Section `[7]` confirms the pooled coefficient sidecar exists with the right columns.

- [ ] **Step 7: Gitignore the new generated artifacts**

Append to `.gitignore`:

```
outputs/full_model_coefficients_*.csv
```

- [ ] **Step 8: Commit**

```bash
git add src/ews/eval.py src/ews/pipeline.py tests/ablation_test.py .gitignore
git commit -m "eval: persist full-model coefficient tables per family for mechanistic interpretation"
```

---

### Task 5: Re-run + rewrite `outputs/ablation_findings.md` as v2

**Why:** v1's findings doc reports numbers that are now obsolete (panel had 72 firms; v2 has 76; CIs were row-level at n=200; v2 are firm-clustered at n=1000; v1 had pooled only; v2 has up to three families; v1 had no coefficient inspection). The doc must be rewritten so report-quotable claims reflect the v2 numbers.

**Files:**
- Create/overwrite: `outputs/ablation_findings.md`

- [ ] **Step 1: Run the full pipeline + capture the validation split size**

```bash
cd /Users/ivanchow/Documents/projects/hku-final
MPLBACKEND=Agg .venv/bin/python src/run.py 2>&1 | tee /tmp/v2_run.log
```

Expected: exit 0, "PHASE 1 PROTOTYPE COMPLETE". Note the `Val: N rows` line in the time-split section; you'll cite that in Limitations.

- [ ] **Step 2: Inspect the v2 ablation table and pooled coefficients**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('outputs/ablation_results.csv')
print(df.round(4).to_string(index=False))
print()
print('--- pooled coefficients (Full model) ---')
print(pd.read_csv('outputs/full_model_coefficients_pooled.csv').round(4).to_string(index=False))
"
```

Note the rows where coefficient signs disagree with credit-risk theory:
- `leverage` should be **positive** (more debt = more risk)
- `liquidity_buffer` should be **negative** (more cash = less risk)
- `profitability` should be **negative** (more profit = less risk)
- `drawdown_12m` should be **positive** (recent stress = future stress)
- `vix` should be **positive** (high VIX = stress regime)
- `credit_spread` should be **positive** (wide spreads = stress)
- `term_spread` is ambiguous (inversion can predict recession)

Any coefficient with a wrong sign is a story for the findings doc.

- [ ] **Step 3: Overwrite `outputs/ablation_findings.md` with the v2 template**

Replace the entire contents of `outputs/ablation_findings.md` with:

```markdown
# Feature Group Ablation — Phase 2 Findings (v2)

**Date:** 2026-06-05
**Panel:** `data/processed/panel_phase2.csv` (76 firms, ~12,100 firm-months, with REITs imputed via `wc_ratio_missing` flag)
**Eval split:** validation (2021–2023, <N> rows from pipeline output)
**Models:** pooled logit, fixed-effects logit (industry + year dummies), discrete-time hazard logit (Shumway-style)
**Uncertainty:** 95% percentile bootstrap, **firm-clustered**, 1,000 resamples
**Note on v1:** This supersedes the v1 findings (row-level bootstrap, 72-firm panel, pooled-only). See git history `v1...v2` for diff.

## What changed since v1

- Panel grew from 72 to 76 firms after imputing `wc_ratio` for REITs (4 firms recovered: SPG, O, PLD, VTR; DE also returned).
- Bootstrap is now firm-clustered at n=1,000 (was row-level at n=200) — CIs are typically wider but honestly reflect within-firm autocorrelation.
- Ablation now runs over all three model families.
- Full-model coefficients persisted per family for mechanistic inspection.

## Result table (all families)

(paste the v2 CSV here as a markdown table — sorted by model_family then AUROC desc)

## Headline finding

<one or two sentences. The defensible claim depends on whether Market-alone-dominates survives both the REIT-included panel and all model families. Write it sharp: e.g. "Across all three model families, Market features alone match or beat the Full model on firm-clustered CIs; the lower CI bound of Market-only AUROC exceeds the upper CI bound of Full-model AUROC for {pooled / fe / hazard}, meaning adding accounting, macro, and filing features measurably hurts discrimination regardless of the model used.">

If the headline does *not* hold across families, write the truth: "Market dominance holds for pooled but inverts under FE; the firm fixed-effects absorb signal that pooled logit attributes to Market features." (Or whatever the actual pattern is.)

## What carries the signal (per family)

For each family present in the CSV, one bulleted paragraph naming the leader, the runner-up, and whether their CIs overlap. Reference the exact numbers.

## Why the Full model loses (mechanism)

Read `outputs/full_model_coefficients_pooled.csv`. Identify any coefficient whose **sign** disagrees with credit-risk theory (see Task 5 Step 2 list above). State which features and what sign you'd expect. This is the most defensible answer to "why does adding more features hurt?" — at least one feature is fitted with the wrong polarity.

Example phrasing if `term_spread` came back negative:
*"In the pooled Full model, `term_spread` has coefficient -1.236 (SE 0.171, p < 0.001). Credit-risk theory predicts that an *inverted* yield curve (negative term spread) precedes distress, so a *positive* coefficient on `term_spread` is what theory predicts in the deterioration-prediction direction. The fitted negative coefficient suggests the model is fitting a level relationship (low rates → easy credit → fewer defaults) rather than the inversion signal — likely an artefact of the 2010–2020 zero-rate training regime."*

## Limitations

- Hazard family <succeeded / partially failed> on this panel. <If failed:> singular-matrix errors on small subsets are documented in the pipeline log; FE family's results are reported as the more robust supplement.
- 1,000 firm-clustered resamples on 76 firms ≈ effective sample size of ~76 unique firms regardless of row count. CI widths are larger than v1's row-level CIs and are the honest measure for cross-firm generalization.
- All families use the same time split (train ≤ 2020 / val 2021–2023). Walk-forward CV across multiple test years is queued for Tier 2.
- `Altman Z-score` subset may still fail (the `z_score` column has NaN/inf for some rows). Treat as "not reportable on this panel."

## Implications for the rest of Phase 3

- **Item #4 (horizon analysis):** run on the leading subset identified above, in the leading model family.
- **Item #5 (threshold sensitivity):** rebuild the labels at 30% / 50% drawdown and rerun this ablation table to confirm rankings are not threshold-specific.
- **Item #7 (error analysis):** slice prediction errors of the leading model by sector, with the REIT firms now present.
- **Item #8 (calibration):** Platt/isotonic calibration applied to the leading family's leading subset. The coefficient inspection in `outputs/full_model_coefficients_*.csv` may inform whether to drop a wrong-signed feature before calibration.

## Reproducibility

- Pipeline command: `MPLBACKEND=Agg python src/run.py`
- Bootstrap seed: 42 (hardcoded in `_bootstrap_auroc_ci`)
- Source CSV: `outputs/ablation_results.csv` (regenerated every run)
- Coefficient sidecars: `outputs/full_model_coefficients_{pooled,fe,hazard}.csv`
- Test script: `tests/ablation_test.py` (7 sections, ~20 assertions)
```

When filling the `<…>` slots, **every number you type must come from the CSV or the coefficient files** (round to 3 decimals). No invented coefficients, no eyeballed p-values, no informal stats language.

- [ ] **Step 4: Read the final doc end-to-end + cross-check**

```bash
cat outputs/ablation_findings.md
.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('outputs/ablation_results.csv')
print(d.round(3).to_string(index=False))
"
```

Confirm every number in the markdown appears in the CSV (rounded). If any number is in the markdown but not the CSV, delete it.

- [ ] **Step 5: Commit**

```bash
git add outputs/ablation_findings.md
git commit -m "report: ablation findings v2 — REIT-included panel, firm-clustered CIs, 3 model families"
```

---

## Self-Review

**1. Spec coverage:** Tier 1 had four items (A–D from the planning conversation). Mapping:
- A (REIT/wc_ratio fix + rerun) → Task 1 + the rerun in Task 5
- B (firm-clustered bootstrap, n=1000) → Task 2
- C (loop over pooled / FE / hazard) → Task 3
- D (coefficient inspection) → Task 4
- (Findings doc rewrite is Task 5, not in original A–D but required to consume the new artifacts.)

**2. Placeholder scan:** All `<…>` markers in the findings template are explicit human-input slots, not placeholders for the engineer — they're labeled "fill in from numbers in the CSV". Every Task 1–4 step has complete code, no TODOs.

**3. Type / signature consistency:**
- `_bootstrap_auroc_ci(y_true, y_pred, firm_ids=None, n_boot=1000, alpha=0.05, seed=42)` — used identically across Tasks 2 and 3.
- `MODEL_FAMILIES: dict[str, callable]` defined in Task 3 — keys (`"pooled"`, `"fe"`, `"hazard"`) reused in Task 4 (`persist_full_model_coefficients`) and pipeline call site.
- CSV schema: `model_family, Feature set, N, AUROC, AUROC_lo, AUROC_hi, AUPRC, Brier` — same across Tasks 3, 4, 5.
- `wc_ratio_missing` feature — added to `FEATURE_COLS` in Task 1, picked up automatically in Task 3's `Full model` subset (which uses `FEATURE_COLS`).
- Coefficient CSV schema: `feature, coef, std_err, p_value` — same in Task 4 and Task 5's interpretation step.

No inconsistencies found.

**4. Dependency order:**
- Task 1 (REIT fix) must precede Task 5 (numbers in findings).
- Task 2 (firm-clustered bootstrap) must precede Task 3 (which uses the new helper signature).
- Task 3 (model family loop) must precede Task 4 (which uses `MODEL_FAMILIES`).
- Task 5 must be last (rewrites findings from final pipeline state).

Order is correct as written.
