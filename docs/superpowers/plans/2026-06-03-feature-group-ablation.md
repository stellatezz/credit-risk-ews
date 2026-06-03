# Feature Group Ablation (Phase 3 Item #2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `ablation_analysis` so the Phase 3 report can answer "which signals actually matter?" defensibly: add the missing `Filing only` group, attach 95% bootstrap confidence intervals on test-set AUROC so close differences aren't over-claimed, persist results to `outputs/ablation_results.csv`, and write a one-page findings markdown summarising the Phase 2 result.

**Architecture:** All code changes are localised to one function (`src/ews/eval.py::ablation_analysis`) and one constant (`PATHS.OUTPUTS` in `src/ews/config.py`). We do not change the pipeline orchestration — `pipeline.py:161` already invokes the function each run. A new tiny helper `_bootstrap_auroc_ci` lives next to `evaluate_model` in `eval.py`. Tests follow the existing `tests/smoke_test.py` runnable-script convention (no pytest dependency added). Final interpretation is hand-written after running the pipeline and reading the resulting CSV.

**Tech Stack:** pandas, numpy, statsmodels.Logit, sklearn.metrics (all already pinned in `requirements.txt` and installed in `.venv`).

---

### File Structure

- **Modify:** `src/ews/config.py` — add one constant: `PATHS.OUTPUTS = <repo>/outputs`
- **Modify:** `src/ews/eval.py` — extend `ablation_analysis`, add `_bootstrap_auroc_ci` helper, add `os`/`PATHS` imports
- **Create:** `tests/ablation_test.py` — runnable assertion script in the style of `tests/smoke_test.py`
- **Generated:** `outputs/ablation_results.csv` (written by `ablation_analysis`)
- **Hand-written:** `outputs/ablation_findings.md` (Task 4, after seeing the numbers)

No new dependencies. No pipeline changes.

---

### Task 1: Add `PATHS.OUTPUTS` constant

**Files:**
- Modify: `src/ews/config.py:20-24`

- [ ] **Step 1: Read the current PATHS class**

Open `src/ews/config.py` and locate lines 20-24:

```python
class PATHS:
    RAW = os.path.join(DATA_DIR, "raw")
    INTERIM = os.path.join(DATA_DIR, "interim")
    PROCESSED = os.path.join(DATA_DIR, "processed")
    FIGURES = os.path.join(REPO_ROOT, "outputs", "figures")
```

- [ ] **Step 2: Replace with the version including `OUTPUTS`**

Edit `src/ews/config.py` so the PATHS class reads:

```python
class PATHS:
    RAW = os.path.join(DATA_DIR, "raw")
    INTERIM = os.path.join(DATA_DIR, "interim")
    PROCESSED = os.path.join(DATA_DIR, "processed")
    OUTPUTS = os.path.join(REPO_ROOT, "outputs")
    FIGURES = os.path.join(OUTPUTS, "figures")
```

Note: `FIGURES` is rewritten to derive from `OUTPUTS` so the two stay consistent.

- [ ] **Step 3: Verify import + value**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c "from ews.config import PATHS; print(PATHS.OUTPUTS); print(PATHS.FIGURES)"
```

Expected output (paths from your repo root):

```
/Users/ivanchow/Documents/projects/hku-final/outputs
/Users/ivanchow/Documents/projects/hku-final/outputs/figures
```

- [ ] **Step 4: Commit**

```bash
git add src/ews/config.py
git commit -m "config: add PATHS.OUTPUTS so eval/diagnostics can persist alongside figures"
```

---

### Task 2: Add `Filing only` group and persist results CSV

**Files:**
- Modify: `src/ews/eval.py:110-148` (the `ablation_analysis` function and its imports)
- Create: `tests/ablation_test.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ablation_test.py`:

```python
"""Ablation extension test — same runnable-script style as tests/smoke_test.py.

Run from repo root:
    python tests/ablation_test.py

Exits 0 on success, nonzero with a clear message on any failure.
Requires data/processed/panel_phase2.csv to exist (run pipeline first).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ews.config import PATHS  # noqa: E402
from ews.eval import ablation_analysis  # noqa: E402


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# Load Phase 2 panel + simple time split (matches pipeline's TRAIN_END_YEAR=2020)
panel_path = os.path.join(PATHS.PROCESSED, "panel_phase2.csv")
assert os.path.isfile(panel_path), f"missing panel: {panel_path} — run pipeline first"
panel = pd.read_csv(panel_path)
train = panel[panel["year"] <= 2020]
test = panel[panel["year"] > 2020]

print("\n[1] ablation_analysis produces Filing only row")
rdf = ablation_analysis(train, test)
check(
    "Filing only group present",
    "Filing only" in rdf["Feature set"].values,
    f"feature sets: {list(rdf['Feature set'])}",
)
check(
    "Filing only uses exactly 1 feature",
    int(rdf.loc[rdf["Feature set"] == "Filing only", "N"].iloc[0]) == 1
    if "Filing only" in rdf["Feature set"].values else False,
)

print("\n[2] CSV persisted to outputs/")
csv_path = os.path.join(PATHS.OUTPUTS, "ablation_results.csv")
check("outputs/ablation_results.csv exists", os.path.isfile(csv_path), csv_path)
if os.path.isfile(csv_path):
    persisted = pd.read_csv(csv_path)
    check(
        "CSV row count matches in-memory result",
        len(persisted) == len(rdf),
        f"csv={len(persisted)} vs memory={len(rdf)}",
    )

print("\n" + "=" * 60)
if FAILURES:
    print(f"ABLATION TEST FAILED — {len(FAILURES)} assertion(s) failed:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ABLATION TEST PASSED")
    sys.exit(0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /Users/ivanchow/Documents/projects/hku-final
.venv/bin/python tests/ablation_test.py
```

Expected: FAIL with `Filing only group present: feature sets: [...no 'Filing only'...]` and `outputs/ablation_results.csv exists: ...` failing. Exit code 1.

- [ ] **Step 3: Add imports + `Filing only` + CSV persist in `eval.py`**

In `src/ews/eval.py`, at the top of the imports block (after the existing `import numpy as np` line), add:

```python
import os
```

And in the `from .config import ...` line, add `PATHS`:

```python
from .config import FEATURE_COLS, FIRMS, LABEL_COL, LEAD_TIME_THRESHOLD, PATHS, TOP_K_FRACTION
```

Then replace the `subsets` dict inside `ablation_analysis` (currently at `eval.py:117-123`):

```python
    subsets = {
        "Accounting only":   ["leverage", "liquidity_buffer", "wc_ratio", "profitability"],
        "Market only":       ["ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Macro only":        ["vix", "term_spread", "credit_spread"],
        "Filing only":       ["late_filing"],
        "Acct + Market":     ["leverage", "liquidity_buffer", "wc_ratio", "profitability",
                              "ret_1m", "ret_3m", "ret_6m", "vol_3m", "vol_6m", "drawdown_12m"],
        "Full model":        FEATURE_COLS,
        "Altman Z-score":    ["z_score"],
    }
```

Finally, replace the tail of the function (the last three lines that print and return `rdf`) with:

```python
    rdf = pd.DataFrame(results)
    print("\n" + rdf.round(4).to_string(index=False))

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    out_path = os.path.join(PATHS.OUTPUTS, "ablation_results.csv")
    rdf.to_csv(out_path, index=False)
    print(f"\n  Saved ablation results to: outputs/ablation_results.csv")

    return rdf
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd /Users/ivanchow/Documents/projects/hku-final
.venv/bin/python tests/ablation_test.py
```

Expected: every line shows `[PASS]` and the script exits 0 with `ABLATION TEST PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/ews/eval.py tests/ablation_test.py
git commit -m "eval: add Filing-only ablation group; persist results to outputs/ablation_results.csv"
```

---

### Task 3: Add 95% bootstrap CIs on AUROC

Adds `AUROC_lo` and `AUROC_hi` columns so the report can claim "Market beats Accounting (Δ=0.04, CIs disjoint)" rather than eyeballing point estimates.

**Files:**
- Modify: `src/ews/eval.py` — add `_bootstrap_auroc_ci`, extend `ablation_analysis` result rows
- Modify: `tests/ablation_test.py` — extend with CI assertions

- [ ] **Step 1: Extend the test with CI assertions**

Append to `tests/ablation_test.py` (immediately before the `print("\n" + "=" * 60)` summary line):

```python
print("\n[3] bootstrap CI columns present and well-formed")
check(
    "AUROC_lo column present",
    "AUROC_lo" in rdf.columns,
    f"columns: {list(rdf.columns)}",
)
check(
    "AUROC_hi column present",
    "AUROC_hi" in rdf.columns,
    f"columns: {list(rdf.columns)}",
)
if {"AUROC_lo", "AUROC_hi"}.issubset(rdf.columns):
    bracketed = ((rdf["AUROC_lo"] <= rdf["AUROC"]) & (rdf["AUROC"] <= rdf["AUROC_hi"])).all()
    check("each AUROC sits inside its own CI", bool(bracketed),
          rdf[["Feature set", "AUROC_lo", "AUROC", "AUROC_hi"]].to_string(index=False))
    nontrivial = ((rdf["AUROC_hi"] - rdf["AUROC_lo"]) > 0).all()
    check("CI widths are nonzero", bool(nontrivial))
```

- [ ] **Step 2: Run the test to verify the new assertions fail**

Run:

```bash
cd /Users/ivanchow/Documents/projects/hku-final
.venv/bin/python tests/ablation_test.py
```

Expected: section `[1]` and `[2]` still PASS; section `[3]` FAILs with `AUROC_lo column present` / `AUROC_hi column present` showing the column missing. Exit code 1.

- [ ] **Step 3: Add `_bootstrap_auroc_ci` helper**

In `src/ews/eval.py`, insert this helper immediately after the existing `evaluate_model` function (before the `compute_lead_time` definition):

```python
# =============================================================================
# Bootstrap CI helper (used by ablation_analysis)
# =============================================================================

def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC.

    Resamples rows with replacement `n_boot` times, computes AUROC per resample,
    returns the (alpha/2, 1-alpha/2) percentiles. Resamples that draw only one
    class are skipped (AUROC undefined). Returns (nan, nan) if no resample is
    usable.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    aurocs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    if not aurocs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aurocs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
```

- [ ] **Step 4: Use the helper inside `ablation_analysis`**

Inside the existing `for name, cols in subsets.items()` loop in `ablation_analysis`, replace the `results.append({...})` block with:

```python
            y_true = test[LABEL_COL].values
            y_pred = p.values if hasattr(p, "values") else np.asarray(p)
            auroc = roc_auc_score(y_true, y_pred)
            lo, hi = _bootstrap_auroc_ci(y_true, y_pred)
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

Run:

```bash
cd /Users/ivanchow/Documents/projects/hku-final
.venv/bin/python tests/ablation_test.py
```

Expected: every check `[PASS]`, exit 0. The `[3]` section now prints a CI-bracket table showing each row's AUROC sitting inside `[AUROC_lo, AUROC_hi]`.

- [ ] **Step 6: Commit**

```bash
git add src/ews/eval.py tests/ablation_test.py
git commit -m "eval: 95% bootstrap CIs on ablation AUROC (200 resamples)"
```

---

### Task 4: Run on Phase 2 panel and write the findings doc

This task is interpretation, not code. Run the pipeline (or the test), read the resulting `outputs/ablation_results.csv`, write a one-page summary the report can quote.

**Files:**
- Create: `outputs/ablation_findings.md`

- [ ] **Step 1: Run the pipeline so the table reflects the live Phase 2 panel**

Run:

```bash
cd /Users/ivanchow/Documents/projects/hku-final
MPLBACKEND=Agg .venv/bin/python src/run.py 2>&1 | tee /tmp/ablation_run.log
```

Expected: pipeline completes with `PHASE 1 PROTOTYPE COMPLETE` (label kept from Phase 1; pipeline.py:171). The `ABLATION` section shows seven rows including `Filing only` and two new columns (`AUROC_lo`, `AUROC_hi`). The final line of the section reports `Saved ablation results to: outputs/ablation_results.csv`.

- [ ] **Step 2: Inspect the resulting table**

Run:

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('outputs/ablation_results.csv')
print(df.round(4).to_string(index=False))
"
```

Expected: 7 rows × 7 columns (`Feature set, N, AUROC, AUROC_lo, AUROC_hi, AUPRC, Brier`). Note which subset has the highest AUROC, whether `Filing only` ≈ 0.5 (no signal) vs > 0.6 (real signal), and whether the top group's CI overlaps with `Full model`.

- [ ] **Step 3: Write `outputs/ablation_findings.md`**

Create `outputs/ablation_findings.md` using this template — fill the `<…>` slots from the inspection in Step 2:

```markdown
# Feature Group Ablation — Phase 2 Findings

**Date:** 2026-06-03
**Panel:** `data/processed/panel_phase2.csv` (72 firms, 11,496 firm-months, 8.7% event rate)
**Eval split:** validation (2021–2023)
**Model:** pooled logistic regression (statsmodels)
**Uncertainty:** 95% percentile bootstrap on validation AUROC, 200 resamples

## Result table

(paste the rounded CSV here as a markdown table)

## Headline finding

<one-sentence claim, e.g. "Market features alone (AUROC 0.XX, CI [...]) predict
12-month deterioration as well as the full 14-feature model (0.XX, CI [...])
— accounting and macro add no measurable lift at Phase 2 scale.">

## What carries the signal

- **Market features:** <reading from the numbers>
- **Accounting features (real SEC, Phase 2):** <reading>
- **Macro features:** <reading — is AUROC still ≈ 0.5 as in Phase 1?>
- **Filing features (`late_filing` alone):** <new in this run — reading>

## Limitations

- Panel excludes 4 of 5 REITs (`O`, `PLD`, `SPG`, `VTR`) and `DE`, dropped by the
  panel `dropna` because `wc_ratio` is structurally NaN for REITs and not
  reported for `DE`. Accounting-only and Full-model rows therefore reflect a
  panel skewed away from rate-sensitive firms — fix tracked separately.
- Bootstrap CIs are computed on the validation split (~324 rows in Phase 1
  proportions; recheck on Phase 2). For very small subsets, CI width is a
  better guide than the point estimate.
- All models are pooled logit; fixed-effects and hazard variants are not
  ablated here.

## Implications for Phase 3

- Item #8 (calibration): <if Market dominates, calibration on Market-only model
  is the priority>
- Item #4 (horizon analysis): <run on the best-performing subset first>
- Item #7 (error analysis): <slice errors of the leading subset by sector>
```

- [ ] **Step 4: Commit the findings**

```bash
git add outputs/ablation_findings.md
git commit -m "report: Phase 2 feature-group ablation findings"
```

(Leave `outputs/ablation_results.csv` unstaged unless the team decides to commit generated artifacts; it's regenerated on every pipeline run.)

---

## Self-Review

**1. Spec coverage:** Phase 3 proposal item #2 says "Compare market-only, accounting-only, macro-only, filing-only and combined models." All five subsets present after Task 2 (Filing only added; the other four already existed). Bootstrap CIs and persisted CSV are additions beyond the literal spec to make the comparison defensible — both justified above.

**2. Placeholder scan:** No `TBD` / `TODO` / vague-handwave lines remain. The findings doc in Task 4 is a *template with explicit `<…>` slots to fill in from numbers the engineer just printed* — those are not placeholders, they are clearly scoped human inputs that can't be predicted before the run.

**3. Type / signature consistency:**
- `_bootstrap_auroc_ci(y_true, y_pred, n_boot=200, alpha=0.05, seed=42) -> (float, float)` — used identically in Task 3 Step 4.
- Result-row keys (`Feature set`, `N`, `AUROC`, `AUROC_lo`, `AUROC_hi`, `AUPRC`, `Brier`) — same set in Task 2 (initial) and Task 3 (extended), and assertions in `tests/ablation_test.py` reference the exact strings.
- `PATHS.OUTPUTS` defined in Task 1, consumed in Task 2 Step 3 and Task 3 (indirectly via the same code path).
- Test file path stable across Task 2 and Task 3 (`tests/ablation_test.py`).

No inconsistencies found.
