# Category, Sector, and Error Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the v2-identified leading model (Market-only, pooled logit) to answer "where does the EWS work and where does it fail?" along two orthogonal slices — industry (sector) and company archetype (Distressed/Cyclical/Stable/Growth/Defensive/Rate-sensitive/Commodity-sensitive) — and pair the per-slice AUROC table with a per-slice false-positive / false-negative breakdown at the project's standard top-decile flagging threshold. Delivers Phase 3 items #3 (company-category performance), #6 (sector-level analysis), and #7 (error analysis) in one bundle.

**Architecture:** Categorisation comes from the `data/List of sample company` Phase 2 doc, parsed into a tracked CSV (`data/firm_categories.csv`) by a small one-off script kept in `scripts/` for reproducibility. `panel.py` gains an `archetype` merge step. New helpers in `eval.py` (`evaluate_by_slice`, `error_analysis_by_slice`) reuse v2's firm-clustered bootstrap and operate on the same train/val split. `pipeline.py` calls them for `industry` and `archetype` columns. Outputs four CSVs (gitignored) and one hand-written findings markdown.

**Tech Stack:** pandas, numpy, statsmodels.Logit (existing), no new libraries. Tests follow the existing `tests/smoke_test.py` / `tests/ablation_test.py` runnable-script pattern (no pytest dependency added).

---

## File Structure

- **Create:** `scripts/extract_firm_categories.py` — one-off parser, idempotent, run again whenever the sample-company doc changes
- **Create:** `data/firm_categories.csv` — committed, source of truth for the panel merge (columns: `ticker, sector_raw, archetype, purpose`)
- **Modify:** `src/ews/panel.py` — load `firm_categories.csv` and left-join on `ticker` before the dropna
- **Modify:** `src/ews/eval.py` — add `evaluate_by_slice`, `error_analysis_by_slice` helpers; reuse `_bootstrap_auroc_ci` and `_fit_pooled` from v2
- **Modify:** `src/ews/config.py` — add `MARKET_FEATURE_COLS` constant (Market-only feature list) so the slice analysis and v2's Market-only ablation cite the same source
- **Modify:** `src/ews/pipeline.py` — call the new analysis functions; persist results
- **Create:** `tests/category_sector_test.py` — runnable assertion script (mirrors `tests/smoke_test.py` style)
- **Create:** `outputs/category_sector_findings.md` — hand-written summary
- **Generated** (gitignored): `outputs/sector_results.csv`, `outputs/category_results.csv`, `outputs/sector_errors.csv`, `outputs/category_errors.csv`

---

### Task 1: Extract firm categories from the Phase 2 sample-company doc

**Why first:** every downstream task depends on a per-ticker `archetype` value. The Phase 2 doc encodes this in markdown; we parse it once and commit the result so the pipeline doesn't depend on doc formatting forever.

**Archetypes (normalised to 7 buckets):**
- `Distressed` — firm has a documented distress / bankruptcy / restructuring history
- `Cyclical` — sensitive to business cycle (auto, airlines, durables)
- `Stable` — defensive benchmark; tech, healthcare, staples that are not growth-stories
- `Growth` — volatile growth (tech growth, semis flagged as growth, etc.)
- `Defensive` — explicit defensive benchmark (pharma, staples large caps)
- `Rate-sensitive` — REITs, real estate, financials
- `Commodity-sensitive` — energy, materials, oil services

**Heuristic** (substring match against the doc's `Category` + `Purpose` strings, case-insensitive, first hit wins):

| keyword fragments | maps to |
|---|---|
| `distress`, `restructur`, `bankrupt` | `Distressed` |
| `cyclical` | `Cyclical` |
| `growth` | `Growth` |
| `defensive` | `Defensive` |
| `rate-sensitive`, `interest-rate`, `real estate` | `Rate-sensitive` |
| `commodity` | `Commodity-sensitive` |
| anything else | `Stable` |

**Files:**
- Create: `scripts/extract_firm_categories.py`
- Create: `data/firm_categories.csv`
- Create: `tests/category_sector_test.py`

- [ ] **Step 1: Write the failing test (creates the test file)**

Create `tests/category_sector_test.py`:

```python
"""Category/sector analysis test — runnable script, mirrors tests/smoke_test.py.

Run from repo root:
    python tests/category_sector_test.py

Exits 0 on success, nonzero with a clear message on any failure.
Requires data/processed/panel_phase2.csv (run pipeline first).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ews.config import FIRMS, PATHS  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


print("\n[1] firm_categories.csv exists and covers every configured ticker")
cat_path = os.path.join(REPO_ROOT, "data", "firm_categories.csv")
check("data/firm_categories.csv exists", os.path.isfile(cat_path), cat_path)
if os.path.isfile(cat_path):
    cats = pd.read_csv(cat_path)
    expected_cols = {"ticker", "sector_raw", "archetype", "purpose"}
    check(
        "schema has ticker, sector_raw, archetype, purpose",
        expected_cols.issubset(set(cats.columns)),
        f"got {sorted(cats.columns)}",
    )
    configured = set(FIRMS.keys())
    in_csv = set(cats["ticker"].astype(str))
    missing = configured - in_csv
    check(
        "every ticker in config.FIRMS has a category row",
        not missing,
        f"missing: {sorted(missing)}" if missing else "",
    )
    valid_archetypes = {
        "Distressed", "Cyclical", "Stable", "Growth",
        "Defensive", "Rate-sensitive", "Commodity-sensitive",
    }
    bad = set(cats["archetype"]) - valid_archetypes
    check(
        "every archetype is one of the 7 buckets",
        not bad,
        f"unexpected: {sorted(bad)}" if bad else "",
    )


print("\n" + "=" * 60)
if FAILURES:
    print(f"CATEGORY/SECTOR TEST FAILED — {len(FAILURES)} assertion(s) failed:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("CATEGORY/SECTOR TEST PASSED")
    sys.exit(0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/ivanchow/Documents/projects/hku-final
.venv/bin/python tests/category_sector_test.py
```

Expected: section [1] FAILs with `data/firm_categories.csv exists: …no such file`. Exit 1.

- [ ] **Step 3: Write the parser script**

Create `scripts/extract_firm_categories.py`:

```python
"""One-off parser: extract firm categories from data/List of sample company.

Reads the Phase 2 markdown doc, walks every | ... | Ticker | ... | Category |
Purpose | row, and writes data/firm_categories.csv with columns
(ticker, sector_raw, archetype, purpose).

`archetype` is normalised to one of seven buckets via case-insensitive
substring match against (Category + Purpose). See the plan
docs/superpowers/plans/2026-06-05-category-sector-error-analysis.md
for the keyword → bucket mapping.

Run:
    python scripts/extract_firm_categories.py

Idempotent — overwrites the CSV.
"""

from __future__ import annotations

import csv
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO_ROOT, "data", "List of sample company")
OUT = os.path.join(REPO_ROOT, "data", "firm_categories.csv")


def normalise_archetype(category: str, purpose: str) -> str:
    """Map (category, purpose) free-text labels to one of seven buckets."""
    blob = f"{category} {purpose}".lower()
    rules: list[tuple[list[str], str]] = [
        (["distress", "restructur", "bankrupt"], "Distressed"),
        (["rate-sensitive", "interest-rate", "real estate"], "Rate-sensitive"),
        (["commodity"], "Commodity-sensitive"),
        (["growth"], "Growth"),
        (["defensive"], "Defensive"),
        (["cyclical"], "Cyclical"),
    ]
    for fragments, bucket in rules:
        if any(f in blob for f in fragments):
            return bucket
    return "Stable"


def parse_rows(doc_text: str) -> list[dict]:
    """Walk the markdown doc, yielding one dict per company row.

    A company row has the shape:
        | <num> | <TICKER> | <Company> | <Category> | <Purpose> |
    Skips header / separator rows by requiring TICKER to look like
    1-5 uppercase letters.
    """
    rows: list[dict] = []
    ticker_re = re.compile(r"^[A-Z]{1,5}$")
    for line in doc_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        # Expected columns: No., Ticker, Company, Category, Purpose
        _num, ticker, _company, category, purpose = cells[0], cells[1], cells[2], cells[3], cells[4]
        if not ticker_re.match(ticker):
            continue
        sector_raw = category.split("/")[0].strip()
        rows.append({
            "ticker": ticker,
            "sector_raw": sector_raw,
            "archetype": normalise_archetype(category, purpose),
            "purpose": purpose,
        })
    return rows


def main() -> int:
    if not os.path.isfile(DOC):
        print(f"ERROR: source doc not found: {DOC}", file=sys.stderr)
        return 1
    with open(DOC) as f:
        rows = parse_rows(f.read())
    if not rows:
        print("ERROR: parser produced 0 rows; check doc format", file=sys.stderr)
        return 1
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "sector_raw", "archetype", "purpose"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the parser**

```bash
cd /Users/ivanchow/Documents/projects/hku-final
.venv/bin/python scripts/extract_firm_categories.py
```

Expected: `Wrote 80 rows to /Users/ivanchow/Documents/projects/hku-final/data/firm_categories.csv`.

- [ ] **Step 5: Sanity-check the output**

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('data/firm_categories.csv')
print(f'rows: {len(df)}')
print()
print('archetype counts:')
print(df.archetype.value_counts())
print()
print('sample rows:')
print(df.head(10).to_string(index=False))
"
```

Expected output: 80 rows; each of the 7 archetypes has at least one firm; `SPG / Simon Property` shows `Rate-sensitive`; `BBBY` shows `Distressed`; `AAPL` shows `Stable`.

If any spot-check looks wrong (e.g., SPG ends up as `Stable`), the keyword heuristic missed something. Adjust the `rules` list in `normalise_archetype` and re-run Step 4.

- [ ] **Step 6: Run the test to verify it passes**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: section [1] all `[PASS]`, exit 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/extract_firm_categories.py data/firm_categories.csv tests/category_sector_test.py
git commit -m "data: parse Phase 2 sample-company doc into firm_categories.csv (7-bucket archetype)"
```

**Do NOT add a Co-Authored-By line.**

---

### Task 2: Merge `archetype` and `sector_raw` into the panel

**Files:**
- Modify: `src/ews/panel.py:50-95` (the `assemble_panel` function)
- Modify: `tests/category_sector_test.py` (add section [2])

- [ ] **Step 1: Extend the test**

Append to `tests/category_sector_test.py` before the `print("\n" + "=" * 60)` summary line:

```python
print("\n[2] panel has archetype + sector_raw columns after assembly")
panel_path = os.path.join(PATHS.PROCESSED, "panel_phase2.csv")
assert os.path.isfile(panel_path), f"missing panel: {panel_path} — run pipeline first"
panel = pd.read_csv(panel_path)
check(
    "archetype column present in panel_phase2.csv",
    "archetype" in panel.columns,
    f"got columns: {list(panel.columns)[-5:]}",  # tail of columns
)
check(
    "sector_raw column present in panel_phase2.csv",
    "sector_raw" in panel.columns,
)
if "archetype" in panel.columns:
    check(
        "no panel row has missing archetype",
        panel["archetype"].notna().all(),
        f"{panel['archetype'].isna().sum()} rows have NaN archetype",
    )
    check(
        "archetype values match the 7-bucket set",
        set(panel["archetype"].unique()).issubset({
            "Distressed", "Cyclical", "Stable", "Growth",
            "Defensive", "Rate-sensitive", "Commodity-sensitive",
        }),
        f"got: {sorted(panel['archetype'].unique())}",
    )
```

- [ ] **Step 2: Run the test to verify section [2] fails**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: section [1] passes; section [2] fails with `archetype column present in panel_phase2.csv: …columns: [..., 'firm_name']`. Exit 1.

- [ ] **Step 3: Edit `src/ews/panel.py` to load and merge the categories**

In `src/ews/panel.py`, find the existing `assemble_panel` function. After the imports at the top of the file, add the load helper at module level (above `assemble_panel`):

```python
import os

from .config import (
    FEATURE_COLS,
    FIRMS,
    LABEL_COL,
    PANEL_START_YEAR,
    PATHS,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
)


def _load_firm_categories() -> pd.DataFrame:
    """Read data/firm_categories.csv → DataFrame[ticker, sector_raw, archetype, purpose].

    Repo-root-anchored path so callers don't need a working directory.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "data", "firm_categories.csv")
    return pd.read_csv(path)
```

Then inside `assemble_panel`, add a merge step. The current sequence ends with the impute + dropna. Insert the category merge *between* the firm-metadata assignment (current step 4) and the year-filter (current step 5):

Current (around lines 78-86):

```python
    # 4. Firm metadata
    panel["industry"] = panel["ticker"].map(lambda t: FIRMS[t]["industry"])
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["firm_name"] = panel["ticker"].map(lambda t: FIRMS[t]["name"])

    # 5. Filter to modeling window + stable sort
    panel = panel[panel["year"] >= PANEL_START_YEAR].copy()
```

Insert between step 4 and step 5:

```python
    # 4. Firm metadata
    panel["industry"] = panel["ticker"].map(lambda t: FIRMS[t]["industry"])
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["firm_name"] = panel["ticker"].map(lambda t: FIRMS[t]["name"])

    # 4b. Attach archetype + raw sector (parsed from Phase 2 sample-company doc).
    #     Left-join so a firm missing from firm_categories.csv keeps a NaN
    #     archetype rather than dropping the row silently.
    cats = _load_firm_categories()[["ticker", "sector_raw", "archetype"]]
    panel = panel.merge(cats, on="ticker", how="left")

    # 5. Filter to modeling window + stable sort
    panel = panel[panel["year"] >= PANEL_START_YEAR].copy()
```

- [ ] **Step 4: Re-run the pipeline so the panel CSV is refreshed**

```bash
MPLBACKEND=Agg .venv/bin/python src/run.py 2>&1 | grep -E "Panel:|Firms:" | head -3
```

Expected: `Panel: 12473 firm-months …`, `Firms: 77 …` (unchanged from v2 — the merge doesn't drop rows).

- [ ] **Step 5: Run the test**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: every `[PASS]`, exit 0. Section [2] confirms the new columns are populated and within the 7-bucket set.

- [ ] **Step 6: Commit**

```bash
git add src/ews/panel.py tests/category_sector_test.py
git commit -m "panel: attach archetype + sector_raw from firm_categories.csv before dropna"
```

**Do NOT add a Co-Authored-By line.**

---

### Task 3: Per-slice AUROC + bootstrap CIs

**Files:**
- Modify: `src/ews/config.py` — add `MARKET_FEATURE_COLS` constant
- Modify: `src/ews/eval.py` — add `evaluate_by_slice`
- Modify: `tests/category_sector_test.py` — add section [3]

- [ ] **Step 1: Extend the test**

Append to `tests/category_sector_test.py` before the summary line:

```python
print("\n[3] evaluate_by_slice produces a per-slice AUROC table")
from ews.eval import evaluate_by_slice  # noqa: E402
from ews.config import MARKET_FEATURE_COLS, LABEL_COL  # noqa: E402

# Use the panel loaded in section [2]; split same way the pipeline does.
train = panel[panel["year"] <= 2020]
val = panel[(panel["year"] > 2020) & (panel["year"] <= 2023)]

sector_rdf = evaluate_by_slice(train, val, slice_col="industry",
                               feature_cols=MARKET_FEATURE_COLS, label_col=LABEL_COL)
check(
    "industry slice returns a DataFrame with required columns",
    {"slice", "n_rows", "n_firms", "n_events", "AUROC", "AUROC_lo", "AUROC_hi"}.issubset(set(sector_rdf.columns)),
    f"got {sorted(sector_rdf.columns)}",
)
check(
    "industry slice has multiple groups",
    len(sector_rdf) >= 5,
    f"got {len(sector_rdf)} rows",
)
if "n_rows" in sector_rdf.columns:
    check(
        "n_rows sums to val length (modulo all-one-class slices that may drop)",
        sector_rdf["n_rows"].sum() >= 0.9 * len(val),
        f"sum={sector_rdf['n_rows'].sum()} vs val={len(val)}",
    )

arch_rdf = evaluate_by_slice(train, val, slice_col="archetype",
                             feature_cols=MARKET_FEATURE_COLS, label_col=LABEL_COL)
check(
    "archetype slice returns >= 3 groups",
    len(arch_rdf) >= 3,
    f"got {len(arch_rdf)} rows",
)
```

- [ ] **Step 2: Run the test to verify section [3] fails**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: section [3] fails with `ImportError: cannot import name 'MARKET_FEATURE_COLS'` or `cannot import name 'evaluate_by_slice'`. Exit 1.

- [ ] **Step 3: Add `MARKET_FEATURE_COLS` to `src/ews/config.py`**

Find the `FEATURE_COLS = [...]` block in `src/ews/config.py`. Immediately after it, add:

```python
# Subset used by Phase 3 slice analyses (item #3/#6/#7). v2 ablation identified
# Market-only pooled logit as the leading interpretable model (highest
# point-estimate AUROC across all three families). Define it here so slice
# analyses, calibration (item #8), and horizon analysis (item #4) cite a
# single source of truth.
MARKET_FEATURE_COLS = [
    "ret_1m", "ret_3m", "ret_6m",
    "vol_3m", "vol_6m", "drawdown_12m",
]
```

- [ ] **Step 4: Add `evaluate_by_slice` to `src/ews/eval.py`**

Insert this function in `src/ews/eval.py` after the existing `ablation_analysis` (use the same module-level structure as other diagnostics):

```python
# =============================================================================
# Per-slice evaluation (Phase 3 items #3 + #6)
# =============================================================================

def evaluate_by_slice(
    train: pd.DataFrame,
    val: pd.DataFrame,
    slice_col: str,
    feature_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    """Fit a pooled logit on `feature_cols` and evaluate per unique value of
    `slice_col` on the val set. AUROC uses firm-clustered bootstrap CIs.

    Returns DataFrame with columns:
      slice, n_rows, n_firms, n_events, event_rate, AUROC, AUROC_lo, AUROC_hi

    Slices with fewer than 2 events or fewer than 30 rows are returned with
    NaN AUROC/CI (uninformative) but kept in the result so the reader sees
    the sample-size landscape.
    """
    print(f"\nPer-slice evaluation on '{slice_col}':")

    # Fit once on train using ALL features in feature_cols.
    m = sm.Logit(train[label_col], sm.add_constant(train[feature_cols])).fit(disp=0)
    val_pred = m.predict(sm.add_constant(val[feature_cols]))

    rows: list[dict] = []
    for slice_val, sub in val.groupby(slice_col, dropna=False):
        sub_idx = sub.index
        y_true = sub[label_col].values
        y_pred = val_pred.loc[sub_idx].values
        firm_ids = sub["ticker"].values
        n_rows = len(sub)
        n_firms = sub["ticker"].nunique()
        n_events = int(y_true.sum())
        event_rate = float(y_true.mean()) if n_rows else float("nan")
        # AUROC undefined if only one class or sample too small
        if n_events < 2 or n_events == n_rows or n_rows < 30:
            auroc = float("nan")
            lo = hi = float("nan")
        else:
            auroc = roc_auc_score(y_true, y_pred)
            lo, hi = _bootstrap_auroc_ci(y_true, y_pred, firm_ids=firm_ids)
        rows.append({
            "slice": slice_val,
            "n_rows": n_rows,
            "n_firms": n_firms,
            "n_events": n_events,
            "event_rate": event_rate,
            "AUROC": auroc,
            "AUROC_lo": lo,
            "AUROC_hi": hi,
        })

    rdf = pd.DataFrame(rows).sort_values("AUROC", ascending=False, na_position="last")
    print(rdf.round(3).to_string(index=False))
    return rdf
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: every `[PASS]`, exit 0. The terminal output will show two per-slice tables (industry and archetype) printed by `evaluate_by_slice`.

- [ ] **Step 6: Commit**

```bash
git add src/ews/config.py src/ews/eval.py tests/category_sector_test.py
git commit -m "eval: per-slice AUROC + clustered-bootstrap CIs (industry and archetype)"
```

**Do NOT add a Co-Authored-By line.**

---

### Task 4: Per-slice error analysis (FP / FN at top-decile threshold)

**Files:**
- Modify: `src/ews/eval.py` — add `error_analysis_by_slice`
- Modify: `tests/category_sector_test.py` — add section [4]

The error analysis uses the top-decile flagging threshold (`TOP_K_FRACTION = 0.10` in config), matching how the analyst workflow actually uses the model. For each slice we count true positives, false positives, false negatives, true negatives, and report precision and recall at that operating point.

- [ ] **Step 1: Extend the test**

Append to `tests/category_sector_test.py` before the summary line:

```python
print("\n[4] error_analysis_by_slice produces per-slice precision/recall + counts")
from ews.eval import error_analysis_by_slice  # noqa: E402

err_sector = error_analysis_by_slice(train, val, slice_col="industry",
                                     feature_cols=MARKET_FEATURE_COLS, label_col=LABEL_COL)
expected = {"slice", "n_rows", "n_events", "n_flags", "TP", "FP", "FN", "TN",
            "precision", "recall"}
check(
    "error CSV has required columns",
    expected.issubset(set(err_sector.columns)),
    f"got {sorted(err_sector.columns)}",
)
if expected.issubset(set(err_sector.columns)):
    # Counts must add up per slice
    counts_ok = ((err_sector["TP"] + err_sector["FP"] + err_sector["FN"] + err_sector["TN"])
                 == err_sector["n_rows"]).all()
    check("TP+FP+FN+TN == n_rows for every slice", bool(counts_ok))
    # Recall = TP/(TP+FN) whenever events > 0
    nonzero = err_sector[err_sector["n_events"] > 0]
    if len(nonzero) > 0:
        expected_recall = nonzero["TP"] / nonzero["n_events"]
        check(
            "recall = TP/(TP+FN) within 1e-6",
            (abs(nonzero["recall"] - expected_recall) < 1e-6).all(),
        )
```

- [ ] **Step 2: Run the test to verify section [4] fails**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: section [4] fails with `ImportError: cannot import name 'error_analysis_by_slice'`. Exit 1.

- [ ] **Step 3: Add `error_analysis_by_slice` to `src/ews/eval.py`**

Insert after `evaluate_by_slice` in `src/ews/eval.py`:

```python
def error_analysis_by_slice(
    train: pd.DataFrame,
    val: pd.DataFrame,
    slice_col: str,
    feature_cols: list[str],
    label_col: str,
    top_k_fraction: float = TOP_K_FRACTION,
) -> pd.DataFrame:
    """Per-slice false-positive / false-negative breakdown at top-decile
    flagging threshold.

    Threshold is chosen as the val-set quantile (1 - top_k_fraction) of
    predicted probabilities — the same population the analyst sees. Slices
    inherit that global threshold so per-slice precision/recall is comparable.

    Returns DataFrame:
      slice, n_rows, n_events, n_flags, TP, FP, FN, TN, precision, recall
    """
    print(f"\nPer-slice error analysis on '{slice_col}' (threshold = top {top_k_fraction:.0%}):")

    m = sm.Logit(train[label_col], sm.add_constant(train[feature_cols])).fit(disp=0)
    val_pred = m.predict(sm.add_constant(val[feature_cols]))
    threshold = float(val_pred.quantile(1 - top_k_fraction))
    print(f"  Global flagging threshold (val top-{int(top_k_fraction*100)}%): {threshold:.4f}")

    val_flagged = (val_pred >= threshold).astype(int)
    y = val[label_col].astype(int)

    rows: list[dict] = []
    for slice_val, sub in val.groupby(slice_col, dropna=False):
        sub_idx = sub.index
        y_sub = y.loc[sub_idx].values
        f_sub = val_flagged.loc[sub_idx].values
        TP = int(((f_sub == 1) & (y_sub == 1)).sum())
        FP = int(((f_sub == 1) & (y_sub == 0)).sum())
        FN = int(((f_sub == 0) & (y_sub == 1)).sum())
        TN = int(((f_sub == 0) & (y_sub == 0)).sum())
        n_rows = len(sub)
        n_events = int(y_sub.sum())
        n_flags = int(f_sub.sum())
        precision = TP / n_flags if n_flags > 0 else float("nan")
        recall = TP / n_events if n_events > 0 else float("nan")
        rows.append({
            "slice": slice_val,
            "n_rows": n_rows,
            "n_events": n_events,
            "n_flags": n_flags,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": precision,
            "recall": recall,
        })

    rdf = pd.DataFrame(rows).sort_values("recall", ascending=False, na_position="last")
    print(rdf.round(3).to_string(index=False))
    return rdf
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: every `[PASS]`, exit 0. Section [4] now prints a per-sector error table.

- [ ] **Step 5: Commit**

```bash
git add src/ews/eval.py tests/category_sector_test.py
git commit -m "eval: per-slice error analysis (precision/recall + TP/FP/FN/TN at top-decile)"
```

**Do NOT add a Co-Authored-By line.**

---

### Task 5: Pipeline integration + hand-write findings doc

**Files:**
- Modify: `src/ews/pipeline.py` — call the two new analyses, persist 4 CSVs
- Modify: `.gitignore` — ignore the 4 new generated CSVs
- Modify: `tests/category_sector_test.py` — add section [5]
- Create: `outputs/category_sector_findings.md`

- [ ] **Step 1: Extend the test**

Append to `tests/category_sector_test.py` before the summary line:

```python
print("\n[5] pipeline persists 4 slice CSVs to outputs/")
for fname in ("sector_results.csv", "category_results.csv",
              "sector_errors.csv", "category_errors.csv"):
    path = os.path.join(PATHS.OUTPUTS, fname)
    check(f"{fname} exists", os.path.isfile(path), path)
```

- [ ] **Step 2: Run the test to verify section [5] fails**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: section [5] fails — no CSVs yet. Exit 1.

- [ ] **Step 3: Update `.gitignore` to ignore the 4 new CSVs**

Append to `.gitignore`:

```
outputs/sector_results.csv
outputs/category_results.csv
outputs/sector_errors.csv
outputs/category_errors.csv
```

- [ ] **Step 4: Call the analyses from `src/ews/pipeline.py`**

Find the existing block in `src/ews/pipeline.py` (around line 161) that calls `ablation_analysis`:

```python
    try:
        ablation_analysis(train, eval_data)
    except Exception as e:
        _warn(f"Ablation failed: {e}")
```

After it, insert:

```python
    # -- Per-slice analyses (Phase 3 items #3 + #6 + #7) -----------------
    from .config import MARKET_FEATURE_COLS  # local import; new constant
    from .eval import evaluate_by_slice, error_analysis_by_slice  # local

    for slice_col, stem in (("industry", "sector"), ("archetype", "category")):
        try:
            rdf = evaluate_by_slice(
                train, eval_data,
                slice_col=slice_col,
                feature_cols=MARKET_FEATURE_COLS,
                label_col=LABEL_COL,
            )
            os.makedirs(PATHS.OUTPUTS, exist_ok=True)
            rdf.to_csv(os.path.join(PATHS.OUTPUTS, f"{stem}_results.csv"), index=False)
            print(f"  Saved {stem} AUROC results to: outputs/{stem}_results.csv")
        except Exception as e:
            _warn(f"Per-slice AUROC ({slice_col}) failed: {e}")

        try:
            edf = error_analysis_by_slice(
                train, eval_data,
                slice_col=slice_col,
                feature_cols=MARKET_FEATURE_COLS,
                label_col=LABEL_COL,
            )
            edf.to_csv(os.path.join(PATHS.OUTPUTS, f"{stem}_errors.csv"), index=False)
            print(f"  Saved {stem} error analysis to: outputs/{stem}_errors.csv")
        except Exception as e:
            _warn(f"Per-slice errors ({slice_col}) failed: {e}")
```

- [ ] **Step 5: Re-run the pipeline**

```bash
MPLBACKEND=Agg .venv/bin/python src/run.py 2>&1 | grep -E "^  Saved|^Per-slice" | head -20
```

Expected: four `Saved … to: outputs/{sector,category}_{results,errors}.csv` lines.

- [ ] **Step 6: Run the test to verify all sections pass**

```bash
.venv/bin/python tests/category_sector_test.py
```

Expected: every `[PASS]`, exit 0.

- [ ] **Step 7: Inspect the four CSVs**

```bash
.venv/bin/python -c "
import pandas as pd
for n in ('sector_results', 'category_results', 'sector_errors', 'category_errors'):
    print(f'\\n=== {n}.csv ===')
    print(pd.read_csv(f'outputs/{n}.csv').round(3).to_string(index=False))
"
```

Note (for the findings doc in Step 8):
- which industries / archetypes have the **highest AUROC** (the model "works on these")
- which have the **lowest AUROC** or NaN (the model "fails on these" — or sample too small)
- which slices have **highest recall at top-decile** (the model catches most of the events for these groups)
- which slices have **lowest precision** (the model flags here a lot but most are false alarms)
- the **REIT-specific** rows (sector_raw='Real Estate' and/or archetype='Rate-sensitive') — this is the deliverable that pays off v2's REIT recovery

- [ ] **Step 8: Write `outputs/category_sector_findings.md`**

Create `outputs/category_sector_findings.md`. Use this template; fill the `<…>` slots from Step 7's tables:

```markdown
# Category & Sector Performance Analysis

**Date:** 2026-06-05
**Model:** Market-only pooled logit (6 features: ret_1m/3m/6m, vol_3m/6m, drawdown_12m) — the v2-identified leader.
**Eval split:** validation 2021–2023 (~2,772 rows).
**Slicing:**
  - **Sector** = `industry` column (8 buckets: Industrial, Consumer, Retail, Energy, Technology, Healthcare, RealEstate, Airlines).
  - **Archetype** = parsed from Phase 2 sample-company doc, normalised to 7 buckets (Distressed, Cyclical, Stable, Growth, Defensive, Rate-sensitive, Commodity-sensitive).
**Uncertainty:** 95% firm-clustered bootstrap, 1,000 resamples (inherited from v2 work).
**Operating threshold:** top decile of val predicted probability (matches analyst workflow).

## Per-sector AUROC

(paste the rounded `outputs/sector_results.csv` here as a markdown table, sorted by AUROC desc; include n_firms and n_events columns so the reader sees sample sizes.)

## Per-archetype AUROC

(paste the rounded `outputs/category_results.csv` here as a markdown table, sorted by AUROC desc.)

## Per-sector error patterns (top-decile threshold)

(paste rounded `outputs/sector_errors.csv` as a markdown table.)

## Per-archetype error patterns (top-decile threshold)

(paste rounded `outputs/category_errors.csv` as a markdown table.)

## Headline findings

Write 2-4 short bullets, each grounded in a specific number from the tables above. Examples (replace with actuals):

- **Works well on:** <sector / archetype list with AUROC + CI>
- **Fails on:** <sector / archetype list with AUROC + CI>; flag slices where AUROC is NaN due to small samples
- **REIT result (paying off v2's recovery):** Real Estate sector AUROC = <X> with <n_firms> firms / <n_events> events; archetype 'Rate-sensitive' AUROC = <X>. State whether the REIT recovery turned out to be informative or whether sample size still limits the conclusion.
- **Where most false alarms concentrate:** <sector / archetype with lowest precision>. State the FP count and the slice's event base rate (often the model over-flags low-base-rate slices).
- **Where most events are missed:** <sector / archetype with lowest recall>. State the FN count.

## Limitations

- The 7-archetype mapping is a heuristic substring match against the Phase 2 doc. Edge cases default to `Stable`; a marker who challenges a specific firm's classification can be referred to `scripts/extract_firm_categories.py` for the deterministic rule.
- Per-slice CIs use firm-clustered bootstrap (correct for credit-risk panel structure) but their width is bounded below by the number of unique firms in each slice. Slices with < 3 firms produce CIs that should be read as "directional only."
- The Market-only model is one of three families tested in v2; pooled was chosen because it had the highest point-estimate AUROC. FE and hazard family slice results are not produced here. Horizon analysis (Phase 3 #4) and threshold sensitivity (#5) are queued separately.

## Implications for the rest of Phase 3

- **Item #4 (horizon analysis)** — run on the leading sector / archetype identified above; the AUROC ceiling is set by where the model already works.
- **Item #5 (threshold sensitivity)** — relabel at 30% / 50% drawdown and rerun this slicing; verify the leaderboard is not threshold-specific.
- **Item #8 (calibration)** — calibrate the Market-only pooled model first; the per-slice recall numbers here are the baseline a calibrated probability output must match or beat at the same threshold.
- **Item #9 (final model selection)** — synthesize this with the v2 ablation table and the horizon/calibration results once they exist.

## Reproducibility

- Pipeline command: `MPLBACKEND=Agg python src/run.py`
- Source CSVs: `outputs/{sector,category}_{results,errors}.csv` (all gitignored, regenerated every run)
- Parser script: `scripts/extract_firm_categories.py`
- Category source-of-truth: `data/firm_categories.csv` (committed)
- Test script: `tests/category_sector_test.py`
```

When filling `<…>` slots, every number must come from one of the four CSVs (round to 3 decimals where applicable). No invented numbers, no eyeballed p-values, no informal stats language.

- [ ] **Step 9: Cross-check the doc**

```bash
cat outputs/category_sector_findings.md | grep -oE '[0-9]+\.[0-9]{1,3}' | sort -u | head -20
```

For each unique number, verify it appears in one of the four CSVs (rounded). Any that doesn't → delete the corresponding claim.

- [ ] **Step 10: Commit**

```bash
git add src/ews/pipeline.py .gitignore tests/category_sector_test.py outputs/category_sector_findings.md
git commit -m "report: per-sector + per-archetype AUROC and error analysis (Phase 3 #3, #6, #7)"
```

**Do NOT add a Co-Authored-By line.**

---

## Self-Review

**1. Spec coverage:** Phase 3 items mapped:
- #3 Company-category performance → Task 3 (archetype slicing) + Task 5 (findings)
- #6 Sector-level analysis → Task 3 (industry slicing) + Task 5 (findings)
- #7 Error analysis → Task 4 (FP/FN slicing) + Task 5 (findings)
- Category data acquisition → Tasks 1 + 2 (parse doc + merge into panel)

All three Phase 3 items have at least one task that implements them. No gaps.

**2. Placeholder scan:** All `<…>` markers in the findings doc template (Task 5 Step 8) are labelled explicit human-input slots — they are intentional spaces for the engineer to fill in from CSVs they have just generated, not lazy TODOs. Every code step contains complete code. Every command line shows the expected output or behaviour.

**3. Type / signature consistency:**
- `evaluate_by_slice(train, val, slice_col, feature_cols, label_col)` defined in Task 3, called identically in Task 5.
- `error_analysis_by_slice(train, val, slice_col, feature_cols, label_col, top_k_fraction=TOP_K_FRACTION)` defined in Task 4, called identically in Task 5 (with default `top_k_fraction`).
- `MARKET_FEATURE_COLS` defined in Task 3, imported in Task 5.
- CSV schemas:
  - results CSVs: `slice, n_rows, n_firms, n_events, event_rate, AUROC, AUROC_lo, AUROC_hi` (Task 3, asserted in Task 3 test, used in Task 5)
  - errors CSVs: `slice, n_rows, n_events, n_flags, TP, FP, FN, TN, precision, recall` (Task 4, asserted in Task 4 test, used in Task 5)
- `archetype` column added to panel in Task 2, consumed in Tasks 3-5 via `slice_col="archetype"`.
- `data/firm_categories.csv` columns `ticker, sector_raw, archetype, purpose` defined in Task 1, asserted in Task 1 test, consumed in Task 2.

No inconsistencies found.

**4. Dependency order:**
- Task 1 (categories CSV) must precede Task 2 (panel merge reads the CSV)
- Task 2 (panel has archetype) must precede Tasks 3-4 (which slice on archetype)
- Tasks 3 + 4 (the per-slice helpers) must precede Task 5 (pipeline calls them)

Order is correct as written.

**5. Honesty about the categorisation:**
The plan calls out the heuristic mapping explicitly in Task 1, in the findings template's Limitations section, and in the script docstring. A reviewer challenging "why is X classified as Y?" has a single deterministic answer (the rules list in `normalise_archetype`).
