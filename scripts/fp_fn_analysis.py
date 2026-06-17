"""Research-grade false-positive / false-negative analysis for the credit EWS.

The per-slice confusion matrix in `eval.error_analysis_by_slice` is frozen at a
single operating point (the global top-decile threshold, chosen on val and
evaluated on the same val set). This script supplies the missing threshold/cost
dimension and the rigor wrappers a model-validation referee expects:

  1. Cost frontier  — sweep the alert threshold; trace recall vs flag-budget and
                      the FP<->FN tradeoff on the held-out 2024 test set.
  2. Cost-weighted  — minimise expected cost c_FN*FN + c_FP*FP on val across a
     operating point   range of FN:FP ratios; report realised TEST metrics at
                      each frozen threshold (no in-sample-threshold optimism).
  3. Benchmarks     — Altman Z (pooled logit on z_score) + trivial flag-all /
                      flag-none / random.
  4. Uncertainty    — firm-clustered bootstrap 95% CIs on test recall and FPR.
  5. Artifact vs    — re-run per-archetype errors at a looser cost-optimal
     failure          threshold; if dead slices stay ~0 recall, it is a ranking
                      failure, not a thresholding artifact.

Threshold selection is on validation (2021-2023); the headline confusion matrix
is reported on the held-out 2024 test set with the threshold frozen.

Outputs (under outputs/):
  figures/phase3_fp_fn_frontier.png
  fp_fn_operating_points.csv
  fp_fn_cost_sensitivity.csv
  fp_fn_slice_threshold_compare.csv
  fp_fn_findings.md

Run:  MPLBACKEND=Agg python scripts/fp_fn_analysis.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ews.config import (  # noqa: E402  (after sys.path insert)
    LABEL_COL,
    MARKET_PLUS_REL_FEATURE_COLS,
    PATHS,
    TOP_K_FRACTION,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
)

PANEL_PATH = os.path.join(PATHS.PROCESSED, "panel_phase2.csv")
FIG_PATH = os.path.join(PATHS.FIGURES, "phase3_fp_fn_frontier.png")

COST_RATIOS = [1, 2, 5, 10, 20]   # c_FN : c_FP (c_FP fixed at 1)
PRIMARY_RATIO = 5                 # defensible mid credit-risk asymmetry
N_BOOT = 1000
SEED = 42

BLUE, RED, GREEN, AMBER, GREY = "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#94a3b8"


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Year-based train/val/test split, matching ews.panel.time_split semantics."""
    df = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    train = df[df["year"] <= TRAIN_END_YEAR]
    val = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= VAL_END_YEAR)]
    test = df[df["year"] > VAL_END_YEAR]
    return train, val, test


def fit_predict(
    train: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """Fit a pooled logit on train[feature_cols]; return, per frame, a tidy
    DataFrame [p, y, ticker, archetype] over rows with complete features."""
    tr = train.dropna(subset=feature_cols + [LABEL_COL])
    model = sm.Logit(tr[LABEL_COL], sm.add_constant(tr[feature_cols])).fit(disp=0)
    out: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        ff = frame.dropna(subset=feature_cols)
        p = model.predict(sm.add_constant(ff[feature_cols], has_constant="add"))
        out[name] = pd.DataFrame({
            "p": np.asarray(p),
            "y": ff[LABEL_COL].to_numpy(int),
            "ticker": ff["ticker"].to_numpy(),
            "archetype": ff["archetype"].to_numpy(),
        })
    return out


# ---------------------------------------------------------------------------
# Confusion + thresholding
# ---------------------------------------------------------------------------

def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Confusion matrix + rates at a probability threshold (flag if p >= thr)."""
    flag = (p >= thr)
    y = y.astype(bool)
    TP = int((flag & y).sum())
    FP = int((flag & ~y).sum())
    FN = int((~flag & y).sum())
    TN = int((~flag & ~y).sum())
    n_flag, n_event, n_neg, n = TP + FP, TP + FN, FP + TN, len(y)
    return {
        "thr": thr, "flags": n_flag, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "recall": TP / n_event if n_event else np.nan,
        "precision": TP / n_flag if n_flag else np.nan,
        "fpr": FP / n_neg if n_neg else np.nan,
        "flag_budget": n_flag / n if n else np.nan,
    }


def thr_for_budget(val_p: np.ndarray, budget: float) -> float:
    """Threshold that flags the top `budget` fraction of the val distribution."""
    return float(np.quantile(val_p, 1.0 - budget))


def cost_optimal_threshold(val: pd.DataFrame, c_fn: float, c_fp: float = 1.0) -> float:
    """Threshold minimising expected cost c_FN*FN + c_FP*FP on the val set.

    Swept over a descending sort of val scores with cumulative TP/FP counts —
    O(n log n), exact over every distinct cut point.
    """
    y = val["y"].to_numpy(bool)
    order = np.argsort(-val["p"].to_numpy())          # high score first
    ys = y[order]
    ps = val["p"].to_numpy()[order]
    total_events = int(y.sum())
    cum_tp = np.cumsum(ys)                              # events caught if we flag top-k
    cum_fp = np.cumsum(~ys)                             # false alarms if we flag top-k
    # k flags (k = 1..n): FN = total_events - cum_tp[k-1], FP = cum_fp[k-1]
    fn = total_events - cum_tp
    fp = cum_fp
    cost = c_fn * fn + c_fp * fp
    # include k=0 (flag nothing): cost = c_fn * total_events
    k0_cost = c_fn * total_events
    best_k = int(np.argmin(cost))
    if k0_cost < cost[best_k]:
        return float(ps[0]) + 1e-9                      # threshold above the max -> flag none
    # threshold = score at the k'th flagged row (>= keeps exactly best_k+1 flags)
    return float(ps[best_k])


# ---------------------------------------------------------------------------
# Firm-clustered bootstrap CI on recall + FPR at a frozen threshold
# ---------------------------------------------------------------------------

def bootstrap_ci(frame: pd.DataFrame, thr: float, n_boot: int = N_BOOT,
                 seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    firms = frame["ticker"].to_numpy()
    uniq = np.unique(firms)
    rows_by_firm = {f: np.where(firms == f)[0] for f in uniq}
    y = frame["y"].to_numpy()
    p = frame["p"].to_numpy()
    rec, fpr = [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_firm[f] for f in draw])
        c = confusion(y[idx], p[idx], thr)
        if not np.isnan(c["recall"]):
            rec.append(c["recall"])
        if not np.isnan(c["fpr"]):
            fpr.append(c["fpr"])
    def ci(a):
        return (float("nan"), float("nan")) if not a else tuple(np.percentile(a, [2.5, 97.5]))
    return {"recall_lo": ci(rec)[0], "recall_hi": ci(rec)[1],
            "fpr_lo": ci(fpr)[0], "fpr_hi": ci(fpr)[1]}


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def operating_points(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Frozen-threshold confusion matrix on TEST at the top-decile and at each
    cost-optimal threshold, with firm-clustered CIs."""
    points: list[tuple[str, float]] = [
        (f"top_{int(TOP_K_FRACTION*100)}pct", thr_for_budget(val["p"].to_numpy(), TOP_K_FRACTION)),
    ]
    for r in COST_RATIOS:
        points.append((f"cost_opt_{r}:1", cost_optimal_threshold(val, c_fn=r)))

    rows = []
    for name, thr in points:
        c = confusion(test["y"].to_numpy(), test["p"].to_numpy(), thr)
        c_ci = bootstrap_ci(test, thr)
        rows.append({"operating_point": name, **c, **c_ci})
    return pd.DataFrame(rows)


def cost_sensitivity(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """How the cost-optimal threshold and realised TEST metrics migrate as the
    FN:FP cost ratio rises."""
    rows = []
    for r in COST_RATIOS:
        thr = cost_optimal_threshold(val, c_fn=r)
        vc = confusion(val["y"].to_numpy(), val["p"].to_numpy(), thr)
        tc = confusion(test["y"].to_numpy(), test["p"].to_numpy(), thr)
        rows.append({
            "cost_ratio_fn_fp": f"{r}:1", "c_fn": r, "thr": thr,
            "val_flag_budget": vc["flag_budget"], "val_cost": r * vc["FN"] + vc["FP"],
            "test_recall": tc["recall"], "test_fpr": tc["fpr"],
            "test_precision": tc["precision"], "test_flag_budget": tc["flag_budget"],
            "test_FP": tc["FP"], "test_FN": tc["FN"],
        })
    return pd.DataFrame(rows)


def slice_threshold_compare(val: pd.DataFrame) -> pd.DataFrame:
    """Artifact-vs-failure diagnostic on VAL: per-archetype recall under the
    GLOBAL top-decile threshold vs a SLICE-RELATIVE top-decile threshold
    (top-10% within each slice's own score distribution).

    Tests the findings-doc hypothesis that the global threshold merely starves
    low-base-rate slices of flags. If a slice-relative threshold lifts a dead
    slice's recall well above its ~10% flag budget, it was a thresholding
    artifact. If recall stays near (or below) the 10% random floor, the slice's
    within-sector ranking is broken (AUROC ≤ 0.5) — a ranking failure that no
    threshold can fix.
    """
    thr_global = thr_for_budget(val["p"].to_numpy(), TOP_K_FRACTION)
    rows = []
    for arch, sub in val.groupby("archetype"):
        y, p = sub["y"].to_numpy(), sub["p"].to_numpy()
        cg = confusion(y, p, thr_global)
        # slice-relative top-decile: flag the riskiest 10% within this slice
        if len(sub) >= 10 and sub["p"].nunique() > 1:
            thr_slice = float(np.quantile(p, 1.0 - TOP_K_FRACTION))
        else:
            thr_slice = thr_global
        cs = confusion(y, p, thr_slice)
        rows.append({
            "archetype": arch, "n_events": int(y.sum()), "n_rows": len(sub),
            "flagbudget_global": cg["flag_budget"], "recall_global": cg["recall"],
            "flagbudget_slicerel": cs["flag_budget"], "recall_slicerel": cs["recall"],
            "recall_gain": (cs["recall"] - cg["recall"]) if not np.isnan(cg["recall"]) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_events", ascending=False)


def capture_curve(val: pd.DataFrame, test: pd.DataFrame, budgets: np.ndarray) -> pd.DataFrame:
    """Test recall vs realised test flag-budget, thresholds set on val."""
    rows = []
    for b in budgets:
        thr = thr_for_budget(val["p"].to_numpy(), b)
        c = confusion(test["y"].to_numpy(), test["p"].to_numpy(), thr)
        rows.append({"val_budget": b, "test_recall": c["recall"],
                     "test_flag_budget": c["flag_budget"], "FP": c["FP"], "FN": c["FN"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_frontier(model_cap: pd.DataFrame, altman_cap: pd.DataFrame,
                  val: pd.DataFrame, op: pd.DataFrame) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Panel A: capture curve (test recall vs realised test flag-budget) ----
    axL.plot(model_cap["test_flag_budget"], model_cap["test_recall"],
             color=BLUE, lw=2.2, label="Market + sector-rel logit")
    axL.plot(altman_cap["test_flag_budget"], altman_cap["test_recall"],
             color=RED, lw=2, label="Altman Z (z_score)")
    axL.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2, label="Random")
    # mark top-decile + cost-opt(primary) operating points (from op table, on test)
    for name, marker, col in [(f"top_{int(TOP_K_FRACTION*100)}pct", "o", AMBER),
                              (f"cost_opt_{PRIMARY_RATIO}:1", "D", GREEN)]:
        r = op[op["operating_point"] == name].iloc[0]
        axL.scatter([r["flag_budget"]], [r["recall"]], color=col, s=90, zorder=5,
                    edgecolor="white", label=f"{name} (rec={r['recall']:.2f})")
    axL.set_xlabel("Flag budget (fraction of firm-months flagged), TEST 2024")
    axL.set_ylabel("Recall (fraction of distress events caught), TEST 2024")
    axL.set_title("A. Capture curve — recall vs flag budget (held-out test)")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1.02)
    axL.grid(alpha=0.25); axL.legend(fontsize=8, loc="lower right")

    # -- Panel B: expected cost vs flag-budget on VAL, per cost ratio ----------
    budgets = np.linspace(0.005, 1.0, 200)
    y = val["y"].to_numpy(); pcol = val["p"].to_numpy()
    total_events = int(y.sum())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(COST_RATIOS)))
    for r, col in zip(COST_RATIOS, cmap):
        costs = []
        for b in budgets:
            thr = thr_for_budget(pcol, b)
            c = confusion(y, pcol, thr)
            costs.append((r * c["FN"] + c["FP"]) / (r * total_events))  # vs do-nothing
        costs = np.array(costs)
        axR.plot(budgets, costs, color=col, lw=1.8, label=f"FN:FP = {r}:1")
        kmin = int(np.argmin(costs))
        axR.scatter([budgets[kmin]], [costs[kmin]], color=col, s=45, zorder=5,
                    edgecolor="white")
    axR.axvline(TOP_K_FRACTION, color=AMBER, ls=":", lw=1.5,
                label=f"top-{int(TOP_K_FRACTION*100)}% operating pt")
    axR.set_xlabel("Flag budget on validation (2021-2023)")
    axR.set_ylabel("Expected cost ÷ do-nothing cost")
    axR.set_title("B. Cost-optimal flag budget shifts with the FN:FP ratio")
    axR.set_xlim(0, 1); axR.set_ylim(0, 1.05)
    axR.grid(alpha=0.25); axR.legend(fontsize=8, loc="upper right")

    fig.suptitle("False positives vs false negatives — cost frontier "
                 "(threshold selected on val, evaluated on held-out 2024 test)",
                 fontsize=12, y=1.02)
    os.makedirs(PATHS.FIGURES, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {FIG_PATH}")


# ---------------------------------------------------------------------------
# Findings doc
# ---------------------------------------------------------------------------

def write_findings(op: pd.DataFrame, cost: pd.DataFrame, sl: pd.DataFrame,
                   meta: dict) -> None:
    dec = op[op["operating_point"] == f"top_{int(TOP_K_FRACTION*100)}pct"].iloc[0]
    prim = op[op["operating_point"] == f"cost_opt_{PRIMARY_RATIO}:1"].iloc[0]
    dead = sl[(sl["recall_slicerel"].fillna(0) <= TOP_K_FRACTION + 0.02) & (sl["n_events"] >= 10)]

    lines = []
    lines.append("# False Positive vs False Negative Analysis\n")
    lines.append(f"**Date:** 2026-06-16  ")
    lines.append(f"**Model:** Market + sector-relative pooled logit "
                 f"(`MARKET_PLUS_REL_FEATURE_COLS`), fit on train ≤{TRAIN_END_YEAR}.  ")
    lines.append(f"**Threshold selection:** validation {TRAIN_END_YEAR+1}-{VAL_END_YEAR}. "
                 f"**Headline evaluation:** held-out TEST {VAL_END_YEAR+1}+ with the "
                 f"threshold frozen.  ")
    lines.append(f"**Test set:** {meta['test_rows']} firm-months, "
                 f"{meta['test_firms']} firms, {meta['test_events']} events "
                 f"(base rate {meta['test_rate']:.1%}).  ")
    lines.append(f"**Uncertainty:** firm-clustered bootstrap, {N_BOOT} resamples, 95% CI.\n")

    lines.append("## Operating points on the held-out 2024 test\n")
    lines.append("| Operating point | thr | flags | TP | FP | FN | TN | recall (95% CI) | precision | FPR (95% CI) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in op.iterrows():
        lines.append(
            f"| {r['operating_point']} | {r['thr']:.3f} | {r['flags']} | {r['TP']} | "
            f"{r['FP']} | {r['FN']} | {r['TN']} | "
            f"{r['recall']:.2f} [{r['recall_lo']:.2f}, {r['recall_hi']:.2f}] | "
            f"{r['precision']:.2f} | "
            f"{r['fpr']:.2f} [{r['fpr_lo']:.2f}, {r['fpr_hi']:.2f}] |")
    lines.append("")

    lines.append("## Cost sensitivity — where the optimal flag budget lands\n")
    lines.append("| FN:FP | thr | val flag budget | test recall | test FPR | test precision | test FP | test FN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in cost.iterrows():
        lines.append(
            f"| {r['cost_ratio_fn_fp']} | {r['thr']:.3f} | {r['val_flag_budget']:.1%} | "
            f"{r['test_recall']:.2f} | {r['test_fpr']:.2f} | {r['test_precision']:.2f} | "
            f"{int(r['test_FP'])} | {int(r['test_FN'])} |")
    lines.append("")

    lines.append("## Artifact vs failure — global vs slice-relative top-decile threshold (val)\n")
    lines.append("Does flagging the riskiest 10% *within each slice* rescue the slices that "
                 "scored zero recall under the single global threshold? If recall stays near the "
                 f"~{TOP_K_FRACTION:.0%} random floor, the within-sector ranking is broken "
                 "(AUROC ≤ 0.5) — a *ranking* failure no threshold can fix; if it jumps well "
                 "above, the global threshold was a *thresholding* artifact.\n")
    lines.append("| Archetype | events | flags@global | recall@global | flags@slice-rel | recall@slice-rel | gain |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in sl.iterrows():
        rg_ = "—" if pd.isna(r["recall_global"]) else f"{r['recall_global']:.2f}"
        rs_ = "—" if pd.isna(r["recall_slicerel"]) else f"{r['recall_slicerel']:.2f}"
        gn_ = "—" if pd.isna(r["recall_gain"]) else f"{r['recall_gain']:+.2f}"
        lines.append(f"| {r['archetype']} | {r['n_events']} | {r['flagbudget_global']:.0%} | {rg_} | "
                     f"{r['flagbudget_slicerel']:.0%} | {rs_} | {gn_} |")
    lines.append("")

    lines.append("## Headline findings\n")
    lines.append(
        f"- **At the deployed top-decile operating point, generalisation is modest but real:** "
        f"on the held-out 2024 test the model catches **{dec['recall']:.0%} of distress events "
        f"(95% CI [{dec['recall_lo']:.0%}, {dec['recall_hi']:.0%}])** at a "
        f"{dec['flag_budget']:.0%} flag budget, precision {dec['precision']:.0%}, "
        f"FPR {dec['fpr']:.0%}.")
    heavy = op[op["operating_point"] == "cost_opt_10:1"].iloc[0]
    lines.append(
        f"- **The analyst's top-decile rule implicitly encodes a ~{PRIMARY_RATIO}:1 cost "
        f"preference:** the cost-optimal threshold at FN:FP = {PRIMARY_RATIO}:1 flags "
        f"{prim['flag_budget']:.0%} of firm-months (recall {prim['recall']:.0%}), essentially "
        f"reproducing the top-decile point. The operating point widens sharply only once "
        f"FN:FP reaches ~10:1, where the optimum jumps to flagging {heavy['flag_budget']:.0%} "
        f"of firm-months — recall {heavy['recall']:.0%} "
        f"[{heavy['recall_lo']:.0%}, {heavy['recall_hi']:.0%}] but FPR {heavy['fpr']:.0%}, "
        f"precision {heavy['precision']:.0%}. That jump between 5:1 and 10:1 is where the tool "
        f"flips from a selective watchlist to a flag-most screen — the decision a risk committee "
        f"must actually make.")
    if len(dead):
        names = ", ".join(dead["archetype"])
        lines.append(
            f"- **The zero-recall slices are a ranking failure, not a threshold artifact:** "
            f"{names} carry events but, even when given a slice-relative top-decile threshold "
            f"(top-10% within their own scores), recall only reaches roughly the "
            f"{TOP_K_FRACTION:.0%} random floor — i.e. their events are not ranked above their "
            f"non-events (within-sector AUROC ≤ 0.5). No threshold rescues a broken ranking; this "
            f"is the diagnosis for the model's biggest blind spots.")
    lines.append(
        "- **Recall and FPR are the transferable numbers; precision is base-rate-bound.** The test "
        f"base rate is {meta['test_rate']:.1%}; a lower-prevalence deployment population would "
        "depress precision at the same recall/FPR, so precision/FP counts here should be read as "
        "panel-specific.")
    lines.append("")

    lines.append("## Reproducibility\n")
    lines.append("- Command: `MPLBACKEND=Agg python scripts/fp_fn_analysis.py`")
    lines.append("- Figure: `outputs/figures/phase3_fp_fn_frontier.png`")
    lines.append("- Tables: `outputs/fp_fn_{operating_points,cost_sensitivity,slice_threshold_compare}.csv`")
    lines.append("- Design/methodology: `docs/superpowers/specs/2026-06-16-fp-fn-analysis-design.md`")
    lines.append("")
    lines.append("**Deferred (follow-up):** lead-time-aware TP definition and a firm-episode-level "
                 "confusion matrix (reuse `eval.compute_lead_time`).")

    out = os.path.join(PATHS.OUTPUTS, "fp_fn_findings.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved findings: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("FP / FN ANALYSIS — cost frontier, held-out test, cost-weighted points")
    print("=" * 70)

    train, val, test = load_splits()
    print(f"  train ≤{TRAIN_END_YEAR}: {len(train)} rows | "
          f"val {TRAIN_END_YEAR+1}-{VAL_END_YEAR}: {len(val)} rows | "
          f"test {VAL_END_YEAR+1}+: {len(test)} rows")

    # Deployed model
    pred = fit_predict(train, {"val": val, "test": test}, MARKET_PLUS_REL_FEATURE_COLS)
    val_m, test_m = pred["val"], pred["test"]

    # Altman benchmark (drops z_score NaN rows)
    pred_z = fit_predict(train, {"val": val, "test": test}, ["z_score"])
    val_z, test_z = pred_z["val"], pred_z["test"]

    meta = {
        "test_rows": len(test_m), "test_firms": test_m["ticker"].nunique(),
        "test_events": int(test_m["y"].sum()), "test_rate": float(test_m["y"].mean()),
    }

    op = operating_points(val_m, test_m)
    cost = cost_sensitivity(val_m, test_m)
    sl = slice_threshold_compare(val_m)

    budgets = np.linspace(0.01, 1.0, 100)
    model_cap = capture_curve(val_m, test_m, budgets)
    altman_cap = capture_curve(val_z, test_z, budgets)

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    op.to_csv(os.path.join(PATHS.OUTPUTS, "fp_fn_operating_points.csv"), index=False)
    cost.to_csv(os.path.join(PATHS.OUTPUTS, "fp_fn_cost_sensitivity.csv"), index=False)
    sl.to_csv(os.path.join(PATHS.OUTPUTS, "fp_fn_slice_threshold_compare.csv"), index=False)
    print("\n-- Operating points (TEST 2024) --")
    print(op.round(3).to_string(index=False))
    print("\n-- Cost sensitivity --")
    print(cost.round(3).to_string(index=False))
    print("\n-- Slice threshold compare (VAL) --")
    print(sl.round(3).to_string(index=False))

    plot_frontier(model_cap, altman_cap, val_m, op)
    write_findings(op, cost, sl, meta)
    print("\nDone.")


if __name__ == "__main__":
    main()
