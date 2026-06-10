"""
Charts: ROC/PR, calibration, risk deciles, firm risk trajectories.

All four render into PATHS.FIGURES (= outputs/figures/). This is the
intentional behavior change from the Phase 1 monolith, which wrote charts
next to the script in src/. Flag this to the team when you ship the refactor.

Trajectory chart owns its own prediction generation — the caller passes the
fitted pooled model + FEATURE_COLS, and the chart handles train/val/test
stitching internally. This is cleaner than the monolith, which stitched
predictions at the call site inside main().
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import FEATURE_COLS, FIRMS, LABEL_COL, PATHS


def _save(fig, filename: str) -> None:
    """Save a figure to PATHS.FIGURES, creating the dir if needed."""
    os.makedirs(PATHS.FIGURES, exist_ok=True)
    path = os.path.join(PATHS.FIGURES, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# =============================================================================
# ROC + PR
# =============================================================================

def plot_roc_pr(y_true: pd.Series, preds_dict: dict[str, pd.Series], filename: str) -> None:
    """Side-by-side ROC and PR curves for up to 3 models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#3b82f6", "#ef4444", "#22c55e"]

    for (name, y_pred), color in zip(preds_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        ax2.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.axhline(y=y_true.mean(), color="k", linestyle="--", alpha=0.4)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, filename)


# =============================================================================
# Calibration reliability + histogram
# =============================================================================

def plot_calibration(y_true: pd.Series, y_pred: pd.Series, model_name: str, filename: str) -> None:
    """Reliability curve + predicted-probability histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=8, strategy="uniform")
    ax1.plot(prob_pred, prob_true, "s-", color="#3b82f6", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="Predicted Probability", ylabel="Actual Event Rate",
            title=f"Calibration — {model_name}")
    ax1.grid(True, alpha=0.3)

    ax2.hist(y_pred, bins=40, color="#3b82f6", alpha=0.7, edgecolor="white")
    ax2.set(xlabel="Predicted Probability", ylabel="Count", title="Prediction Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, filename)


def plot_calibration_comparison(y_true, preds_by_method: dict, filename: str) -> None:
    """Reliability curves for raw vs Platt vs isotonic predictions on one axis.

    `preds_by_method` maps method-name -> predicted-probability array. A curve
    hugging the diagonal is well calibrated.
    """
    colors = {"raw": "#94a3b8", "platt": "#3b82f6", "isotonic": "#ef4444"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in preds_by_method.items():
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", linewidth=2, label=name,
                color=colors.get(name))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set(xlabel="Predicted probability", ylabel="Actual event rate",
           title="Calibration: raw vs Platt vs isotonic (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, filename)


# =============================================================================
# Risk decile bars
# =============================================================================

def plot_risk_deciles(y_true: pd.Series, y_pred: pd.Series, model_name: str, filename: str) -> None:
    """Per-decile actual event rate (the analyst-utility chart)."""
    df = pd.DataFrame({"y": y_true.values, "pred": y_pred})
    n_bins = min(10, df["pred"].nunique())
    if n_bins < 3:
        print(f"  Skipping decile chart (not enough unique predictions)")
        return

    df["decile"] = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop") + 1
    stats = df.groupby("decile").agg(count=("y", "count"), events=("y", "sum"), avg_pred=("pred", "mean"))
    stats["event_rate"] = stats["events"] / stats["count"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(stats.index, stats["event_rate"], color="#3b82f6", edgecolor="white", alpha=0.85)
    ax.axhline(y=y_true.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Overall rate ({y_true.mean():.1%})")
    ax.set(xlabel="Risk Decile (1=safest → highest=riskiest)", ylabel="Actual Event Rate",
           title=f"Risk-Decile Chart — {model_name}")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, stats["event_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, filename)


# =============================================================================
# Firm risk trajectories (the "money chart")
# =============================================================================

def plot_firm_risk_trajectory(
    panel: pd.DataFrame,
    model,
    model_name: str,
    filename: str,
) -> None:
    """Per-firm predicted-risk-over-time subplot grid.

    Caller passes the fitted pooled logit model; this function generates
    predictions across the whole panel internally — no external list-concat
    required (cleaner than the monolith's call site).
    """
    # Predict over the whole panel using the pooled-logit feature set.
    # (For Phase 1, trajectories always use the pooled model per the
    # monolith's main(); that stays the same.)
    X = sm.add_constant(panel[FEATURE_COLS])
    preds = model.predict(X)

    panel_plot = panel.copy().reset_index(drop=True)
    panel_plot["pred"] = preds.values if hasattr(preds, "values") else preds

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
    axes = axes.flatten()

    for idx, (ticker, group) in enumerate(panel_plot.groupby("ticker")):
        if idx >= 10:
            break
        ax = axes[idx]
        group = group.sort_values("date")

        ax.plot(group["date"], group["pred"], color="#3b82f6", linewidth=1.5, label="Predicted risk")

        events = group[group[LABEL_COL] == 1]
        for _, ev in events.iterrows():
            ax.axvline(ev["date"], color="#ef4444", alpha=0.15, linewidth=8)

        ax.axhline(y=0.3, color="#f59e0b", linestyle="--", alpha=0.6, linewidth=1)

        name = FIRMS.get(ticker, {}).get("name", ticker)
        ax.set_title(f"{ticker} ({name})", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7, axis="x", rotation=45)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle(
        f"Firm Risk Trajectories — {model_name}\n"
        f"(Blue = predicted risk | Red shading = actual event | Orange dashed = alert threshold)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    _save(fig, filename)
