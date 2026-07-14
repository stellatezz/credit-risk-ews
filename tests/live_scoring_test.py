"""Offline smoke test for src/ews/scoring.py (no network required).

Verifies that the live-scoring engine reproduces the deployed model:
  1. Coefficients load and cover every FEATURE_COL.
  2. Scoring the committed panel with the committed coefficients reproduces
     the pooled model's known validation AUROC (~0.60).
  3. Driver contributions decompose the logit exactly.
  4. Percentile/decile mapping is monotone and bounded.

Run:  PYTHONPATH=src:. .venv/bin/python tests/live_scoring_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ews import scoring
from ews.config import FEATURE_COLS, TRAIN_END_YEAR, VAL_END_YEAR


def main() -> None:
    coefs = scoring.load_coefficients()
    assert "const" in coefs.index
    assert all(f in coefs.index for f in FEATURE_COLS)
    print(f"[OK] coefficients: const + {len(FEATURE_COLS)} features")

    panel = scoring.load_scored_panel()
    assert panel["pd_score"].between(0, 1).all()
    print(f"[OK] panel scored: {len(panel)} rows, "
          f"PD range {panel['pd_score'].min():.3f}-{panel['pd_score'].max():.3f}")

    # Known-answer test: pooled val AUROC ~0.603 (README / phase2 run).
    from sklearn.metrics import roc_auc_score
    val = panel[(panel["year"] > TRAIN_END_YEAR) & (panel["year"] <= VAL_END_YEAR)]
    auroc = roc_auc_score(val["label_a"], val["pd_score"])
    assert 0.55 < auroc < 0.70, f"val AUROC {auroc:.3f} outside expected band"
    print(f"[OK] val AUROC = {auroc:.3f} (expected ~0.603)")

    # Driver decomposition: const + sum(coef*median) + sum(contributions)
    # must equal the row's logit exactly.
    ref = scoring.build_reference(panel)
    row = panel.iloc[-1]
    tbl = scoring.driver_table(row, coefs, ref)
    base_logit = coefs["const"] + sum(
        coefs[f] * ref["feature_medians"][f] for f in FEATURE_COLS)
    reconstructed = 1 / (1 + np.exp(-(base_logit + tbl["contribution"].sum())))
    assert abs(reconstructed - row["pd_score"]) < 1e-9
    chips = scoring.top_drivers(row, coefs, ref)
    assert chips and "systemic" not in chips
    print(f"[OK] driver decomposition exact; top drivers for {row['ticker']}: {chips}")

    # Percentile / decile sanity.
    pcts = scoring.pd_percentile([0.0, 0.1, 0.9], ref)
    assert (np.diff(pcts) >= 0).all() and pcts[0] == 0 and pcts[-1] <= 100
    decs = scoring.pd_decile(panel["pd_score"].to_numpy(), ref)
    assert decs.min() >= 1 and decs.max() <= 10
    top = panel[decs == 10]
    lift = top["label_a"].mean() / panel["label_a"].mean()
    print(f"[OK] deciles 1-10; top-decile lift = {lift:.2f}x (expect ~3x)")

    ops = scoring.load_operating_points()
    assert len(ops) >= 2 and ops["thr"].between(0, 1).all()
    print(f"[OK] operating points: {list(ops['label'])}")

    print("\nAll live-scoring checks passed.")


if __name__ == "__main__":
    main()
