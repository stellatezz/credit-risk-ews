from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import DATA_DIR, OUTPUT_DIR
from core.utils import step, write_csv
from core.v211 import run_v211


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUTPUT_DIR / "panel_100_label_a_label_b.csv"
    if not panel_path.exists():
        panel_path = DATA_DIR / "panel_100_label_a_label_b.csv"
    panel = step("Loading data", pd.read_csv, panel_path, low_memory=False, parse_dates=["date"])
    perf, main_perf, preds, calibration, deciles = step("Running models", run_v211, panel)
    step("Saving outputs", write_csv, main_perf, OUTPUT_DIR / "v211_fe_hazard_model_performance.csv")
    write_csv(perf, OUTPUT_DIR / "label_a_abs_threshold_robustness_30_40_50.csv")
    write_csv(preds, OUTPUT_DIR / "v211_predictions_fixed_test.csv")
    write_csv(calibration, OUTPUT_DIR / "calibration_diagnostics_v211.csv")
    write_csv(deciles, OUTPUT_DIR / "calibration_deciles_v211.csv")
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
