from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import DATA_DIR, OUTPUT_DIR
from core.data_pipeline import build_monthly_panel, load_macro, load_masterlist, load_price_cache, refresh_prices
from core.label_builder import add_label_a
from core.models import run_fixed_models, run_rolling_models
from core.sec_label_b import attach_label_b, case_validation, fetch_label_b_from_sec, load_cached_label_b
from core.utils import step, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--refresh-sec", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master = step("Loading data", load_masterlist, DATA_DIR)
    prices = step("Processing prices", refresh_prices if args.refresh_prices else load_price_cache, master) if args.refresh_prices else step("Processing prices", load_price_cache)
    macro = step("Loading macro data", load_macro, DATA_DIR)
    panel = step("Building panel", build_monthly_panel, master, prices, macro)
    panel = step("Building Label A", add_label_a, panel)
    coverage, events = step("Building Label B", fetch_label_b_from_sec if args.refresh_sec else load_cached_label_b, master) if args.refresh_sec else step("Building Label B", load_cached_label_b)
    panel = step("Merging labels", attach_label_b, panel, events)
    perf, ci, fixed_preds = step("Running fixed-test models", run_fixed_models, panel)
    rolling_perf, rolling_preds = step("Running rolling-origin models", run_rolling_models, panel)
    validation = step("Validating Label B cases", case_validation, events, rolling_preds)

    step("Saving outputs", write_csv, master, OUTPUT_DIR / "ticker_masterlist_100.csv")
    write_csv(panel, OUTPUT_DIR / "panel_100_label_a_label_b.csv")
    write_csv(coverage, OUTPUT_DIR / "label_b_coverage_100.csv")
    write_csv(events, OUTPUT_DIR / "label_b_events_100.csv")
    write_csv(validation, OUTPUT_DIR / "label_b_case_validation_100.csv")
    write_csv(perf, OUTPUT_DIR / "model_performance_pooled.csv")
    write_csv(ci, OUTPUT_DIR / "model_performance_block_bootstrap_ci.csv")
    write_csv(fixed_preds, OUTPUT_DIR / "model_predictions_fixed_test.csv")
    write_csv(rolling_perf, OUTPUT_DIR / "rolling_origin_performance.csv")
    write_csv(rolling_preds, OUTPUT_DIR / "model_predictions_rolling.csv")
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
