"""Pre-warm the live-watchlist caches so the on-stage demo loads instantly.

Fetches the price gap since the repo cache, reuses/downloads EDGAR filings,
scores every firm (panel + analyst-added), and writes the same disk caches
the dashboard reads (data/interim/live/). Run this before presenting:

    .venv/bin/python scripts/warm_live_cache.py [--force]

--force refetches prices even if today's cache exists.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ews import scoring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="refetch prices even if today's cache exists")
    args = parser.parse_args()

    t0 = time.time()
    print("Warming live watchlist caches (prices → EDGAR → scores)…")
    scores, status = scoring.build_live_scores(force=args.force)

    live = scores[scores["is_live"]]
    latest = scores.sort_values("date").groupby("ticker").tail(1)
    top = latest.sort_values("pd_score", ascending=False).head(10)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"  Firms scored:      {scores['ticker'].nunique()}")
    print(f"  Live firm-months:  {len(live)} "
          f"({live['date'].min():%Y-%m} → {live['date'].max():%Y-%m})"
          if len(live) else "  Live firm-months:  0 (!)")
    print(f"  Prices fresh thru: {status.get('fresh_through')}"
          f"{'  [OFFLINE — cached only]' if status.get('offline') else ''}")
    imputed = latest[latest["fundamentals_imputed"] == True]  # noqa: E712
    if len(imputed):
        print(f"  Fundamentals imputed for: {', '.join(sorted(imputed['ticker']))}")

    print("\nTop 10 by current PD:")
    for i, r in enumerate(top.itertuples(), 1):
        print(f"  {i:>2}. {r.ticker:<6} PD={r.pd_score:.1%}  (as of {r.date:%Y-%m})")

    print("\nCache ready — the dashboard will load instantly today.")


if __name__ == "__main__":
    main()
