import pandas as pd
from src.ews.labels import compute_label_a_from_prices

PRICES_CSV = "data/interim/prices.csv"

print("Loading prices (select columns)...")
# the prices.csv begins with a comment line; tell pandas to ignore comment lines
prices = pd.read_csv(PRICES_CSV, usecols=["ticker", "date", "adj_close"], parse_dates=["date"], comment="#")
print(f"Prices rows: {len(prices)}\nColumns: {prices.columns.tolist()}")

for h in (3, 6, 12):
    print(f"\nComputing labels for horizon={h} months...")
    try:
        lbl = compute_label_a_from_prices(prices, horizon_months=h, threshold=-0.40)
        if lbl.empty:
            print("  -> No labels produced (forward-window too short for this horizon)")
        else:
            print(f"Rows: {len(lbl)} | Events: {int(lbl['label_a'].sum())} | Event rate: {lbl['label_a'].mean():.3%}")
    except Exception as e:
        print(f"  -> Error computing labels for horizon={h}: {e}")

print('\nDone')

