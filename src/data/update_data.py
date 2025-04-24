import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from src.data.preprocess import extend_market_data
from src.utils import parse_config

config = parse_config()
project_root = Path(config['path']['project_root']).resolve()
symbols = config['data']['symbols']
start_date_default = datetime.strptime(config['data']['start_date'], "%Y-%m-%d").date()

def update_market_data():
    today = datetime.today().date()

    for symbol in symbols:
        file_path = project_root / "dataset" / "raw" / "market_data" / f"{symbol[:2]}.csv"
        processed_file_path = project_root / "dataset" / "processed" / "market_data" /f"{symbol[:2]}.csv"
        file_exists = file_path.exists()

        # Load existing data
        if file_exists:
            df_existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
            last_date = df_existing.index[-1].date()
            start_date = last_date + timedelta(days=1)
        else:
            print(f"{symbol}: File not found, downloading from {start_date_default}")
            start_date = start_date_default

        if start_date > today:
            print(f"{symbol}: Already up to date.")
            continue

        # Download new data
        df_new = (
            yf.Ticker(symbol)
            .history(start=start_date, end=today + timedelta(days=1))
            .drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        )

        df_new.index = df_new.index.date
        df_new = df_new[df_new.index >= start_date]

        if df_new.empty:
            print(f"{symbol}: No new data available.")
            continue

        # Format and append
        if file_exists:
            df_existing.index = df_existing.index.date  # Make sure both are same format
            df_combined = pd.concat([df_existing, df_new])
        else:
            df_combined = df_new

        df_extended = extend_market_data(df_combined, symbol[:2])
        df_extended.to_csv(processed_file_path, index_label="Date")

        df_combined.to_csv(file_path, index_label="Date")
        print(f"{symbol}: CSV updated with {len(df_new)} new rows.")

# TODO, bug, sometimes

if __name__ == "__main__":
    update_market_data()
