import pandas as pd
import yfinance as yf
import rasterio
import numpy as np
import json
import pickle
import rasterio
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from src.data.preprocess import extend_market_data
from src.data import DataLoader
from src.utils import parse_config
from src.constants import COMMODITY_TO_CROP_VALUE, STATE_NAME_TO_ABBR
from src.data.weather_util import (
    apply_convolution,
    get_station_metadata,
    get_climate_data,
    latlon_to_pixel,         # assumes (transform, crs, lat, lon, downscale_factor)
    get_k_nearest_crop_areas # assumes (arr, k, station_dict, rmax)
)


config = parse_config()
project_root = Path(config['path']['project_root']).resolve()
symbols = config['data']['symbols']
commodities = config['data']['commodities']
reports = config['data']['reports']
levels = config['data']['levels']

start_date_default = datetime.strptime(config['data']['start_date_default'], "%Y-%m-%d").date()
start_year_default = config['data']['start_year_default']
today = datetime.today().date()

def update_market_data():
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

        # create the directory if it does not exist
        processed_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_extended = extend_market_data(df_combined, symbol[:2])
        df_extended.to_csv(processed_file_path, index_label="Date")

        # create the directory if it does not exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(file_path, index_label="Date")
        print(f"{symbol}: CSV updated with {len(df_new)} new rows.")

def update_USDA_reports():
    dl = DataLoader()
    for commodity in commodities:
        for report in reports:
            for level in levels:
                existing_total_rows = 0
                if (
                    config.get("usda_data", {})
                          .get(commodity, {})
                          .get(report, {})
                          .get(level) is None
                ):
                    print(f"{commodity} - {level} level {report}: No config found, skipping.")
                    continue

                params = config['usda_data'][commodity][report][level]['filters']
                columns = config['usda_data'][commodity][report][level]['columns']

                file_path = project_root / "dataset" / "raw" / "USDA_reports" /f"{report}" / f"{level}" / f"{commodity}_{report}.csv"
                file_exists = file_path.exists()

                if file_exists:
                    df_existing = pd.read_csv(file_path)
                    existing_total_rows = len(df_existing)

                    if "year" not in df_existing.columns:
                        print(f"{commodity} - {level} - {report}: No 'year' column in existing data! Please delete the csv file and download again.")
                        continue
                    last_year = df_existing['year'].max()

                    # Download starting from last_year - 1. We remove duplication later when merging the data
                    start_year = max(last_year - 1, start_year_default)
                else:
                    print(f"{commodity} - {level} - {report}: File not found, downloading from default start year.")
                    start_year = start_year_default

                df_new = dl.get_usda_data(commodity, start_year, params)
                if df_new.empty:
                    print(f"{commodity} - {level} - {report}: No new data found.")
                    continue

                df_new = df_new[columns]

                if file_exists:
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    # Drop exact duplicates rows
                    duplicate_check_columns = [col for col in columns if col != 'load_time']
                    df_combined = df_combined.drop_duplicates(subset=duplicate_check_columns)
                else:
                    df_combined = df_new

                file_path.parent.mkdir(parents=True, exist_ok=True)
                df_combined.to_csv(file_path, index=False)
                print(f"{commodity} - {level} - {report}: Updated with {len(df_combined)-existing_total_rows} new rows (after deduplication).")

def generate_top_production_states(output_path=None, top_k=10):
    # This will generate the top-k production states of the corps listed in the config.toml file using the last year production data
    report = "production"
    level = "state"
    result = {}

    if output_path is None:
        output_path = project_root / "dataset" / "top_production_states.json"

    if output_path.exists():
        print("[WARN]: the top production states file already exists. If you want to overwrite it, delete the existing file.")
        return

    for commodity in commodities:
        file_path = (
            project_root / "dataset" / "raw" / "USDA_reports" /
            report / level / f"{commodity}_{report}.csv"
        )

        if not file_path.exists():
            print(f"[WARN] {commodity}: state-level production report not found at {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            if df['unit_desc'].unique() == ['PCT BY TYPE']:
                continue

            df = df[df["reference_period_desc"] == "YEAR"]
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
            last_year = df["year"].max() - 1

            production_data = df[df["year"] == last_year]
            total_production = production_data["Value"].sum()

            top_states = (
                production_data
                .sort_values(by="Value", ascending=False)
                .head(top_k)
                .loc[:, ["state_name", "Value"]]
                .dropna()
            )

            formatted = [
                {
                    "state": row["state_name"],
                    "percentage": round(row["Value"] / total_production, 4)
                }
                for _, row in top_states.iterrows()
            ]

            result[commodity.upper()] = formatted

        except Exception as e:
            print(f"[ERROR] Failed to process {commodity}: {e}")



    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] Top production states saved to {output_path}")

def downscale_cdl():
    src_path = config['data']['raw_cdl_path']
    downscale_factor = config['data']['downscale_factor']
    crops = config['data']['crops']

    output_dir = project_root / "dataset" / "processed" / "cdl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        print(f"[INFO] Raw sdl file loaded successfully.")
        arr = src.read(1)

        for crop in crops:
            if crop not in COMMODITY_TO_CROP_VALUE:
                print(f"[WARN] Skipping unknown crop: {crop}")
                continue

            cropValue = COMMODITY_TO_CROP_VALUE[crop]
            save_path = output_dir / f"{crop}.npy"

            if save_path.exists():
                print(f"[WARN] downscaled cdl already exists at {save_path}. Please remove it if you want to overwrite.")
                continue

            try:
                arr_downscaled = apply_convolution(
                    arr,
                    kernel=np.ones((downscale_factor, downscale_factor)),
                    cropValue=cropValue
                )
                np.save(save_path, arr_downscaled)
                print(f"[INFO] Saved downscaled array for {crop} to {save_path}")
            except Exception as e:
                print(f"[ERROR] Failed to process {crop}: {e}")

def generate_k_stations_near_corpland():
    # --- Configurations ---
    crops = config['data']['crops']
    # This is the default output path of the previous function
    top_state_file = project_root / "dataset" / "top_production_states.json"
    cdl_path = config['data']['raw_cdl_path']
    downscale_factor = config['data']['downscale_factor']
    k = 5
    rmax = 9999

    # --- Step 1: Load top states ---
    with open(top_state_file, "r") as f:
        top_states = json.load(f)

    for crop in crops:
        downscaled_cdl_path = project_root / "dataset" / "processed" / "cdl" / f"{crop}.npy"
        output_path = project_root / "dataset" / "processed" / "weather_mapping" /f"{crop}_station_to_crop.pkl"

        if not downscaled_cdl_path.exists():
            print(f"[WARN] Downscaled cdl file not found for {crop}.]")
            continue

        if output_path.exists():
            print(f"[WARN] the file {output_path} already exists. Please remove it first if you want to overwrite.")
            continue

        if crop not in top_states:
            raise ValueError(f"No top states found for crop {crop}")
        states = [entry["state"] for entry in top_states[crop]]

        # --- Step 2: Get all weather stations from those states ---
        station_latlon_set = set()
        for state in states:
            station_meta = get_station_metadata(STATE_NAME_TO_ABBR.get(state, ""), "2024-01-01", "2024-01-31")
            for item in station_meta["meta"]:
                ll = item.get("ll")
                if ll:
                    station_latlon_set.add(tuple(ll))

        # --- Step 3: Map (lat, lon) → (row, col) ---
        with rasterio.open(cdl_path) as src:
            transform = src.transform
            crs = src.crs

        station_dict = {}
        for lat, lon in station_latlon_set:
            try:
                row, col = latlon_to_pixel(transform, crs, lat, lon, downscale_factor)
                station_dict[(row, col)] = (lon, lat)
            except Exception:
                continue  # skip out-of-bound stations

        # --- Step 4: Load downscaled CDL ---
        arr = np.load(downscaled_cdl_path)

        # --- Step 5: Compute k-nearest stations ---
        mapping = get_k_nearest_crop_areas(arr, k=k, station_dict=station_dict, rmax=rmax)

        # --- Step 6: Save result as JSON ---
        pickle_data = defaultdict(list)
        for area, station_list in mapping:
            for dist, (lon, lat) in station_list:
                key = str((lon, lat))
                pickle_data[key].append((area, dist))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(pickle_data, f)

        print(f"[INFO] Weather mapping saved to {output_path}")



def update_weather_data():
    state_file = project_root / "dataset" / "top_production_states.json"
    if not state_file.exists():
        print(f"[WARN] the top production states file not found.")
        return

    state_set = set()
    with open(state_file, "r") as f:
        data = json.load(f)
    for crop_data in data.values():
        for entry in crop_data:
            state = entry["state"]
            state_set.add(state)

    for state in state_set:
        state_abbr = STATE_NAME_TO_ABBR.get(state)
        file_path = project_root / "dataset" / "raw" / "weather_data" / f"{state_abbr}_weather.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            df_existing = pd.read_csv(file_path, parse_dates=["date"])
            last_date = df_existing["date"].max().date()
            start_date = last_date - timedelta(days=1)
        else:
            df_existing = None
            start_date = start_date_default

        if start_date > today:
            print(f"[INFO] {state_abbr}: Already up to date.")
            return

        # Fetch new data
        end_date = today.isoformat()
        data = get_climate_data(state_abbr, start_date.isoformat(), end_date)
        if not data or "data" not in data:
            print(f"[WARN] No data returned for {state_abbr}")
            return

        rows = []
        for station_entry in data["data"]:
            lon, lat = station_entry["meta"].get("ll", (None, None))
            station_name = station_entry["meta"]["name"]
            date = start_date_default
            for entry in station_entry["data"]:
                date_str = date.isoformat()
                values = entry
                rows.append({
                    "date": date_str,
                    "station_name": station_name,
                    "lon": lon,
                    "lat": lat,
                    "tmax": values[0],
                    "tmin": values[1],
                    "tavg": values[2],
                    "precip": values[3],
                    "snow": values[4],
                })
                date = date + timedelta(days=1)
        df_new = pd.DataFrame(rows)
        df_new["date"] = pd.to_datetime(df_new["date"])
        df_new = df_new.dropna(subset=["tavg", "tmax", "tmin"], how="all")  # optional cleanup

        if df_existing is not None:
            df_combined = pd.concat([df_existing, df_new])
            df_combined = df_combined.drop_duplicates(subset=["date", "station_name"], keep="last")
        else:
            df_combined = df_new

        df_combined = df_combined.sort_values(by=["date", "station_name"])
        df_combined.to_csv(file_path, index=False)
        print(f"[INFO] {state_abbr}: Weather data updated to {file_path}")

if __name__ == "__main__":
    update_market_data()
    update_USDA_reports()
    generate_top_production_states()
    downscale_cdl()
    generate_k_stations_near_corpland()
    update_weather_data()
