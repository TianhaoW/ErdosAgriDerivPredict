import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from pyproj import Transformer
from src.utils import parse_config

config = parse_config()
us_map_path = config['data']['us_map_path']
raw_cdl_path = config['data']['raw_cdl_path']
downscale_factor = config['data']['downscale_factor']

def inspect_weather_stations_with_map(
        cdl_npy_path: str,
        focus_state: str,
        station_coords: list[tuple[float, float]],
        station_colors: list[float] = None):
    '''
    :param cdl_npy_path: the file address of the crop land layer. If None, the cropland will not be displayed
    :param focus_state: The two digit state code. E.g. 'CA'
    :param station_coords: a list of tuples of (lon, lat), representing the weather station coordinates
    :param station_colors: a list of floats representing the weather station colors (usually use z-score to represent anomaly)
    :return:
    '''

    # Load downscaled .npy file
    arr = np.load(cdl_npy_path)

    # Load original .tif to get original transform and CRS
    with rasterio.open(raw_cdl_path) as src:
        orig_transform = src.transform
        crs = src.crs

    # Project weather station coords to CRS of the raster
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected_coords = [transformer.transform(lon, lat) for lon, lat in station_coords]
    xs, ys = zip(*projected_coords) if projected_coords else ([], [])


    # Adjust transform for downscaled resolution
    new_transform = Affine(
        orig_transform.a * downscale_factor,  # pixel width
        orig_transform.b,
        orig_transform.c,
        orig_transform.d,
        orig_transform.e * downscale_factor,  # pixel height
        orig_transform.f
    )

    # Calculate extent for imshow
    extent = (
        new_transform.c,
        new_transform.c + arr.shape[1] * new_transform.a,
        new_transform.f + arr.shape[0] * new_transform.e,
        new_transform.f
    )

    # Load US states shapefile and convert CRS
    us_map = gpd.read_file(us_map_path)
    us_map = us_map.to_crs(crs)

    state_map = us_map[us_map['STUSPS'] == focus_state]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(arr, cmap='Greens', extent=extent, origin='upper')
    state_map.boundary.plot(ax=ax, color='black', linewidth=1)
    plt.colorbar(im, ax=ax, label='Crop density', shrink=0.5)

    # Plot weather stations
    if station_colors:
        sc = ax.scatter(xs, ys, c=station_colors, cmap='coolwarm', s=60, edgecolor='k', label='Stations', alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Z-score', shrink=0.5)
    else:
        ax.scatter(xs, ys, color='red', s=60, edgecolor='k', label='Stations', alpha=0.7)

    ax.set_title(f"Weather stations with cropland and US Map Overlay in {focus_state}")
    xmin, ymin, xmax, ymax = state_map.total_bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.tight_layout()
    plt.show()

# sample usage
def compute_seasonal_zscore(df, station_col="station_name", date_col="date", value_col="tavg", threshold=3.0):
    """
    Compute Z-scores normalized by day-of-year for each station.
    Flags anomalies based on deviation from mean per day-of-year.

    Returns:
        df with an added column: f"{value_col}_anomaly"
    """
    df = df.copy()
    df["day_of_year"] = pd.to_datetime(df[date_col]).dt.dayofyear

    # Group by station and day-of-year
    grouped = df.groupby([station_col, "day_of_year"])[value_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std")

    z = (df[value_col] - mean) / std
    df[f"{value_col}_zscore"] = z
    df[f"{value_col}_anomaly"] = z.abs() > threshold

    return df

focus_state = 'KS'
commodity = 'WHEAT_WINTER'
date = '2021-03-15'
weather = pd.read_csv(f'./dataset/raw/weather_data/{focus_state}_weather.csv')
for col in ['tmax', 'tmin', 'tavg', 'precip', 'snow']:
    weather[col] = pd.to_numeric(weather[col], errors='coerce')
result = compute_seasonal_zscore(weather)
stations_df = result[result['date']==date]

stations = [tuple(row) for row in stations_df[['lon', 'lat']].to_numpy()]
stations_color = stations_df['tavg_zscore'].abs().to_list()

inspect_weather_stations_with_map(
    f'./dataset/processed/cdl/{commodity}.npy',
    focus_state,
    stations,
    stations_color,
)