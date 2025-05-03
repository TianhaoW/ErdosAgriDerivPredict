import heapq
import requests
import numpy as np
import statsmodels.api as sm
from rasterio.transform import rowcol, xy
from pyproj import Transformer
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Generator


def apply_convolution(matrix: np.ndarray, kernel: np.ndarray, crop_value: int) -> np.ndarray:
    """
    Apply a convolution mask to identify crop coverage in downscaled resolution.
    :param matrix: The loaded tif file matrix
    :param kernel: The kernel for convolution
    :param crop_value: Please use the src/constants/crop_codes to get the crop value
    """
    m, n = matrix.shape
    km, kn = kernel.shape
    out_m = m // km
    out_n = n // kn

    output = np.zeros((out_m, out_n), dtype=np.uint8)

    for i in range(out_m):
        for j in range(out_n):
            patch = matrix[i*km : i*km + km, j*kn : j*kn + kn]
            mask = (patch == crop_value)
            output[i, j] = np.sum(mask * kernel)

    return output


def latlon_to_pixel(transform, crs, lat, lon, downscale_factor) -> tuple[int, int]:
    """
    Convert (lat, lon) → downscaled (row, col) pixel index.
    :param transform: The affine transform. This can be obtained from the raw tif file
    :param crs: The Coordinate Reference System (CRS). This can be obtained from the raw tif file
    :param lat: Latitude
    :param lon: Longitude
    :param downscale_factor: Downscale factor
    :return: (row, col) pixel index in the downscaled resolution
    """
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_proj, y_proj = transformer.transform(lon, lat)
    row, col = rowcol(transform, x_proj, y_proj)
    return int(row // downscale_factor), int(col // downscale_factor)


def pixel_to_latlon(transform, crs, row, col, downscale_factor) -> tuple[float, float]:
    """
    Convert downscaled pixel index (row, col) back to (lat, lon).
    :param transform: The affine transform. This can be obtained from the raw tif file
    :param crs: The Coordinate Reference System (CRS). This can be obtained from the raw tif file
    :param row: the row index in the downscaled resolution
    :param col: the column index in the downscaled resolution
    :param downscale_factor: Downscale factor when converting the raw tif file to downscaled resolution
    :return: the latitude and longitude of the pixel index in the downscaled resolution
    """
    x_proj, y_proj = xy(transform, row * downscale_factor, col * downscale_factor)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x_proj, y_proj)
    return lat, lon

def find_integer_pairs_in_radius(r: int) -> list[tuple[int, int]]:
    """
    Return grid index offsets (i, j) where r <= sqrt(i^2 + j^2) < r+1.
    """
    result = set()
    for x in range(-r - 1, r + 2):
        if abs(x) > r + 1:
            continue
        min_y2 = max(0, r**2 - x**2)
        max_y2 = (r + 1)**2 - x**2
        if min_y2 > max_y2:
            continue
        min_y = int(np.sqrt(min_y2))
        max_y = int(np.sqrt(max_y2)) + 1
        for y in range(min_y, max_y + 1):
            d2 = x**2 + y**2
            if r**2 <= d2 < (r + 1)**2:
                result.update({(x, y), (x, -y), (-x, y), (-x, -y)})
    return list(result)


def daterange(start: date, end: date) -> Generator[date, None, None]:
    """Generate dates from start to end (exclusive)."""
    for n in range((end - start).days):
        yield start + timedelta(n)


def get_station_metadata(state: str, start_date: str, end_date: str) -> dict:
    '''
    This will return the metadata for the weather station in a given state
    :param state: 2 letter abbreviation of a state. E.g. "CA"
    :param start_date: start date in "YYYY-MM-DD" format
    :param end_date: end date in "YYYY-MM-DD" format
    :return: The metadata as a dictionary
    '''
    url = "http://data.rcc-acis.org/StnMeta"
    params = {
        "state": state,
        "sdate": start_date,
        "edate": end_date,
        "elems": ["avgt"],
        "output": "json",
    }
    response = requests.post(url, json=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Failed to get station metadata: {response.status_code}")


def get_climate_data(state: str, start_date: str, end_date: str) -> dict:
    '''
    This will return the climate data for a given state
    :param state: 2 letter abbreviation of a state. E.g. "CA"
    :param start_date: start date in "YYYY-MM-DD" format
    :param end_date: end date in "YYYY-MM-DD" format
    :return: The data key contains [maxt, mint, avgt, pcpn, snow]
    '''
    station_ids = [
        entry["sids"][0]
        for entry in get_station_metadata(state, start_date, end_date)["meta"]
    ]
    url = "http://data.rcc-acis.org/MultiStnData"
    params = {
        "sids": station_ids,
        "sdate": start_date,
        "edate": end_date,
        "elems": [{"name": "maxt", "units": "degreeC"},
                  {"name": "mint", "units": "degreeC"},
                  {"name": "avgt", "units": "degreeC"},
                  {"name": "pcpn", "units": "mm"},
                  {"name": "snow"}],
        "output": "json",
    }
    response = requests.post(url, json=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Failed to get climate data: {response.status_code}")


def climate_data_to_dict(climate_data: dict) -> dict:
    """
    :param climate_data: the climate_data obtained from calling get_climate_data() function
    :return { (lat, lon): daily_values[] } from API response
    """
    result = {}
    for entry in climate_data["data"]:
        if "ll" in entry["meta"]:
            lat, lon = tuple(entry["meta"]["ll"])
            result[(lat, lon)] = entry["data"]
    return result



def get_k_nearest_crop_areas(
    arr: np.ndarray,
    k: int,
    station_dict: dict[tuple[int, int], tuple[float, float]],
    rmax: int = 9999,
) -> list[tuple[float, list[tuple[float, tuple[float, float]]]]]:
    """
    :param arr: the loaded cdl file. Either raw or downscaled. The entry saves the corp area
    :param station_dict: the key is the (row, col) in the cdl file, and the value is (lon, lat)
    :return the corp area, the distance, and the (lon, lat)
    """
    done = np.full(arr.shape, False, dtype=bool)
    results_map = {}
    final_results = []
    active_stations = list(station_dict.keys())

    for r in range(rmax):
        offsets = find_integer_pairs_in_radius(r)
        new_hits = []

        for a, b in active_stations[:]:
            found = False
            for dx, dy in offsets:
                a1, b1 = a + dx, b + dy
                if not (0 <= a1 < arr.shape[0] and 0 <= b1 < arr.shape[1]):
                    continue
                if done[a1, b1]:
                    continue

                found = True
                new_hits.append((a1, b1))
                if (a1, b1) not in results_map:
                    results_map[(a1, b1)] = []
                heapq.heappush(results_map[(a1, b1)], (np.hypot(dx, dy), station_dict[(a, b)]))

            if not found:
                active_stations.remove((a, b))

        for (a1, b1) in new_hits:
            if (a1, b1) in results_map:
                while len(results_map[(a1, b1)]) > k:
                    heapq.heappop(results_map[(a1, b1)])
                if len(results_map[(a1, b1)]) == k:
                    done[a1, b1] = True
                    if arr[a1, b1] > 0:
                        final_results.append((arr[a1, b1], results_map[(a1, b1)]))
                    del results_map[(a1, b1)]

        if not active_stations:
            break

    # Collect remaining
    for (a1, b1), stations in results_map.items():
        if arr[a1, b1] > 0:
            final_results.append((arr[a1, b1], stations))

    return final_results


###########################################################################
# The following function may not be useful

def inverse_distance_weighted_sum(
    station_values: list[tuple[float, tuple[float, float]]],
    data_map: dict,
    day_index: int,
) -> list[float]:
    """
    Calculate weighted value for a pixel based on K-nearest stations.
    """
    result = np.zeros(5)  # 5 weather elements
    for i in range(5):
        values = []
        distances = []
        for dist, coord in station_values:
            val = data_map[coord][day_index][i]
            if val in ['M', 'S'] or (isinstance(val, str) and val.endswith("A")):
                continue
            val = 0 if val == 'T' else float(val)
            values.append(val)
            distances.append(dist)
        if not distances:
            return [-99999]
        if distances[0] == 0:
            result[i] = values[0]
        else:
            inv_sum = sum(1 / d for d in distances)
            result[i] = sum((1 / d) / inv_sum * v for d, v in zip(distances, values))
    return result




def process_daily_weather(date: date, pixel_station_map, station_data_map, start_date: date) -> tuple[date, dict]:
    """
    Process weather summary for one day across all crop regions.
    """
    daily_climate = {}
    day_offset = (date - start_date).days
    for area, stations in pixel_station_map:
        weighted = inverse_distance_weighted_sum(stations, station_data_map, day_offset)
        if len(weighted) == 1:
            continue
        key = tuple(round(x, 1) for x in weighted)
        daily_climate[key] = daily_climate.get(key, 0) + area
    return date, daily_climate


def aggregate_daily_climate(pixel_station_map, station_data_map, start: str, end: str) -> dict[date, dict]:
    """
    Multithreaded daily computation of (weather key → area) across dates.
    """
    start_date = datetime.fromisoformat(start).date()
    end_date = datetime.fromisoformat(end).date()
    results = {}

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_daily_weather, date, pixel_station_map, station_data_map, start_date)
            for date in daterange(start_date, end_date)
        ]
        for future in futures:
            date, daily = future.result()
            results[date] = daily
    return results


def compute_weather_projections(area_by_weather: dict[date, dict]) -> dict[date, list[dict]]:
    """
    Converts climate area data into 5-element daily distributions for each weather metric.
    """
    projections = {}
    for date, dist in area_by_weather.items():
        summary = [{} for _ in range(5)]  # maxt, mint, avgt, pcpn, snow
        for key, area in dist.items():
            for i in range(5):
                summary[i][key[i]] = summary[i].get(key[i], 0) + area
        projections[date] = summary
    return projections


def compute_weighted_statistics(distribution: dict[float, float]) -> dict:
    values = np.array(list(distribution.keys()))
    weights = np.array(list(distribution.values()))
    stats = {}

    if len(values) == 0:
        return {k: None for k in [
            "Weighted Mean", "Weighted Variance", "Weighted Std",
            "Weighted Skewness", "Weighted Kurtosis", "Weighted Median",
            "Min", "Max"
        ]}

    wstats = sm.stats.DescrStatsW(values, weights)
    stats["Weighted Mean"] = wstats.mean
    stats["Weighted Variance"] = wstats.var
    stats["Weighted Std"] = wstats.std

    centered = values - wstats.mean
    w2 = np.sum(weights * centered**2)
    stats["Weighted Skewness"] = np.sum(weights * centered**3) / (w2**1.5)
    stats["Weighted Kurtosis"] = np.sum(weights * centered**4) / (w2**2) - 3

    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)
    median_idx = np.searchsorted(cumsum_weights, total_weight / 2)
    stats["Weighted Median"] = sorted_values[median_idx]

    stats["Min"] = float(np.min(values))
    stats["Max"] = float(np.max(values))
    return stats


def extract_weather_features(weather_projections: dict[date, list[dict]]) -> dict[date, dict]:
    """
    Compute descriptive stats for each of 5 weather metrics, per day.
    """
    all_features = {}
    for date, dists in weather_projections.items():
        features = {}
        metric_names = ["avg_temp", "max_temp", "min_temp", "precip", "snow"]
        for i, dist in enumerate(dists):
            stats = compute_weighted_statistics(dist)
            for key, val in stats.items():
                features[f"{metric_names[i]}_{key.replace(' ', '_').lower()}"] = val
        all_features[date] = features
    return all_features
