import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from rasterio.transform import rowcol

def pixelToLatitudeLongitude(src, row, col):

    transform = src.transform  # Get the affine transformation
    
    x_proj, y_proj = xy(transform, row, col)
    
    # Define projection transformer
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    
    # Convert projected (X, Y) to (Lon, Lat)
    lon, lat = transformer.transform(x_proj, y_proj)
    
    return (lat, lon)

def latitudeLongitudeToPixel(src, lat, lon, m):
    transform = src.transform  # Get affine transform
    crs = src.crs  # Get coordinate system
    # Define the transformer from WGS84 (EPSG:4326) to NAD83 / Conus Albers (EPSG:5070)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    
    # Convert to projected coordinates
    x_proj, y_proj = transformer.transform(lon, lat)
    
    # print(f"Projected X: {x_proj}, Projected Y: {y_proj}")
    row,col = rowcol(transform, x_proj, y_proj)
    return (row//m, col//m)


import heapq
def getLatLonClimateFromMatrixAndList(arr, k, stationDict, rmax = 9999999):
    # Initialize missing variables that weren't in the original code
    done = [[False for _ in range(len(arr[0]))] for _ in range(len(arr))]
    results_map = {}
    trueResults = []
    l = list(stationDict.keys())
    for r in range(rmax):
        indexShift = find_integer_pairs_optimized(r)
        hotIndices = []
        # print(r)
        for (a,b) in l:
            relevant = False
            for (i,j) in indexShift:
                a1 = a+i
                b1 = b+j

                #bounds check
                if a1 < 0 or a1 >= len(arr) or b1 < 0 or b1 >= len(arr[0]):
                    continue
                if done[a1][b1]:
                    continue
                relevant = True
                hotIndices.append((a1,b1))
                if (a1,b1) not in results_map:
                    results_map[(a1,b1)] = []
                heapq.heappush(results_map[(a1,b1)], (np.sqrt(i**2+j**2),stationDict[(a,b)]))
            if not relevant:
                l.remove((a,b))
                
        for (a1,b1) in hotIndices:
            # Ensure the key exists in results_map
            if (a1,b1) in results_map:
                while len(results_map[(a1,b1)]) > k:
                    heapq.heappop(results_map[(a1,b1)])
                if len(results_map[(a1,b1)]) == k:
                    done[a1][b1] = True
                    if arr[a1][b1] != 0:
                        trueResults.append((arr[a1][b1],results_map[(a1,b1)]))
                    # Fixed remove operation - should be del instead of remove
                    del results_map[(a1,b1)]

        if len(l) == 0:
            break
    for (a,b) in results_map.keys():
        if arr[a][b] != 0:
            trueResults.append((arr[a][b],results_map[(a,b)]))
        # print(trueResults)
    return trueResults
                
        
def find_integer_pairs_optimized(r):
    """
    Optimized version that uses geometric properties to reduce the search space.
    We only need to search in a ring-shaped region.
    
    Args:
        r (int): The lower bound of the distance range
        
    Returns:
        list: List of tuples (x, y) satisfying r <= sqrt(x^2 + y^2) < r+1
    """
    result = []
    
    # For optimization, we can limit y based on each x value
    # If x^2 + y^2 is between r^2 and (r+1)^2, then
    # y must be between sqrt(r^2 - x^2) and sqrt((r+1)^2 - x^2)
    for x in range(-r-1, r+2):
        # Skip impossible x values
        if abs(x) > r+1:
            continue
            
        # Calculate y bounds for this x
        min_y_squared = max(0, r**2 - x**2)
        max_y_squared = (r+1)**2 - x**2
        
        # Skip if no valid y exists for this x
        if min_y_squared > max_y_squared:
            continue
            
        min_y = int(min_y_squared**0.5)
        max_y = int(max_y_squared**0.5) + 1
        
        # Check each potential y value
        for y in range(min_y, max_y + 1):
            dist_squared = x**2 + y**2
            if r**2 <= dist_squared < (r+1)**2:
                result.append((x, y))
                # We can also add the symmetric point in the other quadrants
                if y != 0:
                    result.append((x, -y))
                if x != 0:
                    result.append((-x, y))
                if x != 0 and y != 0:
                    result.append((-x, -y))
        
    # Remove duplicates that might have been added in the symmetric additions
    return list(set(result))

import numpy as np

def apply_convolution(matrix, kernel, cropValue):
    """
    Applies a kxk convolution filter with a stride of k on a matrix.
    
    Parameters:
        matrix (2D numpy array): The input matrix.
        kernel (2D numpy array): The kxk filter to apply.
        k (int): The kernel size and stride.
    
    Returns:
        2D numpy array: The downsampled convolved matrix.
    """
    # Get dimensions
    m, n = matrix.shape
    km, kn = kernel.shape

    # Compute output size
    out_m = m // km
    out_n = n // kn

    # Create output matrix
    output = np.zeros((out_m, out_n))

    # Perform convolution with stride k
    for i in range(out_m):
        for j in range(out_n):
            patch = matrix[i*km : i*km + km, j*kn : j*kn + kn]  # Extract kxk patch
            mask = (patch == cropValue)
            masked_matrix_patch = np.where(mask, np.uint8(1),np.uint8(0))
            output[i, j] = np.sum(masked_matrix_patch * kernel)  # Apply convolution
    
    return output

#src is something like src = "../data/2023_30m_cdls.tif" aka HUGE array
#m is the downscale factor for the src array
#cropValue is the "categorization code" here https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.7

class bigArrayParser:
    def __init__(self, src, m, cropValue):
        self.src = rasterio.open(src)
        self.arr = apply_convolution(self.src.read(1), np.ones((m,m)), cropValue)
        self.m = m
        self.cropValue = cropValue
        return
        
    #lonlatlistlist is a list of lists of longitude and latitudes (of weather stations), probably one list for each date
    #k is the k nearest neighbors
    #r is some notion of distance that you can basically think that scales linearly i.e. a unit like miles or kilometers. It is the search radius.
    #for each nonzero amount of 30 by 30 meters of land of cropValue within r pixel radius of some station in a list in lonlatlistlist, 
    # it returns how many 30 by 30 meters of crop production there is, along with k pairs (distance to station, lonlat of station) 
    # associated to the k nearest stations. The algorithm downsamples by a factor of m from src.
    def getKNearestLocations(self, lonlatlistlist, k, r):
        retval = []
        for i in range(len(lonlatlistlist)):
            dat = getKNearestLocationsHelper(lonlatlistlist[i], k, r, m)
            retval.append(dat)
        return retval

    # weatherStations is a list of tuples (lat, lon)
    def getKNearestLocationsHelper(self, weatherStations, k, rmax, m):
        # arr = src.read(1)
        stationDict = {}
        for (lon, lat) in weatherStations:
            # print(lat,lon)
            stationDict[latitudeLongitudeToPixel(self.src, lat, lon, m)] = (lon, lat)
        results = getLatLonClimateFromMatrixAndList(self.arr, k, stationDict, rmax)
        return results

from datetime import date, timedelta
import requests
import json
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
from datetime import timedelta

def get_climate_data(state, start_date, end_date):
    stns = []
    stndata = get_station_data(state, start_date, end_date)["meta"]
    for i in range(len(stndata)):
        stns.append(stndata[i]["sids"][0])
    
    url = "http://data.rcc-acis.org/MultiStnData"
    params = {
        # "sid": sid,  # Station ID
        "sdate": start_date,  # Start date
        "edate": end_date,  # End date
        # "date": date,
        "sids": stns,
        "elems":["maxt","mint","avgt","pcpn","snow"],
        "output": "json"
    }
    
    response = requests.post(url, json=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code, response.text)

def get_station_data(state, start_date, end_date):
    url = "http://data.rcc-acis.org/StnMeta"
    # current_date = str(datetime.now().date())
    # start_date = str((datetime.now()-timedelta(days = 15*1)).date())
    params = {
        # "sid": sid,  # Station ID
        "sdate": start_date,  # Start date
        "edate": end_date,  # End date
        # "date": date,
        "state": state,
        "elems":["avgt"],
        "output": "json"
    }
    
    response = requests.post(url, json=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code, response.text)

def daterange(start_date: date, end_date: date):
    
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)

def get_station_data_by_date(start_date, end_date, state):
    start_date2 = datetime.fromisoformat(start_date)
    end_date2 = datetime.fromisoformat(end_date)
    
    lonlatlistlist = []
    for date in daterange(start_date2, end_date2):
        lonlatlist = []
        d = get_station_data(str(date)[0:10], state)
        for i in range(len(d["data"])):
            if "ll" in d.get("data")[i]["meta"].keys():
                lonlatlist.append(d.get("data")[i]["meta"]["ll"])
        lonlatlistlist.append([str(date)[0:10],lonlatlist])

    return lonlatlistlist
    
def climate_data_to_dict(climate_data):
    ll_to_data = {}
    climate_data2 = climate_data["data"]
    for i in range(len(climate_data2)):
        if "ll" in climate_data2[i]["meta"].keys():
            l1, l2 = climate_data2[i]["meta"]["ll"]
            ll_to_data[(l1, l2)] = climate_data2[i]["data"]
    return ll_to_data

def weightedSum(KNearestLocationDataValue, ll_to_data, days):
    retval = np.zeros(5)
    #5 is the number of elems
    for i in range(5):
        valArr = []
        distArr = []
        for j in range(len(KNearestLocationDataValue)):
            
            if ll_to_data[KNearestLocationDataValue[j][1]][days][i] == 'M' or ll_to_data[KNearestLocationDataValue[j][1]][days][i] == 'S' or ll_to_data[KNearestLocationDataValue[j][1]][days][i][-1] == 'A':
                continue
            if ll_to_data[KNearestLocationDataValue[j][1]][days][i] == 'T':
                valArr.append(0)
                distArr.append(KNearestLocationDataValue[j][0])
            else:
                valArr.append(ll_to_data[KNearestLocationDataValue[j][1]][days][i])
                distArr.append(KNearestLocationDataValue[j][0])
        if len(distArr) == 0:
            return [-99999]
        if distArr[0] == 0:
            retval[i] = valArr[0]
        else:
            totInvDist = sum(1/distArr[j] for j in range(len(distArr)))
            retval[i] = sum((1/distArr[j])/totInvDist * float(valArr[j]) for j in range(len(valArr)))
    return retval

    def get_area_with_climate(k, r, m, state, start_date, end_date):
    
        climate_area_data_over_time={}
        ll_to_data = climate_data_to_dict(get_climate_data(state, start_date, end_date))
        KNearestLocationData = self.getKNearestLocationsHelper(list(ll_to_data.keys()), k, r, m)
        start_date2 = datetime.fromisoformat(start_date)
        end_date2 = datetime.fromisoformat(end_date)
        for date in daterange(start_date2, end_date2):
            climate = {}
            for i in range(len(KNearestLocationData)):
                area = KNearestLocationData[i][0]
                retval = weightedSum(KNearestLocationData[i][1], ll_to_data, (date-start_date2).days)
                if len(retval) == 1:
                    continue
                maxt, mint, avgt, prcp, snow = retval
                if (round(maxt, 1), round(mint, 1), round(avgt, 1), round(prcp, 1), round(snow, 1)) not in climate.keys():
                    climate[(round(maxt, 1), round(mint, 1), round(avgt, 1), round(prcp, 1), round(snow, 1))] = area
                else:
                    climate[(round(maxt, 1), round(mint, 1), round(avgt, 1), round(prcp, 1), round(snow, 1))] += area
            climate_area_data_over_time[date] = climate
        return climate_area_data_over_time