#!/usr/bin/env python
# coding: utf-8

# In[39]:


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

# def getLatitudeLongitudeCrop(arr, src, cropValue):
#     # arr = src.read(1)
#     LatLon = []
#     for i in range(len(arr)):
#         if i % 10 == 0:
#             print(i,len(arr))
#         for j in range(len(arr[0])):
#             if arr[i][j] == cropValue:
#                 LatLon.append(pixelToLatitudeLongitude(src, i, j))
#     return LatLon

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
    # print(f"Array indices: Row {row}, Col {col}")
    
    # height, width = src.height, src.width  # Get raster size
    
    # if 0 <= row < height and 0 <= col < width:
    #     print("Point is inside the raster.")
    # else:
    #     print("Point is outside the raster (negative index).")


# In[40]:


# # src = rasterio.open("../data/2023_30m_cdls.tif")
# # arr = src.read(1)
# dat = getLatitudeLongitudeCrop(arr, src, 1)


# In[68]:


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
                # Replace outofbounds with proper bounds check
                if a1 < 0 or a1 >= len(arr) or b1 < 0 or b1 >= len(arr[0]):
                    continue
                if done[a1][b1]:
                    continue
                relevant = True
                hotIndices.append((a1,b1))
                # if arr[a1][b1] == cropValue:
                #     # Initialize the heap if this is the first time seeing this coordinate
                    # if (a1,b1) not in results_map:
                    #     results_map[(a1,b1)] = []
                    # heapq.heappush(results_map[(a1,b1)], (np.sqrt(i**2+j**2),a,b))
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


# In[18]:


# weatherStations is a list of tuples (lat, lon)
def getKNearestLocationsHelper(arr, cropSrc, weatherStations, k, rmax, m):
    # arr = src.read(1)
    stationDict = {}
    for (lon, lat) in weatherStations:
        # print(lat,lon)
        stationDict[latitudeLongitudeToPixel(cropSrc, lat, lon, m)] = (lat, lon)
    results = getLatLonClimateFromMatrixAndList(arr, k, stationDict, rmax)
    return results
    


# In[20]:


import requests
import json
import requests
import json
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
from datetime import timedelta

def get_climate_data(date, state):
    url = "http://data.rcc-acis.org/MultiStnData"
    current_date = str(datetime.now().date())
    start_date = str((datetime.now()-timedelta(days = 15*1)).date())
    params = {
        # "sid": sid,  # Station ID
        # "sdate": start_date,  # Start date
        # "edate": current_date,  # End date
        "date": date,
        "state": state,
        "elems":["maxt","mint","avgt","pcpn","snow"],
        "output": "json"
    }
    
    response = requests.post(url, json=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print("Error fetching data:", response.status_code, response.text)
d = get_climate_data("2025-03-01","IL")
lonlatlist = []
for i in range(len(d["data"])):
    if "ll" in d.get("data")[i]["meta"].keys():
        lonlatlist.append(d.get("data")[i]["meta"]["ll"])


# In[32]:


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


# In[34]:


# m = 30
# cropValue = 1
# src = rasterio.open("../data/2023_30m_cdls.tif")
# arr = apply_convolution(src.read(1), np.ones((m,m)), cropValue)


# # In[69]:


# r = 10
# k = 4
# dat = getKNearestLocationsHelper(arr, src, lonlatlist, k, r, m)

#src is something like src = rasterio.open("../data/2023_30m_cdls.tif") aka HUGE array
#lonlatlist is a list of longitude and latitudes (of weather stations)
#k is the k nearest neighbors
#r is some notion of distance that you can basically think that scales linearly i.e. a unit like miles or kilometers. It is the search radius.
#m is the downscale factor for the src array
#cropValue is the "categorization code" here https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#what.7
#for each nonzero amount of 30 by 30 meters of land of cropValue within r pixel radius of some station in lonlatlist, it returns how many 30 by 30 meters of crop production there is, along with k pairs (distance to station, lonlat of station) associated to the k nearest stations. The algorithm downsamples by a factor of m from src.
def getKNearestLocations(src, lonlatlist, k, r, m, cropValue):
    arr = apply_convolution(src.read(1), np.ones((m,m)), cropValue)
    dat = getKNearestLocationsHelper(arr, src, lonlatlist, k, r, m)
    return dat





