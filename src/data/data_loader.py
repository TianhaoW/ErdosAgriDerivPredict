import pandas as pd
import requests
import yfinance as yf
from src.data.Preprocess import extend_market_data

_USDA_api_key = "7BACC84D-4D4E-31CE-9828-0EAF5D6338DB"
_NCDC_api_key = "sAfDxhRkTzDkPBjtktOThBJjnCfzAtax"
_USDA_url = 'https://quickstats.nass.usda.gov/api/api_GET/'
_NCDC_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/"

class DataLoader:
    def __init__(self):
        pass

    def get_weather_data(self, start_date: str, end_date: str, data_type: str, location_id: str) -> pd.DataFrame | None:
        """
        :param data_type: TMAX, TMIN, TAVG, PRCP, SNOW, SNWD
        :param location_id: please use the FIPS location id in the format like FIPS:06059, which is the orange county of Irvine.
        :return: a dataframe of weather data. This will return at most 1000 lines of data due to the API constrain

        The attributes are of the form Measurement Flag | Quality Flag | Source Flag | Time of Observation

        See https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/doc/GHCND_documentation.pdf for the details

        Here are all allowed data types:

        - PRCP = Precipitation (mm or inches as per user preference, inches to hundredths on Daily Form pdf file)
        - SNOW = Snowfall (mm or inches as per user preference, inches to tenths on Daily Form pdf file)
        - SNWD = Snow depth (mm or inches as per user preference, inches on Daily Form pdf file)
        - TMAX = Maximum temperature (tenth Celsius)
        - TMIN = Minimum temperature (tenth Celsius)
        - TAVG = Average temperature (tenth Celsuis)

        """
        params = {
            'datasetid': 'GHCND',
            'datatypeid': data_type,
            'startdate': start_date,
            'enddate': end_date,
            'locationid': location_id,
            'limit': 1000,
        }
        response = requests.get(_NCDC_url + 'data', params=params, headers={'token': _NCDC_api_key})

        if response.status_code == 200:
            data = response.json()
            weather_data = pd.DataFrame(data['results'])
            weather_data['date'] = pd.to_datetime(weather_data['date'])
            print("successfully fetched weather data")
        else:
            print(f"Error: {response.status_code}")
            return None

        stations_df = pd.DataFrame(columns=['name', 'latitude', 'longitude'])
        stations_df.index.name = 'station'
        for station in weather_data.station.unique():
            response = requests.get(_NCDC_url + 'stations/' + station, headers={'token': _NCDC_api_key})
            if response.status_code == 200:
                stations_df.loc[station] = [response.json()['name'], response.json()['latitude'], response.json()['longitude']]
            else:
                print(f"Error: cannot access station {station} with error code {response.status_code}")
                return weather_data

        return weather_data.merge(stations_df, on='station', how='left')

    def get_market_data(self, symbol: str, start_date: str, end_date: str, extended=True) -> pd.DataFrame | None:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date).drop(['Dividends', 'Stock Splits'], axis=1)
        return extend_market_data(data) if extended else data


