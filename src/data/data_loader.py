import pandas as pd
import requests
import yfinance as yf
import datetime
from src.data.preprocess import extend_market_data
from src.utils.config_tool import parse_config

_USDA_url = 'https://quickstats.nass.usda.gov/api/api_GET/'
_NCDC_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/"
config = parse_config()

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
        response = requests.get(_NCDC_url + 'data', params=params, headers={'token': config['api']['NCDC_api_key']})

        if response.status_code == 200:
            data = response.json()
            weather_data = pd.DataFrame(data['results'])
            weather_data['date'] = pd.to_datetime(weather_data['date'])
            print("successfully fetched weather data")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

        stations_df = pd.DataFrame(columns=['name', 'latitude', 'longitude'])
        stations_df.index.name = 'station'
        for station in weather_data.station.unique():
            response = requests.get(_NCDC_url + 'stations/' + station, headers={'token': config['api']['NCDC_api_key']})
            if response.status_code == 200:
                stations_df.loc[station] = [response.json()['name'], response.json()['latitude'], response.json()['longitude']]
            else:
                print(f"Error: cannot access station {station} with error code {response.status_code}")
                return None
                # return weather_data

        return weather_data.merge(stations_df, on='station', how='left')

    def get_market_data(self, symbol: str, start_date: str, end_date: str, extended=True) -> pd.DataFrame | None:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date).drop(['Dividends', 'Stock Splits'], axis=1)
        return extend_market_data(data) if extended else data

    def get_usda_data(self, commodity: str, start_year: int, params: dict, raw: bool = False) -> pd.DataFrame | None:
        # the following are base params
        full_params = {
            'key': config['api']['USDA_api_key'],
            'commodity_desc': commodity,
            'year__GE': start_year,
            "format": "JSON",
        }
        full_params.update(params)

        response = requests.get(_USDA_url, params=full_params)

        # If exceeds the data limit, we get the data year by year instead
        if response.status_code == 413:
            print(f"Response too large for {commodity}. Switching to year-by-year download.")

            all_years_data = []

            current_year = datetime.date.today().year
            for year in range(start_year, current_year + 1):
                print(f"Fetching {commodity} - {year}...")
                year_params = full_params.copy()
                year_params.pop('year__GE', None)  # remove year__GE
                year_params['year'] = year  # replace with exact year

                try:
                    year_response = requests.get(_USDA_url, params=year_params)
                    if year_response.status_code == 200:
                        year_data = year_response.json()
                        if "data" in year_data:
                            year_df = pd.DataFrame(year_data["data"])
                            if not year_df.empty:
                                all_years_data.append(year_df)
                    else:
                        print(f"Warning: Failed to fetch {commodity} - {year}. Status {year_response.status_code}")
                except Exception as e:
                    print(f"Exception during fetching {commodity} - {year}: {e}")

            if all_years_data:
                df = pd.concat(all_years_data, ignore_index=True)
            else:
                print(f"No data found for {commodity} from {start_year} onward.")
                return None

        elif response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return None

        else:
            data = response.json()
            df = pd.DataFrame(data['data'])

        if raw:
            return df
        # else, light cleaning of the data
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
        if "Value" in df.columns:
            df["Value"] = df["Value"].str.replace(",", "", regex=True)
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        if "load_time" in df.columns:
            df["load_time"] = pd.to_datetime(df["load_time"]).dt.date
        if "end_code" in df.columns:
            df["end_code"] = pd.to_numeric(df["end_code"], errors="coerce")


        return df


    def get_production_data(self, commodity: str, start_year: int, national_level=False, raw=False) -> pd.DataFrame | None:
        """
        :param commodity: the name of the commodity. For example, "WHEAT", "CORN", "SOYBEANS", "SUNFLOWER", "SUGARCANE", "SUGARBEETS"
        :param start_year: the starting year of the data
        :param raw: If true, this will return the raw data with many additional columns
        :return: the production data obtained from the API

        This function will return the annual production data per state obtained from the USDA API. Please be aware that
        there could be several data in a year. This is usually caused by different data collection methods (survey vs census)
        or different commodity subcategories (e.g. For wheat, there are HRW, SRW, HRS wheat)
        """
        params = {
            'key': config['api']['USDA_api_key'],
            'commodity_desc': commodity,
            'statisticcat_desc': 'PRODUCTION',
            'agg_level_desc': 'STATE',
            'year__GE': start_year,
            'format': 'JSON',
#            'unit_desc': 'BU',
#            'domaincat_desc': 'NOT SPECIFIED',
        }

        if national_level:
            params['agg_level_desc'] = 'NATIONAL'

        response = requests.get(_USDA_url, params=params)

        if response.status_code == 200:
            data = response.json()
            raw_data = pd.DataFrame(data['data'])

            if raw:
                return raw_data
            else:
                # Convert the numerical columns from string to numerical values
                raw_data['year'] = pd.to_numeric(raw_data['year'])
                raw_data['Value'] = raw_data['Value'].str.replace(',', '', regex=True)
                raw_data['Value'] = pd.to_numeric(raw_data['Value'], errors='coerce')

                cols = ['state_name', 'Value', 'unit_desc', 'year', 'source_desc', 'short_desc', 'reference_period_desc']

                return raw_data[cols]
        else:
            print(f"Error: {response.status_code}, {response.text}")

    def get_stocks_data(self, commodity: str, start_year: int, national_level=False, raw=False) -> pd.DataFrame | None:
        """
        :param commodity: the name of the commodity. For example, "WHEAT", "CORN", "SOYBEANS", "SUNFLOWER", "SUGARCANE", "SUGARBEETS"
        :param start_year: the starting year of the data
        :param raw: if true, this will return the raw data with many additional columns
        :param national_level: If true, this will return the US national level data instead of state level data
        :return:

        This function will return the quarterly stock data per state or nationwide obtained from the USDA API.
        """
        params = {
            'key': config['api']['USDA_api_key'],
            'commodity_desc': commodity,
            'statisticcat_desc': 'STOCKS',
            'agg_level_desc': 'STATE',
            'year__GE': start_year,
            'format': 'JSON',
        }

        if national_level:
            params['agg_level_desc'] = 'NATIONAL'

        response = requests.get(_USDA_url, params=params)

        if response.status_code == 200:
            data = response.json()
            raw_data = pd.DataFrame(data['data'])

            if raw:
                return raw_data
            else:
                # Convert the numerical columns from string to numerical values
                raw_data['year'] = pd.to_numeric(raw_data['year'])
                # raw_data['end_month'] = raw_data['end_month'].astype(int)
                raw_data['Value'] = raw_data['Value'].str.replace(',', '', regex=True)
                raw_data['Value'] = pd.to_numeric(raw_data['Value'], errors='coerce')

                pivoted = raw_data.pivot(index=['year', 'end_code', 'state_name'], columns='short_desc',
                                       values='Value').reset_index()

                pivoted.rename(columns={'end_code': 'end_month'}, inplace=True)
                return pivoted
        else:
            print(f"Error: {response.status_code}, {response.text}")

    def get_condition_data(self, commodity: str, start_year: int, exact_year=None, national_level=False,
                        raw=False) -> pd.DataFrame | None:
        """
        :param commodity: the name of the commodity. For example, "WHEAT", "CORN", "SOYBEANS", "SUNFLOWER", "SUGARCANE", "SUGARBEETS"
        :param start_year: the starting year of the data
        :param exact_year: if this value is set, then this function will return the data of that year.
        :param raw: if true, this will return the raw data with many additional columns
        :param national_level: If true, this will return the US national level data instead of state level data
        :return:

        This function will return the growing condition of the selected commodity. This data is provided by the USDA weekly.
        """
        params = {
            'key': config['api']['USDA_api_key'],
            'commodity_desc': commodity,
            'statisticcat_desc': 'CONDITION',
            'agg_level_desc': 'STATE',
            'year__GE': start_year,
            'format': 'JSON',
        }

        if national_level:
            params['agg_level_desc'] = 'NATIONAL'

        if exact_year:
            del params['year__GE']
            params['year'] = exact_year

        response = requests.get(_USDA_url, params=params)

        if response.status_code == 200:
            data = response.json()
            raw_data = pd.DataFrame(data['data'])

            if raw:
                return raw_data
            else:
                # Convert the numerical columns from string to numerical values
                raw_data['year'] = pd.to_numeric(raw_data['year'])
                raw_data['Value'] = raw_data['Value'].str.replace(',', '', regex=True)
                raw_data['Value'] = pd.to_numeric(raw_data['Value'], errors='coerce')

                pivoted = raw_data.pivot(index=['week_ending', 'year', 'state_name', 'end_code'], columns='unit_desc', values='Value').reset_index().set_index('week_ending')
                pivoted.rename(columns={'end_code': 'week_number'}, inplace=True)

                return pivoted
        else:
            print(f"Error: {response.status_code}, {response.text}")

