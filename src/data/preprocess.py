import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries.offsets import BDay
from src.utils import parse_config
from scipy.stats import gamma, norm

config = parse_config()

#########################################################################################################
# Market data preprocess

def _get_expiry_series(contract_months, rule, start_year=2000, end_year=2030):
    expiry_dates = []
    for year in range(start_year, end_year + 1):
        for month in contract_months:
            if rule == "before_15th":
                anchor = pd.Timestamp(year, month, 15)
                expiry = anchor - BDay(1)
            elif rule == "last_business_day":
                expiry = pd.Timestamp(year, month, 1) + pd.offsets.BMonthEnd(0)
            else:
                raise ValueError(f"Unknown expiry rule: {rule}")
            expiry_dates.append(expiry)

    return pd.Series(sorted(expiry_dates))

def extend_market_data(df, symbol='ZW'):
    """
    :param df: The pandas dataframe obtained from yfinance library
    :param symbol: The symbol of the commodity. Supports ZW, ZC, ZS, ZL, ZM, SB, OJ. The default value is 'ZW' to support old code.
    :return: The extended market data
    """
    df.index = pd.to_datetime(df.index).tz_localize(None)

    contract_info = config["contracts"].get(symbol)
    if contract_info is None:
        raise ValueError(f"No expiry rule found for symbol: {symbol}")

    contract_months = contract_info["months"]
    rule = contract_info["rule"]
    expiry_series = _get_expiry_series(contract_months, rule)

    ########################################################
    # Seasonality & Time Features
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Day_Of_Year'] = df.index.dayofyear
    # get the expiry date of this specific contract
    df["expiry"] = df.index.map(lambda x: expiry_series[expiry_series >= x].iloc[0])
    # computing the days to expiry
    df["DTE"] = (df["expiry"] - df.index).dt.days

    ##############################################################
    # Volatility Features:
    # Historical Volatility
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df['7D_Volatility'] = df['Log_Return'].rolling(window=7).std()
    df['14D_Volatility'] = df['Log_Return'].rolling(window=14).std()
    # Average True Range (ATR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['14D_ATR'] = df['TR'].rolling(window=14).mean()
    # Volume-to-Volatility Ratio
    df['Volume_Volatility_Ratio'] = df['Volume'] / df['14D_Volatility']

    ##############################################################
    # Momentum Indicator Features:
    # Relative Strength Index (RSI)
    # Measures the speed and change of price movements.
    # Values above 70 indicate overbought conditions, below 30 indicate oversold conditions.
    delta =df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'14D_RSI'] = 100 - (100 / (1 + rs))

    ###############################################################
    # Trend Indicator Features:
    # Moving Average
    df['7D_MA'] = df['Close'].rolling(window=7).mean()
    df['14D_MA'] = df['Close'].rolling(window=14).mean()
    df['21D_MA'] = df['Close'].rolling(window=21).mean()
    # Exponential Moving Average (EMA)
    # A weighted version of moving average giving more weight to recent prices.
    df['7D_EMA'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['14D_EMA'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['21D_EMA'] = df['Close'].ewm(span=21, adjust=False).mean()

    return df


############################################################################################################
# Weather data preprocess

def compute_spi(rolling_prcp: pd.Series) -> pd.Series:
    """
    Compute Standardized Precipitation Index (SPI) using mixed distribution:
    - Point mass at 0
    - Gamma distribution for non-zero precipitation
    """
    rolling_prcp = rolling_prcp.copy()
    rolling_prcp = rolling_prcp.replace([np.inf, -np.inf], np.nan)

    # Separate valid (non-NaN) entries
    valid = rolling_prcp.dropna()
    if len(valid) < 30:
        return pd.Series(index=rolling_prcp.index, data=np.nan)

    # Estimate q = probability of zero precipitation
    q = (valid == 0).sum() / len(valid)

    # Fit gamma only on positive values
    positive_vals = valid[valid > 0]
    if len(positive_vals) < 10:
        return pd.Series(index=rolling_prcp.index, data=np.nan)

    try:
        shape, loc, scale = gamma.fit(positive_vals, floc=0)
    except Exception:
        return pd.Series(index=rolling_prcp.index, data=np.nan)

    # Mixed CDF: for each value x
    def compute_cdf(x):
        if pd.isna(x):
            return np.nan
        elif x == 0:
            return q  # point mass
        else:
            return q + (1 - q) * gamma.cdf(x, a=shape, loc=loc, scale=scale)

    cdf_values = rolling_prcp.apply(compute_cdf)

    # Handle edge values before applying norm.ppf
    cdf_values = cdf_values.clip(lower=1e-6, upper=1 - 1e-6)

    spi = pd.Series(norm.ppf(cdf_values), index=rolling_prcp.index)
    return spi

def weather_anomaly_preprocess(df: DataFrame):
    '''
    :param df: The weather dataframe obtained from loading the csv file generated by update_data()
    :return: the processed weather dataframe
    '''
    threshold = config['data']['z_score_anomaly_threshold']

    # The T means the rain do occur, but it is too small to measure
    df['precip'] = df['precip'].replace('T', 0.01)
    df['snow'] = df['snow'].replace('T', 0.01)

    for col in ['tmax', 'tmin', 'tavg', 'precip', 'snow']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df["day_of_year"] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df = df.sort_values(by=["station_name", "date"])

    # standard Z-score for the tempeature data
    value_cols = ['tavg', 'tmax', 'tmin']
    grouped_df = df.groupby(['station_name', "day_of_year"])
    for value_col in value_cols:
        grouped = grouped_df[value_col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")

        z = (df[value_col] - mean) / std
        df[f"{value_col}_zscore"] = z
        df[f"{value_col}_anomaly"] = z.abs() > threshold

    # For the precipitation, we use the SPI (Standardized Precipitation Index) and 14 day rolling sum
    df['precip_14d'] = (
        df.groupby(['station_name'])['precip']
        .transform(lambda x: x.rolling(14, min_periods=1).sum())
    )
    df['spi_14d'] = df.groupby(['station_name', "month"])['precip_14d'].transform(compute_spi)

    # This is a more smoothed version using the exponential weighted moving average to emphasize on recent value
    df['precip_ewma'] = (
        df.groupby('station_name')['precip']
        .fillna(0)
        .transform(lambda x: x.ewm(span=14, adjust=False).mean())
    )
    df['spi_ewma'] = df.groupby(['station_name', "month"])['precip_ewma'].transform(compute_spi)


    # For the snow, we use the same method, but a longer rolling window
    df['snow_ewma'] = (
        df.groupby(['station_name'])['snow']
        .fillna(0)
        .transform(lambda x: x.ewm(span=28, adjust=False).mean())
    )
    # df['snow_spi'] = df.groupby(['station_name', "month"])['snow_ewma'].transform(compute_spi)

    return df
