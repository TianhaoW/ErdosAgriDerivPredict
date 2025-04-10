import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


def _compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    return data

# Creating a series of expiry days of those contracts
# delivery months: January, March, May, July, and October
#expiry_months = [12,2,4,6,9]
#expiry_dates = []
#for year in range(2000, 2026):
#    for month in expiry_months:
#        last_day = pd.Timestamp(year, month+1, 31)
#        last_biz_day = last_day - BDay(1)  # Last business day before the last day of the month
#        expiry_dates.append(last_biz_day)

#expiry_series = pd.Series(expiry_dates).sort_values()


ALLOWED_DELIVERY_MONTHS = [1, 3, 5, 7, 10]

def get_expiry(dt):
    """
    Given a timestamp dt, returns the expiry date as the last trading day of the month
    preceding the next allowed delivery month.
    """
    dt = pd.Timestamp(dt)
    # Loop through allowed delivery months (sorted in ascending order)
    for m in sorted(ALLOWED_DELIVERY_MONTHS):
        # For the current year, build the delivery date as the first day of month m
        delivery_date = pd.Timestamp(year=dt.year, month=m, day=1)
        # Expiry is defined as the last day of the month preceding the delivery month
        expiry = delivery_date + MonthEnd(-1)
        # Adjust expiry to the last business day if it falls on a weekend
        if expiry.weekday() >= 5:  # Saturday=5, Sunday=6
            expiry -= BDay(1)
        # Check if this expiry is still in the future relative to dt
        if expiry > dt:
            return expiry

    # If no delivery month in the current year qualifies, use the first allowed month of next year
    m = sorted(ALLOWED_DELIVERY_MONTHS)[0]
    delivery_date = pd.Timestamp(year=dt.year+1, month=m, day=1)
    expiry = delivery_date + MonthEnd(-1)
    if expiry.weekday() >= 5:
        expiry -= BDay(1)
    return expiry


def extend_market_data(df):
    """
    :param df: The pandas dataframe obtained from yfinance library
    :return: The extended market data
    """
    ########################################################
    # Seasonality & Time Features
    # remove time zone featuress
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Day_Of_Year'] = df.index.dayofyear
    # get the expiry date of this specific contract
    
    # df["expiry"] = df.index.map(lambda x: expiry_series[expiry_series >= x].iloc[0])

    df['expiry'] = df.index.to_series().apply(get_expiry)
    # df['expiry'] = df['datetime'].apply(get_expiry)
   
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
    # Exponential Moving Average (EMA)
    # A weighted version of moving average giving more weight to recent prices.
    df['7D_EMA'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['14D_EMA'] = df['Close'].ewm(span=14, adjust=False).mean()

    return df