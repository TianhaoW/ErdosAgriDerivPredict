import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# Creating a series of expiry days of those contracts
contract_months = [3, 5, 7, 9, 12]
expiry_dates = []
for year in range(2000, 2026):
    for month in contract_months:
        fifteenth = pd.Timestamp(year, month, 15)
        last_biz_day = fifteenth - BDay(1)  # Last business day before the 15th
        expiry_dates.append(last_biz_day)

expiry_series = pd.Series(expiry_dates).sort_values()

def extend_market_data(df):
    """
    :param df: The pandas dataframe obtained from yfinance library
    :return: The extended market data
    """
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
    # Exponential Moving Average (EMA)
    # A weighted version of moving average giving more weight to recent prices.
    df['7D_EMA'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['14D_EMA'] = df['Close'].ewm(span=14, adjust=False).mean()

    return df
