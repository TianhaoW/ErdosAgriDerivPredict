import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from src.utils import parse_config

config = parse_config()

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
