# Importing dependencies
import pandas as pd
import numpy as np

# The indicators we will be using for training our model are :
# 1. ATR
# 2. EMA
# 3. Volume
# 4. RSI
# 5. Rolling Means and std Devs

# Creating function to calculate the technical indicators
def calc_rsi(df, period : int = 14):
    """Function to calculate the Relative Strength Index over a particular period of time."""
    delta = df['Close'].diff()   # This calculates the change between consecutive close prices.
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window = period, min_periods = 1).mean()
    # Instead of pd.Series(gain) we could have also used pd.DataFrame(gain, index = [i for i in range(len(gain))] as they both perform the same function. The major reason for using Series is :
    # It is faster and lighter for 1D data
    # Methods like rolling(), mean() e.t.c can be used directly with it as it don't give any column name to the values colmn like the Dataframe Method does.
    # So, if we use DataFrame() we have to use :
    # pd.DataFrame(gain, index = [i for i in range(len(gain))])[0].mean()

    avg_loss = pd.Series(loss).rolling(window = period, min_periods = 1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))   # Formula to convert relative strength(rs) to rsi.
    return rsi

def calc_vwap(df):
    """Function to calculate the Volume weighted average price.
    VWAP = (SUM(Price * Volume))/(SUM(Volume))"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3   # Calculating the avg price in a given time frame through the avg of its close, high and low.
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def calc_atr(df, period = 14):
    """Function to calculate the Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
    atr = true_range.rolling(window = period, min_periods = 1).mean()

    return atr

def generate_features(df):
    """Takes in raw OHCLV dataframe and adds technical indicators and rolling stats."""

    df['Volume'] = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())

    # Technical Indicators
    df['RSI_14'] = calc_rsi(df, period = 14)
    df['EMA_5'] = df['Close'].ewm(span = 5, adjust = False).mean()
    df['EMA_20'] = df['Close'].ewm(span = 20, adjust = False).mean()
    df['VWAP'] = calc_vwap(df)
    df['ATR_14'] = calc_atr(df, period = 14)

    # Rolling statistics
    df['rolling_mean_5'] = df['Close'].rolling(window = 5, min_periods = 1).mean()
    df['rolling_std_5'] = df['Close'].rolling(window = 5, min_periods = 1).std()
    df['rolling_volume_mean_5'] = df['Volume'].rolling(window = 5, min_periods = 1).mean()

    # Dropping the NaN values
    df = df.dropna()

    return df

if __name__ == '__main__':
    df = pd.read_csv('Data/reliance_5min_60days.csv')
    df.drop(columns=['Price'], inplace=True)
    df.dropna(inplace=True)
    df.drop(0, inplace=True)
    clmns = ['Close', 'High', 'Low', 'Open', 'Volume']
    for k in clmns:
        df[k] = df[k].astype(float)
    processed_df = generate_features(df)
    processed_df.to_csv('Data/processed_reliance_data.csv', index = False)


