# Simulating a fake data for testing of our model

import pandas as pd
import numpy as np

np.random.seed(21)   # For reproducibility

# Generating timestamps every 5 seconds from 9:15 AM for 1 hour
timestamps = pd.date_range(start = "2025-04-25 09:15:00", periods = 720, freq = '5S')

# Simulate prices using a normal distribution (centered around 100 rupees)
prices = np.random.normal(loc = 100, scale = 1, size = len(timestamps)).round(2)

# Simulating the volume between 1 and 10 units per tick
volumes = np.random.randint(1, 10, size = len(timestamps))

# Creating tick dataframe
tick_data = pd.DataFrame({
    'timestamp' : timestamps,
    'price' : prices,
    'volume' : volumes
})

# Convert timestamp to datetime for safety
tick_data['datetime'] =pd.to_datetime(tick_data['timestamp'])
# Set datetime as index for resampling
tick_data.set_index("datetime", inplace = True)

# OHLC Resampling
# Using resample('1Min') to group ticks by 1-minute intervals
# .ohlc() gives :
# Open = first value
# High = max
# Low = min
# Close = last

ohlc = tick_data['price'].resample('5Min').ohlc()

# Volume Aggregation : We sum up all the volume in the same minute
volume = tick_data['volume'].resample('5Min').sum()

# Combining the ohlc and volume dataframes
ohlcv = ohlc.copy()
ohlcv['volume'] = volume

print(ohlcv)

# Reset index for downstream use
ohlcv.reset_index(inplace = True)
print('==============================================================================')
print(ohlcv)

# Saving the ohclv data into a csv file
ohlcv.to_csv('Data/dummy_stock_data.csv', index = False)
