import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the CSV file
df = pd.read_csv('./data/swap_with_liq.csv')

# Convert datetime to pandas datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Calculate scaled volumes
df['scaled_volume_USDC'] = np.maximum(df['USDC'] / df['liquidity'], 0)
df['scaled_volume_WETH'] = np.maximum(df['WETH'] / df['liquidity'], 0)

# Set datetime as index for easier resampling
df.set_index('datetime', inplace=True)

# Group by hourly intervals and aggregate
hourly_data = df.groupby(pd.Grouper(freq='H')).agg({
    'price': 'last',  # Close price as last price in the hour
    'scaled_volume_USDC': 'sum',  # Sum of scaled volumes
    'scaled_volume_WETH': 'sum'
}).reset_index()

# Label each group by the right end point (add 1 hour to timestamp)
hourly_data['datetime'] = hourly_data['datetime'] + timedelta(hours=1)

# Rename columns to match requested format
hourly_data.columns = ['time', 'closed_price', 'scaled_volume_USDC', 'scaled_volume_WETH']

# Create complete hourly range from start to end
start_time = df.index.min().floor('H') + timedelta(hours=1)  # Round up to next hour
end_time = df.index.max().ceil('H') + timedelta(hours=1)  # Add hour to end time too
complete_hours = pd.date_range(start=start_time, end=end_time, freq='H')

# Create complete dataframe with all hours
complete_df = pd.DataFrame({'time': complete_hours})
complete_df = complete_df.merge(hourly_data, on='time', how='left')

# Forward fill prices for missing hours and set volumes to 0
complete_df['closed_price'] = complete_df['closed_price'].fillna(method='ffill')
complete_df['scaled_volume_USDC'] = complete_df['scaled_volume_USDC'].fillna(0)
complete_df['scaled_volume_WETH'] = complete_df['scaled_volume_WETH'].fillna(0)

# Handle case where first hour might be NaN (use first available price)
if pd.isna(complete_df['closed_price'].iloc[0]):
    first_price = df['price'].iloc[0]
    complete_df['closed_price'] = complete_df['closed_price'].fillna(first_price)

# Sort by time
complete_df = complete_df.sort_values('time').reset_index(drop=True)

# Save the organized data
complete_df.to_csv('./data/organized_hourly_data.csv', index=False)

print("Data organized successfully!")
print(f"Total hours: {len(complete_df)}")
print(f"Date range: {complete_df['time'].min()} to {complete_df['time'].max()}")
print("\nFirst 5 rows:")
print(complete_df.head())
print("\nLast 5 rows:")
print(complete_df.tail())