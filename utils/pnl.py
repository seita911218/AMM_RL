import pandas as pd
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uniswap import swap_fee, LVR

# Load fee_tier from config
path_config = os.path.dirname(os.path.abspath(__file__)) + '/..' + '/config' + '/config.json'
with open(path_config, "r") as f:
    config = json.load(f)
fee_tier = float(config["fee_tier"])


def pnl(data, L, gas):
    """
    Calculate PnL metrics for each cycle
    
    Args:
        data: pandas DataFrame with columns ['time', 'closed_price', 'scaled_volume_USDC', 'scaled_volume_WETH']
        L: liquidity added for each cycle (float)
    
    Returns:
        pandas DataFrame with columns ['swap_fee_USDC', 'swap_fee_WETH', 'swap_fee_total', 'LVR', 'reward']
    """
    
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        
        # Get current cycle data
        close_price = row['closed_price']
        x_scaled_vol = row['scaled_volume_WETH']  # WETH is tokenX
        y_scaled_vol = row['scaled_volume_USDC']  # USDC is tokenY
        
        # Calculate swap fees
        swap_fee_x, swap_fee_y, swap_fee_total = swap_fee(
            close_price, x_scaled_vol, y_scaled_vol, L
        )
        
        # Calculate LVR (price_in is previous row's closed_price)
        if i == 0:
            # For first row, use current price as price_in
            lvr = 0.0
        else:
            price_in = df.iloc[i-1]['closed_price']
            price_out = close_price
            lvr = LVR(price_in, price_out, L)
        
        # Calculate reward
        reward = swap_fee_total + lvr - gas
        
        results.append({
            'time': row['time'],
            'closed_price': close_price,
            'scaled_volume_WETH': x_scaled_vol,
            'scaled_volume_USDC': y_scaled_vol,
            'swap_fee_WETH': swap_fee_x,  # WETH is tokenX
            'swap_fee_USDC': swap_fee_y,  # USDC is tokenY
            'swap_fee_total': swap_fee_total,
            'LVR': lvr,
            'reward': reward
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":

    data_path = os.path.dirname(os.path.abspath(__file__)) + '/..' + '/data'
    data = pd.read_csv(data_path + '/organized_hourly_data.csv')
    L = int(2.6162685701074442e+17)
    gas = 5
    result = pnl(data, L, gas)
    result.to_csv(data_path + "/pnl.csv")