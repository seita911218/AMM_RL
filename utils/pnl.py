import pandas as pd
import numpy as np
import math
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uniswap import swap_fee, LVR,tick_2_tickspacing

# Load fee_tier from config
path_config = os.path.dirname(os.path.abspath(__file__)) + '/..' + '/config' + '/config.json'
with open(path_config, "r") as f:
    config = json.load(f)
fee_tier = float(config["fee_tier"])
decimal_0 = int(config["decimal_0"])
decimal_1 = int(config["decimal_1"])
def set_LP_range(price,width=0.05):
    pl= (1-width)*price
    pr= (1+width)*price 
    a= 10**(decimal_0 -decimal_1)
    kl = tick_2_tickspacing( math.floor( math.log(pl/a,1.0001)  ))
    kr = tick_2_tickspacing(math.ceil( math.log(pr/a,1.0001) ))

    return 1.0001**kl*10**(decimal_0 -decimal_1) , 1.0001**kr*10**(decimal_0 -decimal_1)

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

def pnl_fix_range(data, L,gas,width=0.05):
    
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
         # Calculate LVR (price_in is previous row's closed_price)
        if i == 0:
            # For first row, use current price as price_in(open price)
            price_in=close_price
            lvr = 0.0
        else:
            price_in = df.iloc[i-1]['closed_price']
            price_out = close_price
            lvr = LVR(price_in, price_out, L)
        price_l,price_r= set_LP_range(price_in,width)
        # prepare Y(USDC) token as ini cap, borrowing X(WETH)
        ini_cap=L*((price_in)**(1/2) -(price_l)**(1/2) )/10**(decimal_0-decimal_1)
        swap_fee_x, swap_fee_y, swap_fee_total = swap_fee(
            close_price, x_scaled_vol, y_scaled_vol, L
        )
        
        # Calculate reward
        reward = swap_fee_total + lvr - gas
        # Use LP range to calculate ini cap and use volume to estimate fee
        LP_return=reward/ini_cap
        results.append({
            'time': row['time'],
            'closed_price': close_price,
            'scaled_volume_WETH': x_scaled_vol,
            'scaled_volume_USDC': y_scaled_vol,
            'swap_fee_WETH': swap_fee_x,  # WETH is tokenX
            'swap_fee_USDC': swap_fee_y,  # USDC is tokenY
            'swap_fee_total':swap_fee_total,
            'LVR': lvr,
            'liquidity':L,
            'price_lower': price_l,
            'price_upper': price_r,
            'reward': reward,
            'return':LP_return,
            'ini_cap':ini_cap
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":

    data_path = os.path.dirname(os.path.abspath(__file__)) + '/..' + '/data'
    data = pd.read_csv(data_path + '/organized_hourly_data.csv')
    gas = 5
    L = int(2.6162685701074442e+17)
    result = pnl(data,L, gas)
    result.to_csv(data_path + "/pnl.csv")