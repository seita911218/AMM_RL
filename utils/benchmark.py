import os
import json
import math
import numpy as np
import pandas as pd

from typing import Optional, Sequence
from pathlib import Path

# ===== Project-specific helpers (AMM math / fee & LVR components) =====
# Expect these functions to return values in account currency (e.g., USDC)
from utils.uniswap import swap_fee, LVR, total_y_amount_2_liquidity

# ===== Load pool/config defaults (can be overridden via __init__ kwargs) =====
_cfg_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "config",
    "config.json",
)
with open(_cfg_path, "r") as f:
    _cfg = json.load(f)

# Common pool metadata / defaults
pool_address = _cfg["pool_address"]
chain = _cfg["chain"]
token0 = _cfg["token0"]
token1 = _cfg["token1"]
decimal_0 = int(_cfg["decimal_0"])            # decimals for token0
decimal_1 = int(_cfg["decimal_1"])            # decimals for token1
fee_tier_default = float(_cfg["fee_tier"])    # e.g., 0.0005 / 0.003 / 0.01
tickspacing_default = int(_cfg["tickspacing"])

def benchmark_1(data, init_equity, gas_fee, out_csv="./result/benchmark_1.csv"):

    """
    Calculate PnL metrics for each cycle
    
    Args:
        data: pandas DataFrame with columns ['time', 'closed_price', 'scaled_volume_USDC', 'scaled_volume_WETH']
        L: liquidity added for each cycle (float)
    
    Returns:
        pandas DataFrame with columns ['swap_fee_USDC', 'swap_fee_WETH', 'swap_fee_total', 'LVR', 'reward']
    """
    
    df = data[["time", "closed_price", "tick", "scaled_volume_WETH", "scaled_volume_USDC", "regime_SR_all"]].copy()
    df["action"] = 1
    
    df["equity"] = None
    df["liquidity"] = None
    df["fee"] = None
    df["LVR"] = None
    df["gas"] = df["action"] * gas_fee
    df["pnl"] = None

    equity = init_equity

    for i in range(1, df.shape[0]):

        df.loc[i, "equity"] = equity
           
        liquidity = total_y_amount_2_liquidity(equity, df.loc[i, "tick"], 500)
        fee = swap_fee(df.loc[i, "closed_price"], df.loc[i, "scaled_volume_WETH"], df.loc[i, "scaled_volume_USDC"], liquidity)[2]
        lvr = LVR(df.loc[i-1, "closed_price"], df.loc[i, "closed_price"], liquidity)
        gas = df.loc[i, "gas"]
        pnl = fee+lvr-gas

        df.loc[i, "liquidity"] = liquidity
        df.loc[i, "fee"] = fee
        df.loc[i, "LVR"] = lvr
        df.loc[i, "pnl"] = pnl

        equity = equity + pnl

        if equity < 0:
            equity = 0
            df =df.fillna(0)
            break

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.iloc[1:].to_csv(out_csv, index=False)

    return df.iloc[1:]

def benchmark_2(data, init_equity, gas_fee, method="micro", out_csv="./result/benchmark_2.csv"):

    """
    Calculate PnL metrics for each cycle
    
    Args:
        data: pandas DataFrame with columns ['time', 'closed_price', 'scaled_volume_USDC', 'scaled_volume_WETH']
        L: liquidity added for each cycle (float)
    
    Returns:
        pandas DataFrame with columns ['swap_fee_USDC', 'swap_fee_WETH', 'swap_fee_total', 'LVR', 'reward']
    """
    
    df = data[["time", "closed_price", "tick", "scaled_volume_WETH", "scaled_volume_USDC", "regime_SR_all", "regime_SR_micro"]].copy()
    df["pred"] = np.where(df["regime_SR_"+method].to_numpy() > 0, 1, 0)
    df["action"] = df["pred"].shift(1).fillna(0)
    
    df["equity"] = None
    df["liquidity"] = None
    df["fee"] = None
    df["LVR"] = None
    df["gas"] = df["action"] * gas_fee
    df["pnl"] = None

    equity = init_equity

    for i in range(df.shape[0]):

        df.loc[i, "equity"] = equity

        if df.loc[i, "action"] == 1:
           
           liquidity = total_y_amount_2_liquidity(equity, df.loc[i, "tick"], 500)
           fee = swap_fee(df.loc[i, "closed_price"], df.loc[i, "scaled_volume_WETH"], df.loc[i, "scaled_volume_USDC"], liquidity)[2]
           lvr = LVR(df.loc[i-1, "closed_price"], df.loc[i, "closed_price"], liquidity)
           gas = df.loc[i, "gas"]
           pnl = fee+lvr-gas

           df.loc[i, "liquidity"] = liquidity
           df.loc[i, "fee"] = fee
           df.loc[i, "LVR"] = lvr
           df.loc[i, "pnl"] = pnl

           equity = equity + pnl

        else:

            df.loc[i, "liquidity"] = 0
            df.loc[i, "fee"] = 0
            df.loc[i, "LVR"] = 0
            df.loc[i, "pnl"] = 0

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return df

if __name__ == "__main__":

    data_path = os.path.dirname(os.path.abspath(__file__)) + '/..' + '/data'
    data = pd.read_csv(data_path + '/hourly_features_fix_combined.csv')
    # benchmark_1(data, 100000, 5)
    # benchmark_2(data, 100000, 5)
   