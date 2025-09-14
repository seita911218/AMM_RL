import numpy as np
import pandas as pd
from uniswap import *
import tqdm

import talib


# Liquidity------------------------------------------------------------------------------------------------------------------------------------------------
def liq_profile(df,tick_min, tick_max,tick_spacing=10):
    total_range = np.arange(tick_min, tick_max + 1, tick_spacing)
    T = len(df)
    # The compplete reseult will be separate into 20 or 21 filessegements
    N = int(T/20 )
    prev_values = np.zeros(len(total_range)) 
    rows_list = []
    t_ini = 0
    for t in tqdm.tqdm(range(T)):
        LP_range = np.arange(df.loc[t,'tickLower'],df.loc[t,'tickUpper'],tick_spacing)
        mask =np.isin(total_range,LP_range)
        prev_values[mask] += df.loc[t,'amount']
        rows_list.append(prev_values.copy())
        if (t % N==0) and (t>0):
            i = int(t/N)
            df_liq = pd.DataFrame(rows_list,columns=total_range)
            time = df.iloc[t_ini:t+1][['time','timestamp','logIndex']].values
            df_liq.loc[:,['time','timestamp','logIndex']] = time
            df_liq = df_liq[['time','timestamp','logIndex']  +list(total_range)]
            df_liq.to_csv(f'./data/df_liq_{i}.parquet')
            t_ini = t+1
            rows_list = []
    if len(rows_list)!=0:
        df_liq = pd.DataFrame(rows_list,columns=total_range)      
        time = df.iloc[t_ini:][['time','timestamp','logIndex']].values
        df_liq.loc[:,['time','timestamp','logIndex']] = time
        df_liq = df_liq[['time','timestamp','logIndex']  +list(total_range)]
        df_liq.to_csv(f'./data/df_liq_{i+1}.parquet')

def get_liq(df_liq,df_swap):
    event_map = pd.concat([df_liq.iloc[:,:3].assign(event='mb'),df_swap.iloc[:,:3].assign(event='swap') ] ).reset_index().sort_values(['timestamp','logIndex'])
    event_map.rename(columns={'index':'label'},inplace=True)

    event_map['mb_label'] = event_map.apply(lambda r: (r['label']) if r['event'] == 'mb' else None, axis=1)
    event_map['mb_label']= event_map['mb_label'].ffill().astype(int)
    event_map=event_map[event_map['event']=='swap'].copy()
    df_swap['mb_label'] =event_map['mb_label'].values
    del event_map

    idx_array = df_swap['mb_label'].values
    ticks_array = df_swap['tick'].values
    liq_array=[]
    for idx,tick in tqdm.tqdm(zip(idx_array,ticks_array)):
        tick = tick_2_tickspacing(tick)
        liq_array.append(df_liq.iloc[idx][str(tick)])
    df_swap['liquidity'] = get_liq(df_liq,df_swap)
    df_swap.drop(columns=['logIndex','mb_label'],inplace=True)
# Features-------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_candles(price_array, interval_length: int):
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []

    interval_length = interval_length - 1

    for i in range(len(price_array)):
        if i < interval_length:
            start_idx = 0
        else:
            start_idx = i - interval_length

        data_slice = price_array[start_idx : i+1]
        
        open_price = data_slice[0]
        high_price = max(data_slice)
        low_price = min(data_slice)
        close_price = data_slice[-1]

        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
        close_prices.append(close_price)
        
    open_prices = np.array(open_prices, dtype=np.float64)
    high_prices = np.array(high_prices, dtype=np.float64)
    low_prices = np.array(low_prices, dtype=np.float64)
    close_prices = np.array(close_prices, dtype=np.float64)

    return open_prices, high_prices, low_prices, close_prices

def ewm_features(df,alpha=0.05):
     price = df['closed_price'].copy()
     logR = np.log(price / price.shift(1))
     hourly_ewma_R = logR.ewm(alpha=alpha, adjust=False).mean()
     hourly_ewm_std = logR.ewm(alpha=alpha, adjust=False).std()


     return hourly_ewma_R,hourly_ewm_std


def features_resample(df_swap,df_mb,quote=token1,T='1h',alpha=0.05):
     def last(x):
          if x.empty:
               return np.nan   
          return x.ffill().iloc[-1]
     df =df_swap.drop(columns=['timestamp'])
     if ~(df['liquidity']>0).any():
           raise ValueError('Invalid liquidity value, must be positive')
    
     df['interval_swap'] = df['time'].diff().dt.seconds
     df.set_index('time',inplace=True)
     df[f'scaled_volume_{token0}'] = np.maximum(df[token0] / df['liquidity'], 0)
     df[f'scaled_volume_{token1}'] = np.maximum(df[token1] / df['liquidity'], 0)
     if quote== token1:
          net_vol = df[token1].copy()
     else:
          net_vol = df[token0].copy()

     total_vol = np.abs(net_vol)
     df['net_volume'] = net_vol   
     df['total_volume'] = total_vol   
     df['scaled_total_volume'] = total_vol/df['liquidity']   
     df['n_swap'] = 1
     
     df=df.resample(T,label='right').agg({
          'net_volume':'sum' ,
          'total_volume':'sum',
          f'scaled_volume_{token0}': 'sum', 
          f'scaled_volume_{token1}': 'sum', 
          'scaled_total_volume':'sum',
          'n_swap': 'sum',
          'closed_price': last,
          'interval_swap': 'mean',
          'liquidity': last,
          'tick': last
            })
     df.loc[:,'net_volume'] = df['net_volume']/df['total_volume']
     df.rename(columns={'net_volume':'volume_imbalance'},inplace=True)
     df.drop(columns=['total_volume'],inplace=True)
     hourly_ewma_R,hourly_ewm_std = ewm_features(df,alpha)
     df['R_ewma'] = hourly_ewma_R.values

     df['volatility_ewm'] = hourly_ewm_std.values
     #TA features

     prices = df['closed_price'].copy()
     df['ma24'] = prices.rolling(24).mean()
     df['ma168'] = prices.rolling(168).mean()

     bb_upper, bb_middle, bb_lower = talib.BBANDS(prices.values, matype=talib.MA_Type.T3)
     df['bb_upper'] = bb_upper
     df['bb_middle'] = bb_middle
     df['bb_lower'] = bb_lower
     open_price, high_price,low_price,closed_price=calculate_candles(prices.values,12)
     df['adxr'] = talib.ADX(high_price, low_price, closed_price, timeperiod=14) 
     df['dx'] = talib.DX(high_price, low_price, closed_price, timeperiod=14)     
     #mint/burn events
     df2 = df_mb.set_index('time')[['amount']].copy()
     df2['n_mb'] = 1
     df2=df2.resample(T,label='right').apply({'n_mb':'sum'})
     df2 =df2[df2.index>= df.index[0]]
     return pd.concat([df,df2],axis=1).dropna()



