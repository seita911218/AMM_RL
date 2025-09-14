import pandas as pd
from datetime import datetime,timezone
import requests
from typing import Iterable



# Fetch data-----------------------------------------------------------------------------------------------------------------------------------------------
def query_data(event_type:str,
                attributes_list: Iterable[str],
                pool_id,
                url,
                start_ts = int(datetime(2021, 5, 5, tzinfo=timezone.utc).timestamp()),
                end_ts   = int(datetime(2024, 5, 1, tzinfo=timezone.utc).timestamp()),
                batch_size = 1000         ):
    '''
Using skip for query pagination may result in the error like 'bad indexers: BadRespons unattestable response: The skip argument must be between 0 and 10000'. 
If this is the case, one needs to run again starting from the last datapoint, or use more efficient method.
    '''
    attributes =""
    for attr in attributes_list:
        attributes=attributes + '\n' + f' {attr}'
    df_list = []
    skip = 0
    while True:
        query = f"""
        {{
            {event_type}(
            first: {batch_size},
            skip: {skip},
            orderBy: timestamp,
            orderDirection: asc,
            where: {{
                pool: "{pool_id}",
                timestamp_gte: {start_ts} 
                timestamp_lte: {end_ts}
            }}
            ) {{
            {attributes}
            }}
        }}
        """
        res = requests.post(url, json={"query": query}).json()
        data = res["data"][event_type]        
        if  len(data)==0:
            break

        df = pd.DataFrame(data)
        df['timestamp'] = df['timestamp'].astype(int)
        df_list.append(df)

        last_ts = df['timestamp'].max()
        skip += len(df)

        if last_ts > end_ts:
            break
        ts_current = float(df['timestamp'].iloc[-1])
        print(f'\rcurrent ts:{ts_current}, end:{end_ts}',end='')
    return pd.concat(df_list)
