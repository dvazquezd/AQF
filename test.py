import json
import pandas as pd
from lib.ApiClient import ApiClient
import lib.DataTransformer as transf
import lib.BricMortar as bm



data_frames = {'ticker': pd.DataFrame, 
               'sma': pd.DataFrame,
               'macd': pd.DataFrame,
               'rsi': pd.DataFrame,
               'sma_': pd.DataFrame,
               'unemployment': pd.DataFrame,
               'nonfarm': pd.DataFrame,
               'cpi': pd.DataFrame,
               'news': pd.DataFrame
               }

df_key = data_frames.keys()   
print(df_key)
