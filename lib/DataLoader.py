import json
import pandas as pd
import DataTransformer as transf
import BricMortar as bm


def combine_dataframes(df_historical,df_current,f_dataframes, combine_configuration):
   '''
   Combine same datasets types deleting duplicates
   '''
   all_keys = df_current.keys()

   for enconomic in ['unemployment','nonfarm_payroll','cpi']:
       f_dataframes[enconomic] = df_current[enconomic]
       all_keys.remove(enconomic)
       
   for key in all_keys:
       df_combined = pd.concat([df_historical[key], df_current[key]]).drop_duplicates(subset=combine_configuration[key], keep='last')
       f_dataframes[key] = df_combined   
   
   return f_dataframes


def combine_data(df_historical,df_current,subset_columns):
   '''
        Combine same datasets types deleting duplicates
   '''
   df_combined = pd.concat([df_historical, df_current]).drop_duplicates(subset=subset_columns, keep='last')
   return df_combined


def load_data(dfs,client, symbols, months, periods):
    '''
    '''
    for symbol in symbols:
        for month in months:
            # Obtener y transformar datos de ticker
            json_data_ticker = client.get_intraday_data(symbol, month)
            if json_data_ticker: 
                h_ticker = transf.transform_intraday(symbol,json_data_ticker)
                dfs['ticker'] = combine_data(dfs['ticker'], h_ticker, subset_columns=['ticker','datetime'])

            json_data_macd = client.get_macd(symbol, month)
            if json_data_macd:
                h_macd = transf.transform_macd(symbol,json_data_macd)
                dfs['macd'] = combine_data( dfs['macd'], h_macd, subset_columns=['ticker','datetime'])
            
            for period in periods:
                json_data_sma = client.get_sma(symbol, month,period)
                if json_data_sma:
                    h_sma = transf.transform_sma(symbol,json_data_sma,period)
                    dfs['sma'] = combine_data(dfs['sma'], h_sma, subset_columns=['ticker','datetime','period'])

                json_data_rsi = client.get_rsi(symbol, month, period)
                if json_data_rsi:
                    h_rsi = transf.transform_rsi(symbol,json_data_rsi,period)
                    dfs['rsi'] = combine_data(dfs['rsi'], h_rsi, subset_columns=['ticker','datetime','period'])

    return dfs


def load_economics(dfs, client, indicators):
    '''
    '''
    for indicator in indicators:
        print(f'Getting data: {{\'function\': {indicator}}}')
        json_data = client.get_economic_indicator(indicator)
        if json_data:
            df_economic = transf.transform_economic_data(json_data)
            bm.write_csv(df_economic, f'data/df_{indicator}.csv')
            dfs[indicator] = df_economic
            print(f'Getting data: {{\'function\': {indicator}}} saved successfully!')
        else:
            print(f'Error fetching data for {indicator}')
    return dfs 


def load_news(dfs,client, months, topics):
    '''
    '''
    for month in months:
        for topic in topics:
            time_from, time_to = bm.get_time_range(month)
            json_data = client.get_news_sentiment(topic,time_from, time_to)
            if json_data is not None:
                h_news = transf.transform_news_data(json_data, topic)
                dfs['news'] = combine_data(dfs['news'], h_news, subset_columns=['title','time_published','ticker','affected_topic'])
    
    return dfs
    

def transform_indicators(dfs, period, tech_indicator):
    '''
    '''
    df_transformed =  dfs[tech_indicator][dfs[tech_indicator]['period'] == period][['ticker', 'datetime', tech_indicator]].copy()
    df_transformed.rename(columns={tech_indicator: f'{tech_indicator}_{period}'},inplace = True)
    return df_transformed


def merge_datasets(dfs, periods, tec_columns, economic_columns):
    '''
    Prepara el dataset final combinando datos de cotización con indicadores técnicos y económicos.
    '''
    # Paso 1: Transformar RSI y SMA
    for period in periods:
        for tech_indicator in ['rsi','sma']:
            dfs[f'{tech_indicator}_{period}'] = transform_indicators(dfs, period, tech_indicator)

    # Paso 2: Unir los datasets de indicadores técnicos al dataset de cotización
    dfs['merged_tec_info'] = dfs['ticker']
    for key, cols in tec_columns.items():
        dfs['merged_tec_info'] = pd.merge(dfs['merged_tec_info'], dfs[key][cols], on=['ticker', 'datetime'], how='left')

    for key, col_name in economic_columns.items():
         dfs['merged_tec_info'] = pd.merge(
             dfs['merged_tec_info'],
            dfs[key][['year_month', 'value']].rename(columns={'value': col_name}),
            on='year_month',
            how='left'
        )
             
    return dfs


def retrieve_data(dfs):
    '''
    '''
    for df in dfs.keys():
        dfs[df] = bm.read_csv(f'data/df_{df}.csv')

    return dfs


def save_dataframes(dfs):
    '''
    '''
    all_keys = dfs.keys()
    for df in all_keys:
        bm.write_csv(dfs[df],f'data/df_{df}.csv')


def load_config():
    '''
    '''
    with open('../config/config.json', 'r') as config_file:
        config = json.load(config_file)

    return config