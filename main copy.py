import json
import pandas as pd
from lib.ApiClient import ApiClient
import lib.DataTransformer as transf
import lib.BricMortar as bm


def combine_data(df_historical,df_current,subset_columns):
   '''
   Combine same datasets types deleting duplicates
   '''
   df_combined = pd.concat([df_historical, df_current]).drop_duplicates(subset=subset_columns, keep="last")
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
            bm.write_csv(df_economic, f"data/df_{indicator}.csv")
            dfs[indicator] = df_economic
            print(f'Getting data: {{\'function\': {indicator}}} saved successfully!')
        else:
            print(f"Error fetching data for {indicator}")
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


def main():
    '''
    '''
    # Variables and configurations
    historical_needed = True
    symbols = ['NVDA']
    periods = [20, 40, 200]
    #technical_indicators = ['macd','rsi','sma']
    economic_indicators = ['unemployment', 'nonfarm_payroll', 'cpi']
    #all_dfs = ['df_macd','df_rsi','df_sma','df_ticker','df_news']
    topics = ['technology']#,'blockchain','financial_markets','economy_macro','economy_monetary','economy_fiscal']
    dataframes = {'ticker': pd.DataFrame(),
                  'macd': pd.DataFrame(),
                  'rsi': pd.DataFrame(),
                  'rsi_20': pd.DataFrame(),
                  'rsi_40': pd.DataFrame(),
                  'rsi_200': pd.DataFrame(),
                  'sma': pd.DataFrame(),
                  'sma_20': pd.DataFrame(),
                  'sma_40': pd.DataFrame(),
                  'sma_200': pd.DataFrame(),
                  'unemployment': pd.DataFrame(),
                  'cpi': pd.DataFrame(),
                  'nonfarm_payroll': pd.DataFrame(),
                  'news': pd.DataFrame(),
                  'merged_tec_info': pd.DataFrame()}
    h_dataframes = dataframes
    tec_columns = {
        'macd': ['ticker', 'datetime', 'MACD', 'MACD_Signal', 'MACD_Hist'],
        'rsi_200': ['ticker', 'datetime', 'rsi_200'],
        'rsi_40': ['ticker', 'datetime', 'rsi_40'],
        'rsi_20': ['ticker', 'datetime', 'rsi_20'],
        'sma_200': ['ticker', 'datetime', 'sma_200'],
        'sma_40': ['ticker', 'datetime', 'sma_40'],
        'sma_20': ['ticker', 'datetime', 'sma_20']
    }
    economic_columns = {
        'cpi': 'cpi',
        'nonfarm_payroll': 'nonfarm',
        'unemployment': 'unemployment'
    }
    

    #Objects
    client = ApiClient()

    months = bm.get_months(2024, historical_needed)
    dataframes = load_data(dataframes,client, symbols, months, periods)
    dataframes = load_economics(dataframes, client, economic_indicators)
    dataframes = load_news(dataframes, client, months, topics)
    dataframes = merge_datasets(dataframes, periods, tec_columns, economic_columns)

    if not historical_needed:
        h_dataframes = retrieve_data(h_dataframes)
        #dataframes = combine_data(dataframes, h_dataframes)
    
    print(dataframes)
    save_dataframes(dataframes)



    '''  
        df_ticker = combine_data(df_ticker, dh_ticker, subset_columns=['ticker','datetime'])
        df_macd = combine_data(df_macd, dh_mac, subset_columns=['ticker','datetime'])
        df_sma = combine_data(df_sma, dh_sma, subset_columns=['ticker','datetime','period'])
        df_rsi = combine_data(df_rsi, dh_rsi, subset_columns=['ticker','datetime','period'])
        df_news = combine_data(df_news, dh_news, subset_columns=['title','time_published','ticker','affected_topic'])

        bm.write_csv(df_ticker,'data/df_ticker.csv')
        bm.write_csv(df_macd,'data/df_macd.csv')
        bm.write_csv(df_sma,'data/df_sma.csv')
        bm.write_csv(df_rsi,'data/df_rsi.csv')
        bm.write_csv(df_news,'data/df_news.csv')

        df_aqf = merge_datasets(df_ticker, df_macd, df_rsi, df_sma, df_unemployment, df_nonfam, df_cpi)
        bm.write_csv(df_aqf,'data/df_aqf.csv')
    '''


if __name__ == "__main__":
    main()