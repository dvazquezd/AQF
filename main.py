import json
import pandas as pd
from lib.ApiClient import ApiClient
import lib.DataTransformer as transf
import lib.BricMortar as bm

# Variables y configuraciones
historical_needed = True
symbols = ['NVDA']
periods = [20, 40, 200]
economic_indicators = ['unemployment', 'nonfarm_payroll', 'cpi']
technical_indicators = ['macd','rsi','sma']
all_dfs = ['df_macd','df_rsi','df_sma','df_ticker','df_news']
topics = ['technology','blockchain','financial_markets','economy_macro','economy_monetary','economy_fiscal']


client = ApiClient()

def combine_data(df_historical,df_current,subset_columns):
   df_combined = pd.concat([df_historical, df_current]).drop_duplicates(subset=subset_columns, keep="last")
   return df_combined


def load_data(client, symbols, historical_needed,months):
    df_ticker, df_macd, df_rsi, df_sma = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for symbol in symbols:
        for month in months:
            # Obtener y transformar datos de ticker
            json_data_ticker = client.get_intraday_data(symbol, month)
            if json_data_ticker: 
                h_ticker = transf.transform_intraday(symbol,json_data_ticker)
                df_ticker = combine_data(df_ticker, h_ticker, subset_columns=['ticker','datetime'])

            json_data_macd = client.get_macd(symbol, month)
            if json_data_macd:
                h_macd = transf.transform_macd(symbol,json_data_macd)
                df_macd = combine_data(df_macd, h_macd, subset_columns=['ticker','datetime'])
            
            for period in periods:
                json_data_sma = client.get_sma(symbol, month,period)
                if json_data_sma:
                    h_sma = transf.transform_sma(symbol,json_data_sma,period)
                    df_sma = combine_data(df_sma, h_sma, subset_columns=['ticker','datetime','period'])

                json_data_rsi = client.get_rsi(symbol, month, period)
                if json_data_rsi:
                    h_rsi = transf.transform_rsi(symbol,json_data_rsi,period)
                    df_rsi = combine_data(df_rsi, h_rsi, subset_columns=['ticker','datetime','period'])

    if historical_needed:
        bm.write_csv(df_ticker,'data/df_ticker.csv')
        bm.write_csv(df_sma,'data/df_sma.csv')
        bm.write_csv(df_rsi,'data/df_rsi.csv')
        bm.write_csv(df_macd,'data/df_macd.csv')

    return df_ticker, df_sma, df_rsi, df_macd


def load_economics(client, indicators):
    economic_dataframes = {}
    for indicator in indicators:
        print(f"Fetching {indicator} data...")
        json_data = client.get_economic_indicator(indicator)
        if json_data:
            df_economic = transf.transform_economic_data(json_data)
            bm.write_csv(df_economic, f"data/df_{indicator}.csv")  # Sobrescribir siempre
            economic_dataframes[indicator] = df_economic
            print(f"{indicator.capitalize()} data saved successfully!")
        else:
            print(f"Error fetching data for {indicator}")
    return economic_dataframes['unemployment'], economic_dataframes['nonfarm_payroll'], economic_dataframes['cpi']


def load_news(client, months, topics):
    df_news = pd.DataFrame()
    for month in months:
        for topic in topics:
            time_from, time_to = bm.get_time_range(month)
            json_data = client.get_news_sentiment(topic,time_from, time_to)
            if json_data is not None:
                h_news = transf.transform_news_data(json_data, topic)
                df_news = combine_data(df_news, h_news, subset_columns=['title','time_published','ticker','affected_topic'])
    return df_news
    

def transform_indicators(df_tech, period, indicator_name):
    df_transformed = df_tech[df_tech['period'] == period][['ticker', 'datetime', indicator_name]].copy()
    df_transformed.rename(columns={indicator_name: f'{indicator_name}_{period}'},inplace = True)
    return df_transformed


# Preparar el dataset final
def merge_datasets(df_ticker, df_macd, df_rsi, df_sma, df_unemployment, df_nonfam, df_cpi):
    """Prepara el dataset final combinando datos de cotización con indicadores técnicos y económicos."""
    # Paso 1: Transformar RSI y SMA
    df_rsi_200 = transform_indicators(df_rsi, 200, 'RSI')
    df_rsi_40 = transform_indicators(df_rsi, 40, 'RSI')
    df_rsi_20 = transform_indicators(df_rsi, 20, 'RSI')

    df_sma_200 = transform_indicators(df_sma, 200, 'SMA').copy()
    df_sma_40 = transform_indicators(df_sma, 40, 'SMA').copy()
    df_sma_20 = transform_indicators(df_sma, 20, 'SMA')

    # Paso 2: Unir los datasets de indicadores técnicos al dataset de cotización
    df_final = pd.merge(df_ticker, df_macd[['ticker', 'datetime', 'MACD', 'MACD_Signal', 'MACD_Hist']], on=['ticker', 'datetime'], how='left')
    df_final = pd.merge(df_final, df_rsi_200[['ticker', 'datetime','RSI_200']], on=(['ticker', 'datetime']), how='left')
    df_final = pd.merge(df_final, df_rsi_40[['ticker', 'datetime','RSI_40']], on=(['ticker', 'datetime']), how='left')
    df_final = pd.merge(df_final, df_rsi_20[['ticker', 'datetime','RSI_20']], on=(['ticker', 'datetime']), how='left')
    df_final = pd.merge(df_final, df_sma_200[['ticker', 'datetime','SMA_200']], on=(['ticker', 'datetime']), how='left')
    df_final = pd.merge(df_final, df_sma_40[['ticker', 'datetime','SMA_40']], on=(['ticker', 'datetime']), how='left')
    df_final = pd.merge(df_final, df_sma_20[['ticker', 'datetime','SMA_20']], on=(['ticker', 'datetime']), how='left')

    df_final = pd.merge(df_final, df_cpi[['year_month', 'value']].rename(columns={'value': 'cpi'}), on='year_month', how='left')
    df_final = pd.merge(df_final, df_nonfam[['year_month', 'value']].rename(columns={'value': 'nonfarm'}), on='year_month', how='left')
    df_final = pd.merge(df_final, df_unemployment[['year_month', 'value']].rename(columns={'value': 'unemployment'}), on='year_month', how='left')

    return df_final


def retrieve_data(all_dfs):
    for df in all_dfs:
        if df == 'df_macd':
            dh_mac = bm.read_csv('data/df_macd.csv')
        if df == 'df_rsi':
            dh_rsi = bm.read_csv('data/df_rsi.csv')
        if df == 'df_sma':
            dh_sma = bm.read_csv('data/df_sma.csv')
        if df == 'df_ticker':
            dh_ticker = bm.read_csv('data/df_ticker.csv')
        if df == 'df_news':
            df_news = bm.read_csv('data/df_news.csv')

    return dh_ticker, dh_sma, dh_rsi, dh_mac, df_news


def main():
    if historical_needed:
        months = bm.get_months(2022,historical_needed)
        df_ticker, df_sma, df_rsi, df_macd = load_data(client, symbols, historical_needed,months)
        df_unemployment, df_nonfam, df_cpi = load_economics(client, economic_indicators)
        df_news = load_news(client,months, topics)
        bm.write_csv(df_news,'data/df_news.csv')
        df_aqf = merge_datasets(df_ticker, df_macd, df_rsi, df_sma, df_unemployment, df_nonfam, df_cpi)
        bm.write_csv(df_aqf,'data/df_aqf.csv')
    else:
        months = bm.get_months(2024,historical_needed)
        dh_ticker, dh_sma, dh_rsi, dh_mac, dh_news = retrieve_data(all_dfs)
        df_ticker, df_sma, df_rsi, df_macd = load_data(client, symbols, historical_needed,months)
        df_unemployment, df_nonfam, df_cpi = load_economics(client, economic_indicators)
        df_news = load_news(client,months, topics)
        
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


if __name__ == "__main__":
    main()