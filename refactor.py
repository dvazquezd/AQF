import json
import pandas as pd
from lib.ApiClient import ApiClient
import lib.DataTransformer as transf
import lib.BricMortar as bm

'''
    Configurations and variables
'''
historical_needed = False
symbols = ['NVDA']
periods = [20, 40, 200]
economic_indicators = ['unemployment', 'nonfarm_payroll', 'cpi']
technical_indicators = ['macd','rsi','sma']
all_dfs = ['df_macd','df_rsi','df_sma','df_ticker','df_news']
topics = ['technology','blockchain','financial_markets','economy_macro','economy_monetary','economy_fiscal']


def combine_data(df_historical,df_current,subset_columns):
   '''
    Combine same datasets types deleting duplicates
   '''
   df_combined = pd.concat([df_historical, df_current]).drop_duplicates(subset=subset_columns, keep="last")
   return df_combined


def load_data(client, symbols, historical_needed,months):
    '''
        Getting data for ticker direclty from alphavantage
        ticker quotation
        technical indicators as macd, rsi and sma
    '''
    data_frames = {key: pd.DataFrame() for key in ['df_ticker', 'df_macd', 'df_rsi', 'df_sma']}    

    for symbol in symbols:
        for month in months:
            # Obtener y transformar datos de ticker
            if json_data := client.get_intraday_data(symbol, month):
                transformed_data = transf.transform_intraday(symbol, json_data)
                data_frames['df_ticker'] = combine_data(data_frames['df_ticker'], transformed_data, ['ticker', 'datetime'])

            if json_data := client.get_macd(symbol, month):
                transformed_data = transf.transform_macd(symbol, json_data)
                data_frames['df_macd'] = combine_data(data_frames['df_macd'], transformed_data, ['ticker', 'datetime'])

            for period in periods:
                if json_data := client.get_sma(symbol, month, period):
                    transformed_data = transf.transform_sma(symbol, json_data, period)
                    data_frames['df_sma'] = combine_data(data_frames['df_sma'], transformed_data, ['ticker', 'datetime', 'period'])

                if json_data := client.get_rsi(symbol, month, period):
                    transformed_data = transf.transform_rsi(symbol, json_data, period)
                    data_frames['df_rsi'] = combine_data(data_frames['df_rsi'], transformed_data, ['ticker', 'datetime', 'period'])

    if historical_needed:
        for key, df in data_frames.items():
            bm.write_csv(df, f'data/{key}.csv')

    return data_frames['df_ticker'], data_frames['df_sma'],data_frames['df_rsi'],data_frames['df_macd']


def load_economics(client, indicators):
    '''
        Loading economics indicators as
        nonfarmpyaroll
        consumer price index
        unemployment
    '''
    economic_dataframes = {}

    for indicator in indicators:
        print(f"Getting data: Loading historic {indicator} data")
        if json_data := client.get_economic_indicator(indicator):
            df_economic = transf.transform_economic_data(json_data)
            bm.write_csv(df_economic, f"data/df_{indicator}.csv")
            economic_dataframes[indicator] = df_economic
            print(f"{indicator.capitalize()} data saved successfully!")
        else:
            print(f"Error fetching data for {indicator}")

    return (economic_dataframes.get(key) for key in indicators)


def load_news(client, months, topics):
    '''
        Loading news data for each topic
    '''
    df_news = pd.DataFrame()

    for month in months:
        for topic in topics:
            time_from, time_to = bm.get_time_range(month)
            if json_data := client.get_news_sentiment(topic, time_from, time_to):
                transformed_data = transf.transform_news_data(json_data, topic)
                df_news = combine_data(df_news, transformed_data, ['title', 'time_published', 'ticker', 'affected_topic'])

    return df_news


def transform_indicators(df_tech, period, indicator_name):
    '''
        Trnasforming each technical indicateor as rsi and msa
        into one colum foer each period (20, 40, 200)
    '''
    df_transformed = df_tech[df_tech['period'] == period][['ticker', 'datetime', indicator_name]].copy()
    df_transformed.rename(columns={indicator_name: f'{indicator_name}_{period}'},inplace = True)
    return df_transformed


def merge_datasets(df_ticker, df_macd, df_rsi, df_sma, df_unemployment, df_nonfarm, df_cpi, periods):
    '''
        Mergin all dataset in one dataset: ticker, technichal indicators and macro economics indicators
    '''
    technical_data = {}
    for indicator, df in {'RSI': df_rsi, 'SMA': df_sma}.items():
        for period in periods:
            technical_data[f'{indicator}_{period}'] = transform_indicators(df, period, indicator)
            print(technical_data)

    df_final = df_ticker.copy()
    
    for key, df_tech in technical_data.items():
        df_final = pd.merge(df_final, df_tech, on=['ticker', 'datetime'], how='left')
    
    df_final = pd.merge(df_final, df_macd[['ticker', 'datetime', 'MACD', 'MACD_Signal', 'MACD_Hist']], on=['ticker', 'datetime'], how='left')

    for econ_df, col_name in zip([df_cpi, df_nonfarm, df_unemployment], ['cpi', 'nonfarm', 'unemployment']):
        df_final = pd.merge(df_final, econ_df.rename(columns={'value': col_name}), on='year_month', how='left')

    return df_final            


def retrieve_data(all_dfs):
    """Recupera datos almacenados en archivos CSV."""
    data = {}
    for df_name in all_dfs:
        data[df_name] = bm.read_csv(f'data/{df_name}.csv')

    return tuple(data.get(name) for name in all_dfs)


def main():

    client = ApiClient()

    if historical_needed:
        months = bm.get_months(2024, historical_needed)
        df_ticker, df_sma, df_rsi, df_macd = load_data(client, symbols, historical_needed, months)
        df_unemployment, df_nonfarm, df_cpi = load_economics(client, economic_indicators)
        df_news = load_news(client, months, topics)
        bm.write_csv(df_news, 'data/df_news.csv')
        df_aqf = merge_datasets(df_ticker, df_macd, df_rsi, df_sma, df_unemployment, df_nonfarm, df_cpi, periods)
        bm.write_csv(df_aqf, 'data/df_aqf.csv')
    else:
        months = bm.get_months(2024, historical_needed)
        dh_ticker, dh_sma, dh_rsi, dh_mac, dh_news = retrieve_data(all_dfs)
        df_ticker, df_sma, df_rsi, df_macd = load_data(client, symbols, historical_needed, months)
        df_unemployment, df_nonfarm, df_cpi = load_economics(client, economic_indicators)
        df_news = load_news(client, months, topics)

        for key, df_new in zip(all_dfs, [df_ticker, df_macd, df_sma, df_rsi, df_news]):
            df_old = locals().get(f'dh_{key.split("_")[1]}')
            combined_df = combine_data(df_new, df_old, subset_columns=["ticker", "datetime"] if key != "df_news" else ["title", "time_published", "ticker", "affected_topic"])
            bm.write_csv(combined_df, f'data/{key}.csv')

        df_aqf = merge_datasets(df_ticker, df_macd, df_rsi, df_sma, df_unemployment, df_nonfarm, df_cpi, periods)
        bm.write_csv(df_aqf, 'data/df_aqf.csv')


if __name__ == "__main__":
    main()