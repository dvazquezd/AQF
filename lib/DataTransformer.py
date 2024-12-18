import pandas as pd
from datetime import datetime

def manage_dates(df, type):

    if type is None:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M')
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        df['year_month'] = pd.to_datetime(df['datetime']).dt.to_period('M').astype(str)
    else:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
        df['year_month'] = df['year_month'] + 1
        df['year_month'] = df['year_month'].astype(str)
        
    return df


# Funciones específicas para tipos de datos
def transform_intraday(symbol,data):
    records = [
        {
            'datetime': pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'ticker': symbol,
            'open': float(ohlcv['1. open']),
            'high': float(ohlcv['2. high']),
            'low': float(ohlcv['3. low']),
            'close': float(ohlcv['4. close']),
            'volume': int(ohlcv['5. volume'])
        }
        for datetime_str, ohlcv in data.items()
    ]

    df = manage_dates(pd.DataFrame(records), None)

    return df


def transform_sma(symbol, data, period):
    """Transforma los datos RSI en un DataFrame, añadiendo el periodo."""
    records = [
        {
            'ticker': symbol,
            'datetime':   pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'SMA': float(sma_data['SMA']),
            'period': period  
        }
        for datetime_str, sma_data in data.items()
    ]

    df = manage_dates(pd.DataFrame(records), None)

    return df


def transform_macd(symbol,data):
    records = [
        {
            'ticker': symbol,
            'datetime':  pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'MACD': macd['MACD'],
            'MACD_Signal': macd['MACD_Signal'],
            'MACD_Hist': macd['MACD_Hist']
        }
    for datetime_str, macd in data.items()
    ]

    df = manage_dates(pd.DataFrame(records), None)

    return df
    

def transform_rsi(symbol, data, period):
    """Transforma los datos RSI en un DataFrame, añadiendo el periodo."""
    records = [
        {
            'ticker': symbol,
            'datetime':  pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'RSI': float(rsi_data['RSI']),
            'period': period 
        }
        for datetime_str, rsi_data in data.items()
    ]

    df = manage_dates(pd.DataFrame(records),None)
    
    return df


def transform_economic_data(data):
    """Transforma los datos de un indicador económico en un DataFrame."""
    records = [
        {
            'datetime': pd.to_datetime(entry['date']),
            'value': float(entry['value'])
        }
        for entry in data['data']
    ]

    df = manage_dates(pd.DataFrame(records),'economics')
     
    return df


def transform_news_data(data, topic):
    records = []

    feed_items = data
    
    for item in feed_items:
        title = item['title'].replace(',', '')
        
        
        # Convertir time_published al formato deseado
        time_published = datetime.strptime(item['time_published'][:8] + ' ' + item['time_published'][9:11], '%Y%m%d %H').strftime('%Y-%m-%d %H:00:00')
        
        overall_sentiment_score = item['overall_sentiment_score']
        overall_sentiment_label = item['overall_sentiment_label']
        
        # Recorremos cada ticker_sentiment y cada topic
        for ticker_data in item['ticker_sentiment']:
            ticker = ticker_data['ticker']
            relevance_score = ticker_data['relevance_score']
            ticker_sentiment_score = ticker_data['ticker_sentiment_score']
            ticker_sentiment_label = ticker_data['ticker_sentiment_label']
            
            for topic_data in item['topics']:
                affected_topic = topic_data['topic']
                affected_topic_relevance_score = topic_data['relevance_score']
                
                # Agregar la fila con toda la información, incluyendo las nuevas columnas de topics
                records.append([
                    title, time_published, overall_sentiment_score, 
                    overall_sentiment_label, ticker, relevance_score, 
                    ticker_sentiment_score, ticker_sentiment_label, 
                    affected_topic, affected_topic_relevance_score, topic
                ])
    
    # Crear el DataFrame final con las columnas especificadas
    df_transformed = pd.DataFrame(records, columns=[
        'title', 'time_published', 'overall_sentiment_score', 
        'overall_sentiment_label', 'ticker', 'relevance_score', 
        'ticker_sentiment_score', 'ticker_sentiment_label', 
        'affected_topic', 'affected_topic_relevance_score', 'topic'
    ])
    
    return df_transformed