import pandas as pd
from datetime import datetime

def manage_dates(df, date_type):
    """
    Processes date and time information in a DataFrame to extract and add new columns
    for date, year-month, and other related formats based on the provided parameters.
    Depending on the value of `date_type`, it applies specific transformations for
    handling date-based information within the input DataFrame.

    Args:
        df (DataFrame): The input pandas DataFrame containing a 'datetime' column with
            date and time information.
        date_type (Optional[Any]): A parameter to determine the processing logic. If
            None, specific operations are performed on the 'datetime' column. Otherwise,
            alternative date transformations are applied.

    Returns:
        DataFrame: The modified DataFrame with additional columns for date, year_month,
            and other processed date-related information.
    """

    if date_type is None:
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
    """
    Transform intraday data into a structured DataFrame format for further processing.

    This function takes stock symbol and raw intraday data, processes it into records
    that include datetime, ticker, and OHLCV values, and converts it into a DataFrame.
    It also applies date management to the resulting DataFrame.

    Args:
        symbol (str): The stock ticker symbol.
        data (dict): A dictionary containing raw intraday data, where keys are datetime
            strings and values are OHLCV (open, high, low, close, volume) data.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed and structured data
        with datetime, ticker, open, high, low, close, and volume columns.
    """
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
    """
    Transforms raw data into a DataFrame with Simple Moving Average (SMA) calculations
    for a given stock symbol and period. The function processes the provided data
    to extract relevant information, computes the SMA based on the input period,
    and reformats the output into a structured DataFrame.

    Parameters:
        symbol: str
            The stock ticker symbol.
        data: dict
            A dictionary containing timestamped data where each key
            is a datetime string and each value contains SMA information.
        period: int
            The time period over which the SMA is calculated.

    Returns:
        pd.DataFrame
            A pandas DataFrame containing the processed data with columns for the
            stock ticker, datetime, SMA value, and the associated period.

    Raises:
        None
    """
    records = [
        {
            'ticker': symbol,
            'datetime':   pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'sma': float(sma_data['SMA']),
            'period': period  
        }
        for datetime_str, sma_data in data.items()
    ]

    df = manage_dates(pd.DataFrame(records), None)

    return df


def transform_macd(symbol,data):
    """
    Transforms the given MACD data for a financial symbol into a DataFrame.

    This function processes the provided MACD data and converts it into
    a structured pandas DataFrame format. It standardizes the date-time format,
    associates the data with the given ticker symbol, and prepares it
    for further analysis or processing by managing the dates.

    Args:
        symbol (str): The financial symbol (e.g., stock ticker) associated with
            the MACD data.
        data (Dict[str, Dict[str, float]]): A dictionary where the keys are
            date-time strings, and the values are dictionaries containing the
            MACD metrics ('MACD', 'MACD_Signal', 'MACD_Hist').

    Returns:
        pd.DataFrame: A pandas DataFrame containing the transformed MACD data,
            with columns for date-time, symbol, and MACD metrics, and standardized
            date-time formatting.
    """
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
    """
    Transforms raw RSI data into a structured DataFrame format. This function processes RSI (Relative Strength Index) data
    associated with a particular ticker symbol and a specific period, reformats the data, and returns it as a DataFrame after
    managing the date fields.

    Args:
        symbol: The stock or asset ticker symbol as a string.
        data: A dictionary where keys are datetime strings and values are dictionaries containing RSI data with an 'RSI' key.
        period: The RSI calculation period as an integer.

    Returns:
        A pandas DataFrame containing the processed RSI data, with columns:
        - ticker (same as the input symbol)
        - datetime (formatted as 'YYYY-MM-DD HH:MM')
        - rsi (RSI value as a float)
        - period (RSI calculation period).

    Raises:
        Does not explicitly raise exceptions, but may encounter errors during data handling operations or invalid input
        data structures.
    """
    records = [
        {
            'ticker': symbol,
            'datetime':  pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'rsi': float(rsi_data['RSI']),
            'period': period 
        }
        for datetime_str, rsi_data in data.items()
    ]

    df = manage_dates(pd.DataFrame(records),None)
    
    return df


def transform_economic_data(data):
    """
    Transforms economic data into a structured DataFrame format.

    The function processes raw economic data by converting date strings into datetime objects and
    values into floats. It then compiles the processed data into a pandas DataFrame and processes
    dates with the specified context.

    Args:
        data (dict): A dictionary containing economic data. The dictionary should have a 'data'
            key with a list of dictionaries, each containing 'date' and 'value' keys.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed economic data with formatted dates
        and values.

    Raises:
        KeyError: If the expected keys 'data', 'date', or 'value' are not found in the input data.
        ValueError: If the value cannot be converted to float or the date is not in a parseable
            format.
    """
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
    """
    Transforms a nested data structure containing news records into a structured pandas
    DataFrame. This transformation extracts specific values, modifies datetime formats,
    and maps nested ticker and topics data into separate rows for comprehensive analysis.

    Parameters:
        data (list): A list of dictionaries containing nested news data and attributes.
        topic (str): A string representing the topic assigned to all extracted records.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted and structured news data.
    """
    records = []

    feed_items = data
    
    for item in feed_items:
        title = item['title'].replace(',', '')
        
        
        # Convertir time_published al formato deseado
        timepublished =  pd.to_datetime(datetime.strptime(item['time_published'][:8] + ' ' + item['time_published'][9:11], '%Y%m%d %H').strftime('%Y-%m-%d %H:00:00'))
        
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
                    title, timepublished, overall_sentiment_score, 
                    overall_sentiment_label, ticker, relevance_score, 
                    ticker_sentiment_score, ticker_sentiment_label, 
                    affected_topic, affected_topic_relevance_score, topic
                ])
    
    # Crear el DataFrame final con las columnas especificadas
    df_transformed = pd.DataFrame(records, columns=[
        'title', 'timepublished', 'overall_sentiment_score', 
        'overall_sentiment_label', 'ticker', 'relevance_score', 
        'ticker_sentiment_score', 'ticker_sentiment_label', 
        'affected_topic', 'affected_topic_relevance_score', 'topic'
    ])

    df_transformed.rename(columns={'timepublished': 'datetime'}, inplace=True)

    return df_transformed