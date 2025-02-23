import pandas as pd
from datetime import datetime

def manage_dates(df, date_type):
    """
    Processes date-related columns in a pandas DataFrame based on the specified
    date type parameter. Adjusts and formats the 'datetime', 'date', and 'year_month'
    columns depending on the input `date_type`. This function helps standardize date
    manipulation and transformation within the DataFrame.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame containing a 'datetime' column to process.
    date_type : Optional
        A parameter that determines the processing logic for the date columns.
        If None, a specific format is applied to the columns; otherwise,
        alternative logic is used.

    Returns:
    pandas.DataFrame
        The modified DataFrame with the adjusted and formatted date-related columns.
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

def transform_intraday(symbol, data):
    """
    Transforms intraday stock market data into a structured pandas DataFrame format. Converts raw data of
    time-series with OHLCV (Open, High, Low, Close, Volume) into a standardized, cleaner format for further
    processing. Ensures proper datetime formatting and includes the stock ticker for identification.

    Parameters:
        symbol (str): The stock ticker or symbol associated with the provided data.
        data (dict): Dictionary containing OHLCV data for different timestamps. The keys are datetime strings,
                     and the values are dictionaries with OHLCV values.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns ['datetime', 'ticker', 'open', 'high', 'low', 'close',
                                                       'volume']. The DataFrame rows represent individual
                       records of OHLCV data at specific timestamps.

    Raises:
        TypeError: Raises if the inputs are not of the correct types: 'symbol' as string and 'data' as
                   a dictionary.

    Notes:
        The datetime values are converted and formatted into a consistent string format 'YYYY-MM-DD HH:MM'
        for ease of processing.
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
    Transforms price data into a DataFrame containing Simple Moving Averages (SMA) for a specific period.

    This function iterates through the given dictionary, calculating SMAs for
    a specified financial symbol and organizes the results into a pandas DataFrame.
    It reformats the date-time and merges data as part of the transformation.

    Args:
        symbol (str): The ticker symbol representing the financial instrument.
        data (dict): A dictionary containing date-time strings as keys and
            SMA calculation results as values.
        period (int): The time period for which the SMA is calculated.

    Returns:
        pandas.DataFrame: A DataFrame containing records with columns such as
        'ticker', 'datetime', 'sma', and 'period'.
    """
    records = [
        {
            'ticker': symbol,
            'datetime': pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'sma': float(sma_data['SMA']),
            'period': period
        }
        for datetime_str, sma_data in data.items()
    ]

    df = manage_dates(pd.DataFrame(records), None)

    return df

def transform_macd(symbol, data):
    """
    Transforms MACD (Moving Average Convergence Divergence) data for a given symbol into a DataFrame.

    This function iterates over the provided MACD data, processes each entry to create a dictionary
    containing the ticker symbol, datetime (formatted as a string), and MACD data (MACD, MACD_Signal,
    MACD_Hist). The list of dictionaries is then converted into a DataFrame, and dates are managed using
    the manage_dates function.

    Parameters:
        symbol (str): The ticker symbol associated with the MACD data.
        data (dict): Dictionary containing MACD data for specific datetime keys. Each key is a datetime
                    string, and the value is another dictionary with keys 'MACD', 'MACD_Signal',
                    and 'MACD_Hist'.

    Returns:
        pandas.DataFrame: A DataFrame containing the transformed MACD data with the provided symbol
                          and datetime formatted as 'YYYY-MM-DD HH:MM'.
    """
    records = [
        {
            'ticker': symbol,
            'datetime': pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
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
    Transforms RSI data into a structured DataFrame format.

    This function takes RSI-related data, along with a symbol and period,
    and constructs a structured DataFrame. The input data consists of a symbol,
    a dictionary mapping datetime strings to RSI data, and a period. The function
    processes the input, formats the datetime strings, and computes a DataFrame
    with enhanced structure and date management.

    Args:
        symbol: str. The stock or financial instrument's symbol.
        data: dict. A dictionary where keys are datetime strings
              and values are dictionaries containing RSI data.
        period: int. The look-back period for calculating RSI.

    Returns:
        pd.DataFrame: A DataFrame containing the transformed RSI information,
        including ticker, formatted datetime, RSI values, and the specified period.
    """
    records = [
        {
            'ticker': symbol,
            'datetime': pd.to_datetime(datetime_str).strftime('%Y-%m-%d %H:%M'),
            'rsi': float(rsi_data['RSI']),
            'period': period
        }
        for datetime_str, rsi_data in data.items()
    ]

    df = manage_dates(pd.DataFrame(records), None)

    return df

def transform_economic_data(data):
    """
    Transform economic data into a processed DataFrame.

    This function processes raw economic data provided in a structured format
    and prepares it for analysis. It parses datetime strings into Pandas
    Timestamps and converts numerical values to float type. The resulting
    data is then transformed into a Pandas DataFrame and additional date-related
    operations are handled via the `manage_dates` function.

    Args:
        data (dict): A dictionary containing raw economic data with at least
                     the following structure:
                     - data (list): A list of dictionaries, where each dictionary
                                   contains:
                                   - 'date' (str): A date string.
                                   - 'value': A numerical value that can be cast
                                              to a float.

    Returns:
        pd.DataFrame: A Pandas DataFrame with the processed economic data,
        including managed date operations.
    """
    records = [
        {
            'datetime': pd.to_datetime(entry['date']),
            'value': float(entry['value'])
        }
        for entry in data['data']
    ]

    df = manage_dates(pd.DataFrame(records), 'economics')

    return df

def transform_news_data(data, topic):
    """
    Transforms news data into a structured and consistent format for analysis.

    This function processes raw news data, extracting and transforming relevant
    information into a Pandas DataFrame. The input data is expected to contain
    various elements, including titles, publication times, sentiment scores,
    tickers, and topics. The function performs tasks such as datetime conversion,
    data aggregation, and score mapping to construct a comprehensive dataset.

    Args:
        data (list): A list of dictionaries representing news entries. Each entry
            should contain fields including 'title', 'time_published',
            'overall_sentiment_score', 'overall_sentiment_label',
            'ticker_sentiment', and 'topics'.
        topic (str): The topic to associate with each transformed record for
            additional grouping or classification.

    Returns:
        pandas.DataFrame: A DataFrame containing the transformed news data,
            structured with columns such as 'title', 'datetime',
            'overall_sentiment_score', 'overall_sentiment_label', 'ticker',
            'relevance_score', 'ticker_sentiment_score', 'ticker_sentiment_label',
            'affected_topic', 'affected_topic_relevance_score', and 'topic'.
    """
    records = []
    feed_items = data

    for item in feed_items:
        title = item['title'].replace(',', '')

        # Convert time_published to the desired format
        timepublished = pd.to_datetime(
            datetime.strptime(item['time_published'][:8] + ' ' + item['time_published'][9:11], '%Y%m%d %H')
            .strftime('%Y-%m-%d %H:00:00')
        )

        overall_sentiment_score = item['overall_sentiment_score']
        overall_sentiment_label = item['overall_sentiment_label']

        # Iterate through each ticker sentiment and each topic
        for ticker_data in item['ticker_sentiment']:
            ticker = ticker_data['ticker']
            relevance_score = ticker_data['relevance_score']
            ticker_sentiment_score = ticker_data['ticker_sentiment_score']
            ticker_sentiment_label = ticker_data['ticker_sentiment_label']

            for topic_data in item['topics']:
                affected_topic = topic_data['topic']
                affected_topic_relevance_score = topic_data['relevance_score']

                # Add the row with all information, including the new topic columns
                records.append([
                    title, timepublished, overall_sentiment_score,
                    overall_sentiment_label, ticker, relevance_score,
                    ticker_sentiment_score, ticker_sentiment_label,
                    affected_topic, affected_topic_relevance_score, topic
                ])

    # Create the final DataFrame with specified columns
    df_transformed = pd.DataFrame(records, columns=[
        'title', 'timepublished', 'overall_sentiment_score',
        'overall_sentiment_label', 'ticker', 'relevance_score',
        'ticker_sentiment_score', 'ticker_sentiment_label',
        'affected_topic', 'affected_topic_relevance_score', 'topic'
    ])

    df_transformed.rename(columns={'timepublished': 'datetime'}, inplace=True)

    return df_transformed
