import pandas as pd
import utils.utils as ut
import loader.DataTransformer as Transformer
from datetime import datetime
from utils.utils import get_time_now

def combine_dataframes(df_historical, df_current, f_dataframes, combine_configuration):
    """
    Combine historical and current dataframes based on a configuration and updates the provided
    dictionary with the merged dataframes. This function separates economic indicators data before
    merging the remaining datasets and ensures no duplicate entries based on the given configuration.

    Parameters:
        df_historical (dict): Dictionary containing historical dataframes for different datasets.
        df_current (dict): Dictionary containing current dataframes for different datasets.
        f_dataframes (dict): Dictionary where the merged dataframes will be stored.
        combine_configuration (dict): Configuration used to determine key columns for merging
                                       and identifying duplicates.

    Returns:
        dict: The updated f_dataframes dictionary with assigned or merged datasets.
    """
    all_keys = list(df_current.keys())

    # Assign economic indicators separately
    for economic in ['unemployment', 'nonfarm_payroll', 'cpi']:
        f_dataframes[economic] = df_current[economic]
        all_keys.remove(economic)

    # Merge remaining datasets
    for key in all_keys:
        df_combined = pd.concat([df_historical[key], df_current[key]]).drop_duplicates(subset=combine_configuration[key],
                                                                                       keep='last')
        f_dataframes[key] = df_combined.sort_values(by='datetime', ascending=True)

    return f_dataframes

def combine_data(df_historical, df_current, subset_columns):
    """
    Combine historical and current data, ensuring no duplicate records based on specified columns.

    This function merges two DataFrames, applying specific logic depending on whether the
    dataframes are empty or contain data. It drops duplicate records based on the
    provided subset of columns and ensures valid and non-duplicate records in the
    final combined DataFrame.

    Args:
        df_historical (pd.DataFrame): A DataFrame containing historical data. It may be empty.
        df_current (pd.DataFrame): A DataFrame containing the current or new data. It may also be empty.
        subset_columns (list): A list of column names used to identify and drop duplicate records
            from the combined DataFrame.

    Returns:
        pd.DataFrame: A single DataFrame combining both historical and current data without duplicates.
        If the input DataFrames are both empty, an empty DataFrame is returned.
    """
    df_historical = df_historical.dropna(how='all') if not df_historical.empty else pd.DataFrame()
    df_current = df_current.dropna(how='all') if not df_current.empty else pd.DataFrame()

    if df_historical.empty and df_current.empty:
        return pd.DataFrame()

    if df_historical.empty and not df_current.empty:
        return df_current

    if not df_historical.empty and df_current.empty:
        return df_historical

    # Concatenate only if there is valid data
    df_combined = pd.concat([df_historical, df_current]).drop_duplicates(subset=subset_columns, keep='last')

    return df_combined

def load_data(dfs, client, symbols, months, periods):
    """
    Fetches and processes financial data for specified symbols and timeframes. The function retrieves
    ticker, MACD, SMA, and RSI indicators for the provided symbols and for multiple specified periods
    and months. Data is transformed using a `Transformer` class before being combined into DataFrames.

    Arguments:
        dfs (dict): A dictionary where the processed DataFrames (e.g., 'ticker', 'macd', 'sma', 'rsi')
        are stored and updated.
        client: An object responsible for fetching raw JSON data for financial symbols from an external
        source.
        symbols (list[str]): A list of stock or asset symbols for which data will be fetched and processed.
        months (list[str]): A list of month identifiers (e.g., '202301' for January 2023) defining the
        timeframes of required data.
        periods (dict): A dictionary specifying multiple periods for which SMA and RSI need to be calculated.
        Keys include 'sma' and 'rsi', each pointing to their respective lists of periods.

    Returns:
        dict: A dictionary containing updated DataFrames for each calculated indicator.

    Raises:
        None
    """
    for symbol in symbols:
        for month in months:
            # Fetch and transform ticker data
            json_data_ticker = client.get_intraday_data(symbol, month)
            if json_data_ticker:
                h_ticker = Transformer.transform_intraday(symbol, json_data_ticker)
                dfs['ticker'] = combine_data(dfs['ticker'], h_ticker, subset_columns=['ticker', 'datetime'])

            json_data_macd = client.get_macd(symbol, month)
            if json_data_macd:
                h_macd = Transformer.transform_macd(symbol, json_data_macd)
                dfs['macd'] = combine_data(dfs['macd'], h_macd, subset_columns=['ticker', 'datetime'])

            for period in periods['sma']:
                json_data_sma = client.get_sma(symbol, month, period)
                if json_data_sma:
                    h_sma = Transformer.transform_sma(symbol, json_data_sma, period)
                    dfs['sma'] = combine_data(dfs['sma'], h_sma, subset_columns=['ticker', 'datetime', 'period'])

            for period in periods['rsi']:
                json_data_rsi = client.get_rsi(symbol, month, period)
                if json_data_rsi:
                    h_rsi = Transformer.transform_rsi(symbol, json_data_rsi, period)
                    dfs['rsi'] = combine_data(dfs['rsi'], h_rsi, subset_columns=['ticker', 'datetime', 'period'])

    return dfs

def load_economics(dfs, client, indicators):
    """
    Load economic data for specified indicators, transform it, and store it as CSV files.

    This function retrieves economic data for the provided list of indicators from a given
    client. It processes the data by transforming it into a structured format, saves the
    processed data as separate CSV files for each indicator, and updates the provided
    dictionary with the transformed DataFrame objects for further use.

    Args:
        dfs (dict): A dictionary to store DataFrame objects corresponding to each indicator.
        client: The data source client used to fetch economic indicator data.
        indicators (list): A list of economic indicators to retrieve.

    Returns:
        dict: An updated dictionary containing transformed DataFrame objects for each
              indicator.
    """
    for indicator in indicators:
        print(f'{get_time_now()} :: Getting data: {{\'function\': {indicator}}}')
        json_data = client.get_economic_indicator(indicator)
        if json_data:
            df_economic = Transformer.transform_economic_data(json_data)
            ut.write_csv(df_economic, f'data/df_{indicator}.csv')
            dfs[indicator] = df_economic
            print(f'{get_time_now()} :: Getting data: {{\'function\': {indicator}}} saved successfully!')
        else:
            print(f'{get_time_now()} :: Getting data: Error fetching data for {indicator}')
    return dfs

def load_news(dfs, client, months, topics):
    """
    Loads and processes news sentiment data for specified topics and time ranges, updating the provided dataframe. This
    function iterates over months and topics, retrieves news sentiment data using the provided client, and processes it
    using a transformer. It then combines the processed data with existing news data in the dataframe.

    Args:
        dfs (dict of pandas.DataFrame): A dictionary containing dataframes to be updated. The key 'news' within
                                      the dictionary is used to store combined news data after retrieval and processing.
        client: An instance of a client class capable of retrieving news sentiment data. Must have a method
                `get_news_sentiment` that accepts arguments for topic, start time, and end time.
        months (list of str): A list containing month identifiers for which news data should be retrieved. Each
                              month represents a range of time.
        topics (list of str): A list of topics for which news sentiment data is required.

    Returns:
        dict of pandas.DataFrame: The modified dictionary containing dataframes with updated news data, including
                                  combined and processed news information under the 'news' key.

    Raises:
        This function does not explicitly raise exceptions but relies on the behavior of the client and data transformation
        functions it calls within.
    """
    for month in months:
        for topic in topics:
            time_from, time_to = ut.get_time_range(month)
            json_data = client.get_news_sentiment(topic, time_from, time_to)
            if json_data is not None:
                h_news = Transformer.transform_news_data(json_data, topic)
                dfs['news'] = combine_data(dfs['news'], h_news, subset_columns=['title', 'datetime', 'ticker',
                                                                                'affected_topic'])

    return dfs

def transform_indicators(dfs, period, tech_indicator):
    """
    Transforms technical indicators for a specified period.

    This function isolates and renames the specified technical indicator for
    a given period from the input data, creating a transformed DataFrame
    that includes only the ticker, datetime, and renamed indicator column.

    Args:
        dfs: DataFrame
            The input DataFrame containing technical indicator data. It must
            include the specified technical indicator, period column, ticker,
            and datetime columns.
        period: int
            The period to filter the technical indicator on. Rows matching this
            period in the DataFrame will be extracted.
        tech_indicator: str
            The name of the technical indicator to transform. This column's
            values will be renamed based on the period.

    Returns:
        DataFrame
            A transformed DataFrame with the following columns:
            - ticker
            - datetime
            - technical indicator renamed to reflect the specified period

    """
    df_transformed = dfs[tech_indicator][dfs[tech_indicator]['period'] == period][['ticker', 'datetime', tech_indicator]].copy()
    df_transformed.rename(columns={tech_indicator: f'{tech_indicator}_{period}'}, inplace=True)
    return df_transformed

def merge_datasets(dfs, periods, tec_columns, economic_columns):
    """
    Merges multiple datasets into a single dataset by combining various technical and economic indicator data.

    This function transforms selected indicators for different periods, merges datasets based on technical indicators
    and their columns, and integrates economic indicators into a comprehensive dataset for analysis. The final dataset
    is prepared for further processing or analysis.

    Parameters:
        dfs (dict): A dictionary containing datasets with keys representing dataset names and values as DataFrames.
        periods (dict): A dictionary mapping indicator names to a list of time periods for transformation.
        tec_columns (dict): A dictionary mapping technical indicator dataframe names to their relevant column names
            for merging.
        economic_columns (dict): A dictionary mapping economic indicator names to the desired column names in the
            merged dataset.

    Returns:
        dict: The modified dictionary of DataFrames containing the merged dataset under the key `'merged_tec_info'`.
    """
    # Step 1: Transform RSI and SMA
    for key in periods.keys():
        for period in periods[key]:
            dfs[f'{key}_{period}'] = transform_indicators(dfs, period, key)

    # Step 2: Merge technical indicator datasets with the ticker dataset
    dfs['merged_tec_info'] = dfs['ticker']
    for key, cols in tec_columns.items():
        dfs['merged_tec_info'] = pd.merge(dfs['merged_tec_info'], dfs[key][cols], on=['ticker', 'datetime'], how='left')

    # Merge economic indicators
    for key, col_name in economic_columns.items():
        dfs['merged_tec_info'] = pd.merge(
            dfs['merged_tec_info'],
            dfs[key][['year_month', 'value']].rename(columns={'value': col_name}),
            on='year_month',
            how='left'
        )

    return dfs

def retrieve_data(dfs):
    """
    Retrieve and preprocess data for a given dictionary of dataframes.

    This function reads CSV files for each key in the input dictionary,
    assigns the CSV data to the respective key, and converts specific
    date-related columns to datetime format.

    Arguments:
        dfs (dict): A dictionary where keys are strings that represent
        identifiers and values are placeholder dataframes that serve as
        references for preprocessing steps.

    Returns:
        dict: A dictionary with updated dataframes containing preprocessed
        data, where date-related columns are cast to datetime and non-date
        values are coerced to NaT.
    """
    for key in dfs.keys():
        dfs[key] = ut.read_csv(f'data/df_{key}.csv')

        # Identify date-related columns and convert them to datetime
        for col in ['datetime', 'date', 'year_month']:
            if col in dfs[key].columns:
                dfs[key][col] = pd.to_datetime(dfs[key][col], errors='coerce')

    return dfs

def save_dataframes(dfs):
    """
    Saves multiple dataframes as CSV files.

    This function iterates over a dictionary where keys represent names or identifiers
    for dataframes and values are the actual dataframes. Each dataframe is exported
    and saved to a CSV file in the 'data/' directory with a filename that includes
    its corresponding dictionary key.

    Args:
        dfs (Dict[str, DataFrame]): A dictionary where keys are strings representing
                                    dataframe names or identifiers, and values are
                                    the dataframes to be saved.

    Returns:
        None
    """
    all_keys = dfs.keys()
    for df in all_keys:
        ut.write_csv(dfs[df], f'data/df_{df}.csv')
