import pandas as pd

import utils.utils as ut
import loader.DataTransformer as Transformer


def combine_dataframes(df_historical, df_current, f_dataframes, combine_configuration):
    """
    This function combines historical and current economic data stored in dataframes based on a
    specified configuration that determines unique columns for deduplication. The combined data
    is updated in the provided final dataframes dictionary and returned to the caller.

    Parameters
    ----------
    df_historical : dict
        Dictionary containing historical dataframes with keys representing data categories.
    df_current : dict
        Dictionary containing current dataframes with keys representing data categories.
    f_dataframes : dict
        Dictionary used to store the final combined dataframes. Initially used as an input and
        updated with the merged results.
    combine_configuration : dict
        Configuration mapping data category keys to lists, indicating column names to be used
        for removing duplicates while combining historical and current data.

    Returns
    -------
    dict
        Final dictionary with combined and updated dataframes for all keys, sorted by the
        'datetime' column where applicable.

    Raises
    ------
    KeyError
        If a required key for combining data is missing in either the historical or current
        dataframe dictionaries.
    """
    all_keys = list(df_current.keys())

    for enconomic in ['unemployment','nonfarm_payroll','cpi']:
       f_dataframes[enconomic] = df_current[enconomic]
       all_keys.remove(enconomic)

    for key in all_keys:
       df_combined = pd.concat([df_historical[key], df_current[key]]).drop_duplicates(subset=combine_configuration[key], keep='last')
       f_dataframes[key] = df_combined.sort_values(by='datetime', ascending=True)

    return f_dataframes


def combine_data(df_historical,df_current,subset_columns):
    """
    Combines and deduplicates historical and current data based on specified subset columns.

    This function takes two dataframes, one containing historical data and the other containing
    current data, combines them, and removes duplicates based on the specified subset of columns.
    In case of duplicates, the most recent data (from `df_current`) is retained.

    Parameters:
    df_historical : pd.DataFrame
        The dataframe containing historical data.
    df_current : pd.DataFrame
        The dataframe containing current data.
    subset_columns : list
        A list of column names to consider for identifying duplicates.

    Returns:
    pd.DataFrame
        A combined dataframe with duplicates removed based on the subset columns.
    """
    df_combined = pd.concat([df_historical, df_current]).drop_duplicates(subset=subset_columns, keep='last')
    return df_combined


def load_data(dfs,client, symbols, months, periods):
    """
    Loads and processes financial data for specified symbols, months, and periods using client-provided
    functions for different types of indicators (ticker data, MACD, SMA, RSI). The loaded data is
    transformed and combined into the appropriate DataFrame within the provided dictionary. The aim
    is to retrieve and organize intraday data and various indicators systematically.

    Parameters:
        dfs (dict): A dictionary to store and combine financial data. Expects keys such as 'ticker',
                    'macd', 'sma', and 'rsi'.
        client: An object that provides methods for retrieving financial data for symbols, including
                get_intraday_data, get_macd, get_sma, and get_rsi.
        symbols (list[str]): A list of symbol strings representing the financial instruments to be
                             processed.
        months (list[str]): A list of strings representing months for which data needs to be retrieved.
        periods (dict): A dictionary of periods for indicators such as 'sma' and 'rsi' where each key
                        contains a list of integers representing the time frames.

    Returns:
        dict: The updated dictionary of DataFrames (dfs) with combined and transformed financial data for
              all specified symbols, months, and indicator periods.

    Raises:
        None
    """
    for symbol in symbols:
        for month in months:
            # Obtener y transformar datos de ticker
            json_data_ticker = client.get_intraday_data(symbol, month)
            if json_data_ticker: 
                h_ticker = Transformer.transform_intraday(symbol, json_data_ticker)
                dfs['ticker'] = combine_data(dfs['ticker'], h_ticker, subset_columns=['ticker','datetime'])

            json_data_macd = client.get_macd(symbol, month)
            if json_data_macd:
                h_macd = Transformer.transform_macd(symbol, json_data_macd)
                dfs['macd'] = combine_data( dfs['macd'], h_macd, subset_columns=['ticker','datetime'])
            
            for period in periods['sma']:
                json_data_sma = client.get_sma(symbol, month, period)
                if json_data_sma:
                    h_sma = Transformer.transform_sma(symbol, json_data_sma, period)
                    dfs['sma'] = combine_data(dfs['sma'], h_sma, subset_columns=['ticker','datetime','period'])

            for period in periods['rsi']:
                json_data_rsi = client.get_rsi(symbol, month, period)
                if json_data_rsi:
                    h_rsi = Transformer.transform_rsi(symbol, json_data_rsi, period)
                    dfs['rsi'] = combine_data(dfs['rsi'], h_rsi, subset_columns=['ticker','datetime','period'])

    return dfs


def load_economics(dfs, client, indicators):
    """
    Loads economic data for the specified indicators using the provided client.

    The function retrieves economic data for each indicator from the specified
    client, transforms it into data frames, saves them as CSV files, and stores
    them into the given dictionary. If the data for an indicator cannot be
    fetched, an error message will be logged.

    Args:
        dfs (dict): A dictionary to store the transformed data frames. Keys are
        indicators, and values are data frames obtained after processing.
        client: An instance of a client used to fetch economic indicator data.
        The client is expected to have a method `get_economic_indicator`.
        indicators (list): A list of economic indicator names for which data
        will be fetched and processed.

    Returns:
        dict: The updated dictionary after adding the processed data frames for
        the specified indicators.
    """
    for indicator in indicators:
        print(f'Getting data: {{\'function\': {indicator}}}')
        json_data = client.get_economic_indicator(indicator)
        if json_data:
            df_economic = Transformer.transform_economic_data(json_data)
            ut.write_csv(df_economic, f'data/df_{indicator}.csv')
            dfs[indicator] = df_economic
            print(f'Getting data: {{\'function\': {indicator}}} saved successfully!')
        else:
            print(f'Error fetching data for {indicator}')
    return dfs 


def load_news(dfs,client, months, topics):
    """
    Loads news data and sentiments for specified topics within given time ranges. The function processes
    the fetched news sentiment data, applies necessary transformations, and combines it with an existing
    dataframe. It iterates over multiple months and topics to retrieve and merge the relevant information.

    Args:
        dfs (dict): A dictionary where key 'news' references a dataframe storing news data.
        client: The client or API interface used to fetch news sentiment data.
        months (list): A list of months used to define time ranges for which news data is fetched.
        topics (list): A list of topics for which news sentiments are retrieved.

    Returns:
        dict: The input dictionary `dfs` with updated and merged news data under the 'news' key.
    """
    for month in months:
        for topic in topics:
            time_from, time_to = ut.get_time_range(month)
            json_data = client.get_news_sentiment(topic,time_from, time_to)
            if json_data is not None:
                h_news = Transformer.transform_news_data(json_data, topic)
                dfs['news'] = combine_data(dfs['news'], h_news, subset_columns=['title','datetime','ticker','affected_topic'])
    
    return dfs
    

def transform_indicators(dfs, period, tech_indicator):
    """
    Transform and filter technical indicator data for a specific period.

    This function processes a given dictionary of DataFrames containing technical
    indicators. It filters the DataFrame associated with the specified technical
    indicator to return only the entries for the given period. The column representing
    the technical indicator is renamed to include the period as a suffix for clarity.

    Parameters:
        dfs (dict[str, pandas.DataFrame]): A dictionary where the keys are technical
            indicator names and the values are corresponding DataFrames containing
            the data.
        period (int): The specific period for which data should be filtered and
            transformed.
        tech_indicator (str): The name of the technical indicator to process, which
            corresponds to a key in the `dfs` dictionary.

    Returns:
        pandas.DataFrame: A transformed DataFrame containing the filtered data for
        the specified technical indicator and period, with the column renamed to
        include the period as a suffix.
    """
    df_transformed =  dfs[tech_indicator][dfs[tech_indicator]['period'] == period][['ticker', 'datetime', tech_indicator]].copy()
    df_transformed.rename(columns={tech_indicator: f'{tech_indicator}_{period}'},inplace = True)
    return df_transformed


def merge_datasets(dfs, periods, tec_columns, economic_columns):
    """
    Merges multiple datasets of technical and economic indicators with a financial dataset.

    This function transforms technical indicators using specified periods, merges the transformed
    data with a dataset containing financial information, and integrates economic indicator
    data into the resulting dataset. The output dataset provides a consolidated view of both
    technical and economic indicators along with the original financial dataset.

    Parameters:
        dfs (dict): A dictionary of DataFrames, where each DataFrame corresponds to a dataset
            such as financial data, technical indicators, and economic indicators. Each DataFrame
            is indexed by a key representing its type (e.g., 'ticker', 'technical', etc.).
        periods (dict): A dictionary where keys represent technical indicator types and values are
            lists of time periods over which to transform the indicators.
        tec_columns (dict): A dictionary mapping technical indicator types to lists of column names
            to include in the merged dataset.
        economic_columns (dict): A dictionary mapping economic dataset keys to new column names
            for their respective indicator values in the merged dataset.

    Returns:
        dict: A dictionary of DataFrames with the updated 'merged_tec_info' key containing the
            merged dataset of financial data, technical indicators, and economic data.
    """
    # Paso 1: Transformar RSI y SMA
    for key in periods.keys():
        for period in periods[key]:
            dfs[f'{key}_{period}'] = transform_indicators(dfs, period, key)

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
    """
    Retrieve and process data from multiple CSV files into a dictionary of DataFrames.

    This function processes a set of CSV files contained in a dictionary of file
    paths. For each key in the dictionary, it reads the corresponding CSV file,
    identifies specific columns, and converts columns related to dates into
    datetime objects to ensure proper data manipulation and consistency.

    Parameters:
        dfs (dict): A dictionary where the keys represent identifiers for the data,
        and the values are pandas DataFrame objects, which will be updated with the
        processed data. Each CSV file is expected to use a naming convention like
        'data/df_<key>.csv'.

    Returns:
        dict: The input dictionary with updated DataFrames, where date-related
        columns have been converted to datetime objects, and the original data
        has been replaced with the new processed DataFrames.
    """
    for key in dfs.keys():
        dfs[key] = ut.read_csv(f'data/df_{key}.csv')

        # Identificar columnas relacionadas con fechas y convertirlas a datetime
        for col in ['datetime', 'date', 'year_month']:
            if col in dfs[key].columns:
                dfs[key][col] = pd.to_datetime(dfs[key][col], errors='coerce')

    return dfs


def save_dataframes(dfs):
    """
    Function that saves multiple pandas dataframes to CSV files. Each dataframe is written
    to a separate file, with filenames derived from the keys of the input dictionary.

    Parameters:
    dfs : dict
        A dictionary where keys are strings representing the names of the dataframes,
        and values are pandas.DataFrame objects containing the data to be saved.

    Returns:
    None
    """
    all_keys = dfs.keys()
    for df in all_keys:
        ut.write_csv(dfs[df], f'data/df_{df}.csv')