import os
import sys
from gen_dataset.CheckNewsDataset import CheckNewsDataset
from gen_dataset.CheckTecDataset import CheckTecDataset


def check_news_dataset(df, checker, target_ticker):
    """
    Checks and processes a news dataset for a specific target ticker.

    This function filters the given dataset for a specified target ticker using the
    provided checker object. After filtering, it generates relevant features for
    the data entries associated with the target ticker.

    Parameters:
    df: DataFrame
        The input dataset containing news data.

    checker: Any
        An object that provides methods for filtering the dataset by a ticker
        and generating features.

    target_ticker: str
        The ticker symbol used to filter the news dataset.

    Returns:
    DataFrame
        A filtered and processed dataset containing news entries for the
        target ticker with additional generated features.
    """
    df_news = checker.filter_by_ticker(df, target_ticker)

    return df_news

def check_tec_dataset(df, checker):
    """
        Performs a series of operations to correct and enhance the provided dataset
        using the specified checker. This includes calculating missing indicators,
        applying date-time adjustments, performing corrections, and adding advanced
        features.

        Parameters:
        df : DataFrame
            The input dataset that needs to be corrected and enhanced.
        checker : object
            An object containing methods for handling various correction and
            enhancement operations.

        Returns:
        DataFrame
            The corrected and enhanced dataset after all transformations.
    """
    # Corregir valores faltantes en el dataset merged_tec_info
    df_tec = checker.calculate_missing_indicators(df)
    df_tec = checker.apply_date_time_actions(df_tec)
    df_tec = checker.apply_corrections(df_tec)
    df_tec = checker.apply_advanced_features(df_tec)
    target_ticker = checker.get_target_ticker(df_tec)

    return df_tec, target_ticker


def run_gen_dataset(dfs):
    """
        Processes and validates a dataset by initializing a correction module and
        invoking a dataset checker on a specific key of the provided data.

        Parameters:
        dfs: dict
            A dictionary containing datasets, including a key 'tec_info'
            associated with relevant data for technical validation.

        Returns:
        pandas.DataFrame
            A DataFrame containing the validated and possibly corrected
            version of the 'tec_info' dataset.
    """
    # Inicializar el módulo de corrección
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    tec_checker = CheckTecDataset()
    news_checker = CheckNewsDataset()

    df_tec, target_ticker = check_tec_dataset(dfs['tec_info'],tec_checker)
    df_news = check_news_dataset(dfs['news'],news_checker, target_ticker)

    return df_tec, df_news