import os
import sys
from gen_dataset.CheckTecDataset import CheckTecDataset


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

    return df_tec


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
    checker = CheckTecDataset()
    df_tec = check_tec_dataset(dfs['tec_info'],checker)

    return df_tec