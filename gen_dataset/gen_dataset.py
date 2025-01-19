import os
import sys
from gen_dataset.check_dataset import CheckDataset


def check_tec_dataset(df, checker):
    # Corregir valores faltantes en el dataset merged_tec_info
    df_tec = checker.calculate_missing_indicators(df)
    df_tec = checker.apply_date_time_actions(df_tec)
    df_tec = checker.apply_corrections(df_tec)
    df_tec = checker.apply_advanced_features(df_tec)

    return df_tec


def run_gen_dataset(dfs):
    """
    Genera el dataset final combinando las correcciones necesarias.
    """
    # Inicializar el módulo de corrección
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    checker = CheckDataset()
    df_tec = check_tec_dataset(dfs['tec_info'],checker)

    return df_tec




