import os
import sys
from gen_dataset.CheckNewsDataset import CheckNewsDataset
from gen_dataset.CheckTecDataset import CheckTecDataset


def check_news_dataset(checker):
    """
    """
    checker.generate_ticker_features()

    return checker

def check_tec_dataset(checker):
    """
    """
    # Corregir valores faltantes en el dataset merged_tec_info
    checker.calculate_missing_indicators()
    checker.apply_date_time_actions()
    checker.apply_corrections()
    checker.apply_advanced_features()
    checker.get_target_ticker()

    return checker


def run_gen_dataset(dfs):
    """
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    tec_checker = CheckTecDataset(dfs['tec_info'])
    news_checker = CheckNewsDataset(dfs['news'],tec_checker.target_ticker)

    tec_checker = check_tec_dataset(tec_checker)
    news_checker = check_news_dataset(news_checker)

    return tec_checker.df , news_checker.df