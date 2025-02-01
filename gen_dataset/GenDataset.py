import os
import sys

import utils.utils
from utils.utils import get_time_now
from gen_dataset.CheckNewsDataset import CheckNewsDataset
from gen_dataset.CheckTecDataset import CheckTecDataset
from gen_dataset.DatasetGenerator import DatasetGenerator

def generate_dataset(gen):
    """
    """
    gen.complete_missing_times()
    gen.aggregate_previous_hours()
    gen.merge_datasets()

    return gen

def check_news_dataset(checker):
    """
    """
    checker.generate_ticker_features()
    checker.generate_topic_features()

    return checker

def check_tec_dataset(checker):
    """
    """
    # Corregir valores faltantes en el dataset merged_tec_info
    checker.apply_economic_indicators()
    checker.calculate_missing_indicators()
    checker.apply_date_time_actions()
    checker.apply_corrections()
    checker.apply_advanced_features()
    checker.get_target_ticker()

    return checker


def run_gen_dataset(dfs):
    """
    """
    print(f'{utils.utils.get_time_now()} :: Dataset Generation: Starting dataset generation')
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    tec_checker = CheckTecDataset(dfs['tec_info'])
    news_checker = CheckNewsDataset(dfs['news'],tec_checker.target_ticker)

    tec = check_tec_dataset(tec_checker)
    news = check_news_dataset(news_checker)

    ds_generator = DatasetGenerator(news_checker.df, tec_checker.df)
    ds = generate_dataset(ds_generator)

    return ds.df