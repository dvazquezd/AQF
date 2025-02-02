import os
import sys
from utils.utils import get_time_now
from gen_dataset.CheckNewsDataset import CheckNewsDataset
from gen_dataset.CheckTecDataset import CheckTecDataset
from gen_dataset.DatasetGenerator import DatasetGenerator
from gen_dataset.FeatureEngineering import FeatureEngineering

def feature_engineering(fe):
    fe.add_lags()
    fe.add_advanced_features()
    fe.add_moving_avg()
    fe.delete_no_necessary_col()
    fe.final_ds_checks()

    return fe

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
    checker.generate_news_global_metrics()

    return checker

def check_tec_dataset(checker):
    """
    """
    # Corregir valores faltantes en el dataset merged_tec_info
    checker.apply_economic_indicators()
    checker.calculate_missing_indicators()
    checker.apply_date_time_actions()
    checker.apply_corrections()
    checker.get_target_ticker()

    return checker


def run_gen_dataset(dfs):
    """
    """
    print(f'{get_time_now()} :: Dataset Generation: Starting dataset generation')
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    tec_checker = CheckTecDataset(dfs['tec_info'])
    news_checker = CheckNewsDataset(dfs['news'],tec_checker.target_ticker)

    tec = check_tec_dataset(tec_checker)
    news = check_news_dataset(news_checker)

    ds_generator = DatasetGenerator(news.df, tec.df)
    ds = generate_dataset(ds_generator)

    feature = FeatureEngineering(ds.df)
    ds = feature_engineering (feature)

    return ds.df