import os
import sys
from utils.utils import get_time_now
from gen_dataset.CheckNewsDataset import CheckNewsDataset
from gen_dataset.CheckTecDataset import CheckTecDataset
from gen_dataset.DatasetGenerator import DatasetGenerator
from gen_dataset.FeatureEngineering import FeatureEngineering


def feature_engineering(fe):
    """
    Feature engineering workflow for processing and enhancing dataset features.

    This function performs a sequence of transformations on the provided dataset
    by utilizing the input object's feature engineering methods. It includes
    operations such as generating lag features, adding advanced features,
    calculating moving averages, removing unnecessary columns, and performing
    final dataset integrity checks.

    Parameters:
    fe : object
        An object that contains methods for performing feature engineering, such as
        adding lags, advanced features, moving averages, deleting unnecessary
        columns, and final checks.

    Returns:
    object
        The modified input object after all feature engineering transformations
        have been applied.
    """
    fe.add_lags()
    fe.add_advanced_features()
    fe.add_moving_avg()
    fe.delete_no_necessary_col()

    return fe

def generate_dataset(gen):
    """
        Generates a complete and aggregated dataset by utilizing the provided generator.

        This function performs a sequence of operations to process and combine data.
        It completes any missing timestamps in the data, aggregates data over previous
        time intervals, and merges datasets for comprehensive analysis.

        Args:
            gen (Generator): An instance of a data generator class that provides
                             methods to process, aggregate, and merge datasets.

        Returns:
            Generator: The modified generator instance with updated and
                       fully prepared data.
    """
    gen.complete_missing_times()
    gen.aggregate_previous_hours()
    gen.merge_datasets()

    return gen

def check_news_dataset(checker):
    """
    Processes news dataset using predefined feature generation methods on the provided
    checker object.

    This function orchestrates the execution of various feature generation functionalities
    associated with the checker object. It ensures that ticker features, topic features,
    and global news metrics are appropriately generated and updated within the checker.

    Parameters:
    checker : object
        An object that contains methods to generate ticker features, topic features,
        and global news metrics. The specific methods required for execution must be
        implemented within this object.

    Returns:
    object
        The updated checker object after generating ticker features, topic features,
        and global news metrics.
    """
    checker.generate_ticker_features()
    checker.generate_topic_features()
    checker.generate_news_global_metrics()

    return checker

def check_tec_dataset(checker):
    """
    Performs a series of checks and operations on a given dataset by applying various
    economic indicators, calculating missing data, applying date and time adjustments,
    and performing corrections, before obtaining the target ticker.

    Args:
        checker: An instance of a class responsible for managing and validating the
            dataset. This instance contains methods to manipulate the dataset
            according to predefined logic.

    Returns:
        The modified checker instance after performing all operations and
        modifications on the dataset.
    """
    checker.apply_economic_indicators()
    checker.calculate_missing_indicators()
    checker.apply_date_time_actions()
    checker.apply_corrections()
    checker.get_target_ticker()

    return checker


def run_gen_dataset(dfs):
    """
    Generates and processes a dataset using input dataframes for technical and news datasets.

    This function orchestrates the process of validating and generating datasets from
    technical and news-related datasets. It makes use of utility functions and classes to
    perform the checks, merging, and feature engineering steps required to create the
    final dataset for further use.

    Args:
        dfs (Dict[str, DataFrame]): A dictionary holding the input dataframes.
            Keys:
            - 'tec_info': DataFrame containing technical data information.
            - 'news': DataFrame containing news-related data.

    Returns:
        DataFrame: The final processed dataset after merging, validation, and
        feature engineering steps.
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

    print(f'{get_time_now()} :: Dataset Generation: Dataset generation ended')

    return ds.df