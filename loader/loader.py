import os
import sys
import pandas as pd
import utils.utils as ut
import loader.data_loader as data_loader
from loader.api_client import ApiClient

def run_loader():
    """
    Runs the data loading and merging process based on the configuration provided in 'loader_config'.

    This function initializes the required objects and data structures to handle the data loading operations.
    Depending on the configuration, it can fetch new data, retrieve historical data, or combine existing dataframes,
    followed by saving them for further use. Supports handling technical, economic, and news-related datasets.

    Returns a dictionary containing specific aggregated or merged dataframes depending on the configuration
    process executed.

    Raises:
        Any specific exception encountered during data loading, merging, or saving processes.

    Returns
    -------
    dict
        A dictionary containing results of the data loading or processing operation:
        - 'tec_info': Aggregated or merged technical information dataframe.
        - 'news': Aggregated news-related dataframe.
    """
    # Objects
    client = ApiClient()
    config = ut.load_config('loader_config')

    # Create empty dataframes
    dataframes = {key: pd.DataFrame() for key in config['dataframes']}
    h_dataframes = dataframes.copy()
    f_dataframes = dataframes.copy()

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    if config['charge_new_values']:
        months = ut.get_months(config['historical_year'], config['historical_needed'])
        dataframes = data_loader.load_data(dataframes, client, config['symbols'], months, config['periods'])
        dataframes = data_loader.load_economics(dataframes, client, config['economic_indicators'])
        dataframes = data_loader.load_news(dataframes, client, months, config['topics'])
        dataframes = data_loader.merge_datasets(dataframes, config['periods'], config['tec_columns'], config['economic_columns'])

        if config['historical_needed']:
            data_loader.save_dataframes(dataframes)
            return {'tec_info': dataframes['merged_tec_info'], 'news': dataframes['news']}

        h_dataframes = data_loader.retrieve_data(h_dataframes)
        f_dataframes = data_loader.combine_dataframes(h_dataframes, dataframes, f_dataframes, config['combine_configuration'])
        data_loader.save_dataframes(f_dataframes)
        return {'tec_info': f_dataframes['merged_tec_info'], 'news': f_dataframes['news']}

    h_dataframes = data_loader.retrieve_data(h_dataframes)
    return {'tec_info': h_dataframes['merged_tec_info'], 'news': h_dataframes['news']}