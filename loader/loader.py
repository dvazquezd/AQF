import os
import sys
import pandas as pd
import utils.utils as ut
import loader.DataLoader as DataLoader
from loader.ApiClient import ApiClient

def run_loader():
    """
    Executes the data loading and preprocessing workflow based on configuration.

    The function is designed to load, process, and save financial and economic
    data based on the parameters specified in the configuration file. It
    initializes data structures, manages historical and current data loading
    conditions, and processes datasets using specified operations. The function
    determines whether to charge new values, retrieve historical data, or
    combine datasets based on config settings.

    Parameters:
    None

    Returns:
    dict: A dictionary containing processed technical information ('tec_info')
          and news data ('news') in respective keys.

    Raises:
    KeyError: Raised if a required key is missing from the configuration.
    ValueError: Raised if the configuration values are invalid or do not match
                expected formats.
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
        dataframes = DataLoader.load_data(dataframes, client, config['symbols'], months, config['periods'])
        dataframes = DataLoader.load_economics(dataframes, client, config['economic_indicators'])
        dataframes = DataLoader.load_news(dataframes, client, months, config['topics'])
        dataframes = DataLoader.merge_datasets(dataframes, config['periods'], config['tec_columns'], config['economic_columns'])

        if config['historical_needed']:
            DataLoader.save_dataframes(dataframes)
            return {'tec_info': dataframes['merged_tec_info'], 'news': dataframes['news']}

        h_dataframes = DataLoader.retrieve_data(h_dataframes)
        f_dataframes = DataLoader.combine_dataframes(h_dataframes, dataframes, f_dataframes, config['combine_configuration'])
        DataLoader.save_dataframes(f_dataframes)
        return {'tec_info': f_dataframes['merged_tec_info'], 'news': f_dataframes['news']}

    h_dataframes = DataLoader.retrieve_data(h_dataframes)
    return {'tec_info': h_dataframes['merged_tec_info'], 'news': h_dataframes['news']}