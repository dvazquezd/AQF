import os
import sys
import pandas as pd
import utils.utils as ut
import loader.DataLoader as dl
from utils.ConfigManager import ConfigManager
from loader.ApiClient import ApiClient



def run_loader():
    """
        Run the data loader to process and manage dataframes using configurations and loaded data.

        The function initializes and processes dataframes using various data sources and configurations,
        including historical and economic data, news, and specified periods. The function manages two major
        scenarios based on the historical data necessity: either saving fully processed dataframes or combining
        historical and loaded dataframes for a more comprehensive result.

        Returns
        -------
        dict
            A dictionary containing the processed technical information and news dataframes.

        Raises
        ------
        KeyError
            If required configuration keys are missing during the process.
        AttributeError
            If attributes are not found in the passed objects or configurations.
        ImportError
            If modules specified in sys.path are unavailable or not found.
    """
    # Objetos
    client = ApiClient()
    config = ConfigManager('loader_config')

    # Crear dataframes vacíos
    dataframes = {key: pd.DataFrame() for key in config.get_section('dataframes')}
    h_dataframes = dataframes.copy()
    f_dataframes = dataframes.copy()

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    months = ut.get_months(config['historical_year'], config['historical_needed'])
    dataframes = dl.load_data(dataframes,client, config['symbols'], months, config['periods'])
    dataframes = dl.load_economics(dataframes, client, config['economic_indicators'])
    dataframes = dl.load_news(dataframes, client, months, config['topics'])
    dataframes = dl.merge_datasets(dataframes, config['periods'], config['tec_columns'], config['economic_columns'])

    if config['historical_needed']:
        dl.save_dataframes(dataframes)
        return {'tec_info': dataframes['merged_tec_info'], 'news': dataframes['news']}        
    else:
        h_dataframes = dl.retrieve_data(h_dataframes)
        f_dataframes = dl.combine_dataframes(h_dataframes, dataframes,f_dataframes, config.get_section('combine_configuration'))
        dl.save_dataframes(f_dataframes)
        return {'tec_info': f_dataframes['merged_tec_info'], 'news': f_dataframes['news']}