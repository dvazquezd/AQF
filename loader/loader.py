import os
import sys
import pandas as pd
import utils.utils as ut
import loader.DataLoader as DataLoader
from loader.ApiClient import ApiClient



def run_loader():
    """
    """
    # Objetos
    client = ApiClient()
    config = ut.load_config('loader_config')

    # Crear dataframes vacíos
    dataframes = {key: pd.DataFrame() for key in config['dataframes']}
    h_dataframes = dataframes.copy()
    f_dataframes = dataframes.copy()

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    months = ut.get_months(config['historical_year'], config['historical_needed'])
    dataframes = DataLoader.load_data(dataframes, client, config['symbols'], months, config['periods'])
    dataframes = DataLoader.load_economics(dataframes, client, config['economic_indicators'])
    dataframes = DataLoader.load_news(dataframes, client, months, config['topics'])
    dataframes = DataLoader.merge_datasets(dataframes, config['periods'], config['tec_columns'], config['economic_columns'])

    if config['historical_needed']:
        DataLoader.save_dataframes(dataframes)
        return {'tec_info': dataframes['merged_tec_info'], 'news': dataframes['news']}        
    else:
        h_dataframes = DataLoader.retrieve_data(h_dataframes)
        f_dataframes = DataLoader.combine_dataframes(h_dataframes, dataframes, f_dataframes, config['combine_configuration'])
        DataLoader.save_dataframes(f_dataframes)
        return {'tec_info': f_dataframes['merged_tec_info'], 'news': f_dataframes['news']}
