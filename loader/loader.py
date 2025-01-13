import os
import sys
import pandas as pd
import loader.BricMortar as bm
import loader.DataLoader as dl
import utils.utils as ut
from utils.config_manager import ConfigManager
from loader.ApiClient import ApiClient



def run_loader():
    # Objetos
    client = ApiClient()
    config = ConfigManager('loader_config')

    # Crear dataframes vacíos
    dataframes = {key: pd.DataFrame() for key in config.get_section('dataframes')}
    h_dataframes = dataframes.copy()
    f_dataframes = dataframes.copy()

    months = bm.get_months(2022, config['historical_needed'])
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


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    run_loader()