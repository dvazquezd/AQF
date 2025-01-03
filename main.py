import pandas as pd
import lib.BricMortar as bm
import lib.DataLoader as dl
from lib.ApiClient import ApiClient


def main():
    '''    
    '''
    #Objects
    client = ApiClient()
    config = dl.load_config()

    print(config)

    # Variables y configuraciones
    historical_needed = config['historical_needed']
    symbols = config['symbols']
    periods = config['periods']
    economic_indicators = config['economic_indicators']
    topics = config['topics']
    dataframes = {key: pd.DataFrame() for key in config['dataframes']}
    h_dataframes = dataframes
    f_dataframes = dataframes    
    tec_columns = config['tec_columns']
    economic_columns = config['economic_columns']
    combine_configuration = config['combine_configuration']

    months = bm.get_months(2022, historical_needed)
    dataframes = dl.load_data(dataframes,client, symbols, months, periods)
    dataframes = dl.load_economics(dataframes, client, economic_indicators)
    dataframes = dl.load_news(dataframes, client, months, topics)
    dataframes = dl.merge_datasets(dataframes, periods, tec_columns, economic_columns)

    if not historical_needed:
        h_dataframes = dl.retrieve_data(h_dataframes)
        f_dataframes = dl.combine_dataframes(dataframes, h_dataframes,f_dataframes, combine_configuration)   

    dl.save_dataframes(f_dataframes)

if __name__ == '__main__':
    main()