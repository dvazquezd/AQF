import os
import sys
import pandas as pd
import utils.utils as ut
from gen_dataset.check_dataset import CheckDataset


def run_gen_dataset(dfs):
    '''    
    Genera el dataset final combinando las correcciones necesarias.
    '''
    #Objects
    config = ut.load_config('gen_dataset_config')

    # Inicializar el módulo de corrección
    checker = CheckDataset(config)

    # Corregir valores faltantes en el dataset merged_tec_info
    df_tec = checker.calculate_missing_indicators(dfs['tec_info'])
    df_tec = checker.apply_date_time_actions(df_tec)
    df_tec = checker.apply_corrections(df_tec)
    df_tec = checker.apply_advanced_features(df_tec)



    # Continuar con el resto de la generación del dataset...
    return df_tec


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    run_gen_dataset()





