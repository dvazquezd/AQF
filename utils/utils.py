import os
import json

def load_config(config_file):
    '''
    '''
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Carpeta raíz (AQF)
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', f'{config_file}.json')
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)

    return config