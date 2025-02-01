import os
import json
import pandas as pd
from datetime import datetime, timedelta

def load_config(config_file):
    """
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Carpeta raíz (AQF)
    config_path = os.path.join(base_dir, 'config', f'{config_file}.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    return config


def generate_month_list(start_year, end_year=None, frequency='monthly'):
    """
    """
    current_year = datetime.now().year
    current_month = datetime.now().month

    if end_year is None:
        end_year = current_year

    month_list = []

    for year in range(start_year, end_year + 1):
        end_month = 12 if year < current_year else current_month

        if frequency == 'monthly':
            for month in range(1, end_month + 1):
                month_str = f'{year}-{month:02d}'
                month_list.append(month_str)
        elif frequency == 'quarterly':
            for quarter in range(1, 5):  # Cuatro trimestres por año
                month_str = f'{year}-Q{quarter}'
                month_list.append(month_str)

    return month_list


def generate_current_month():
    """
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    return f'{current_year}-{current_month}'


def read_csv(file_path):
    """
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()


def write_csv(df, file_path):
    """
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Crear el directorio si no existe
    df.to_csv(file_path, encoding='utf-8', index=False)


def dataframes_creator(names):
    """
    """
    return {name: pd.DataFrame() for name in names}


def get_time_range(year_month):
    """
    """
    year = int(year_month[:4])
    month = int(year_month[5:])

    # Primer día del mes
    time_from = datetime(year, month, 1).strftime('%Y%m%dT%H%M')

    # Último día del mes
    if month == 12:  # Si es diciembre, ir al enero del año siguiente
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:  # En otros casos, ir al primer día del mes siguiente y restar un día
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    time_to = last_day.strftime('%Y%m%dT%H%M')

    return time_from, time_to


def get_months(year, historical_needed):
    """
    """
    if historical_needed:
        months = generate_month_list(year)
    else:
        months = [generate_current_month()]

    return months

def get_time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")