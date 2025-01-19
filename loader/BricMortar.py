import json
import os
from datetime import datetime, timedelta

import pandas as pd


def generate_month_list(start_year, end_year=None, frequency='monthly'):
    """
    Genera una lista de meses desde start_year hasta el año actual o el año especificado.
    frequency: 'monthly' (por defecto) o 'quarterly'
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
    current_year = datetime.now().year
    current_month = datetime.now().month
    return f'{current_year}-{current_month}'


def save_json(data, file_name, indent_num=4):
    """Guarda los datos en formato JSON."""
    with open(file_name, 'w') as hist_file:
        json.dump(data, hist_file, indent=indent_num)


def read_csv(file_path):
    """Leer un archivo CSV si existe, si no, retorna un DataFrame vacío."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()


def write_csv(df, file_path):
    """Sobrescribe un archivo CSV con el nuevo DataFrame."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Crear el directorio si no existe
    df.to_csv(file_path, index=False)


def dataframes_creator(names):
    return {name: pd.DataFrame() for name in names}


def get_time_range(year_month):
    # Convertir la cadena de entrada a un objeto datetime para el primer día del mes
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


def get_months(year,historical_needed):
    if historical_needed:
        months = generate_month_list(year)
    else:
        months = [generate_current_month()]

    return months