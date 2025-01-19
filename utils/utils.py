import os
import json
import pandas as pd
from datetime import datetime, timedelta

def load_config(config_file):
    """
        Load the configuration from the specified JSON configuration file.

        This function reads the contents of a JSON file located in the 'config'
        directory within the base directory of the project. It decodes the JSON
        content into a Python dictionary and returns it. The base directory is
        determined dynamically and assumed to be the directory containing the
        parent of the current script's location.

        Arguments:
            config_file (str): The name of the configuration file (without the
                '.json' suffix).

        Returns:
            dict: A dictionary representing the configuration data read from the
            JSON file.

        Raises:
            FileNotFoundError: If the configuration file does not exist at the
                expected location.
            json.JSONDecodeError: If the content of the configuration file is
                invalid JSON.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Carpeta raíz (AQF)
    CONFIG_PATH = os.path.join(BASE_DIR, 'config', f'{config_file}.json')
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)

    return config


def generate_month_list(start_year, end_year=None, frequency='monthly'):
    """
        Generates a list of months or quarters between the specified years.

        This function creates a list of months or quarters represented as strings,
        starting from the `start_year` and potentially extending to the `end_year`.
        If `end_year` is not provided, it defaults to the current year. The frequency
        of the list can be specified as either 'monthly' or 'quarterly'. When 'monthly',
        it includes individual months, while 'quarterly' includes quarters of each year.
        The range automatically adapts based on the current month for the current year.

        Args:
            start_year (int): The starting year of the range.
            end_year (Optional[int]): The ending year of the range. Defaults to the current year if not provided.
            frequency (str): The frequency of the list, either 'monthly' or 'quarterly'. Defaults to 'monthly'.

        Returns:
            List[str]: A list of formatted strings representing months (e.g., 'YYYY-MM')
            or quarters (e.g., 'YYYY-QN') depending on the specified frequency.

        Raises:
            None
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
        Generate the current month and year in 'YYYY-MM' format.

        This function retrieves the current date using the system's
        current time and formats it to return the year and month
        as a string in 'YYYY-MM' format.

        Returns:
            str: A string representing the current year and month
                 in the format 'YYYY-MM'.
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    return f'{current_year}-{current_month}'


def save_json(data, file_name, indent_num=4):
    """
        Saves the given data to a JSON file with a specified indentation level.

        This function utilizes the `json` module to serialize Python objects
        and write them to a file in JSON format. It allows specifying the
        number of spaces used for indentation when formatting the JSON file.

        Parameters:
        data: Any
            The Python object to be serialized and written to the JSON file.
        file_name: str
            The name or path of the file where the JSON data will be saved.
        indent_num: int, optional
            The number of spaces to use for indentation in the JSON file.
            Defaults to 4.
    """
    with open(file_name, 'w') as hist_file:
        json.dump(data, hist_file, indent=indent_num)


def read_csv(file_path):
    """
        Reads a CSV file from the specified file path into a Pandas DataFrame. If the file is not
        found, returns an empty DataFrame.

        Arguments:
            file_path (str): The path to the CSV file that needs to be read.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the CSV file if successfully
            read, otherwise an empty DataFrame.

        Raises:
            FileNotFoundError: Raised when the specified file is not found.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()


def write_csv(df, file_path):
    """
        Writes a pandas DataFrame to a CSV file while ensuring that the directory exists.
        If the directory specified in the file path does not exist, it creates the directory
        before saving the file.

        Args:
            df (DataFrame): The pandas DataFrame to be written to the CSV file.
            file_path (str): The full file path, including the directory and filename, where
                the CSV file will be saved.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Crear el directorio si no existe
    df.to_csv(file_path, index=False)


def dataframes_creator(names):
    """
        Generates a dictionary of empty pandas DataFrames with keys derived from provided names.

        This function takes a list of names and creates a dictionary where each name
        in the list becomes a key, and its corresponding value is an empty pandas DataFrame.

        Parameters:
            names (list of str): A list of strings representing the keys for the
            dictionary.

        Returns:
            dict: A dictionary where keys are derived from the input list of names,
            and values are empty pandas DataFrame objects.

        Raises:
            None
    """
    return {name: pd.DataFrame() for name in names}


def get_time_range(year_month):
    """
        Generate a time range for the given year and month.

        This function takes a string in the format 'YYYY-MM', extracts the year and
        month, and calculates the range of time for the entire month. The result
        includes the first and last days of the month, formatted as strings in
        the format 'YYYYMMDDTHHMM'.

        Parameters:
            year_month (str): A string representing the year and month in the format
            'YYYY-MM'.

        Returns:
            tuple[str, str]: A tuple containing two strings: the starting timestamp
            ('YYYYMMDDTHHMM') and the ending timestamp ('YYYYMMDDTHHMM').
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
        Retrieve a list of months based on the provided year and need for historical data.

        This function generates a list of months depending on whether historical data
        is needed or the focus is solely on the current month. The main goal is to
        create a relevant list of months based on the given inputs.

        Args:
            year (int): The year for which the months need to be generated.
            historical_needed (bool): A flag indicating whether historical data
                is required or not.

        Returns:
            List[str]: A list of months either from the whole year or limited to the
            current month based on the historical_needed flag.
    """
    if historical_needed:
        months = generate_month_list(year)
    else:
        months = [generate_current_month()]

    return months