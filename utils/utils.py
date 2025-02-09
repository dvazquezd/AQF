import os
import json
import pandas as pd
from datetime import datetime, timedelta

def load_config(config_file):
    """
    Load configuration from a JSON file located in the config directory.

    This function reads a JSON configuration file from the 'config' directory
    relative to the root folder and returns its content as a dictionary. The
    root folder is determined dynamically using the directory of the current
    file. The configuration file must have a '.json' extension.

    Parameters:
        config_file: str
            The name of the configuration file without the '.json' extension.

    Returns:
        dict
            A dictionary containing the contents of the loaded configuration
            file.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder (AQF)
    config_path = os.path.join(base_dir, 'config', f'{config_file}.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    return config


def generate_month_list(start_year, end_year=None, frequency='monthly'):
    """
    Generates a list of timestamped strings based on the provided start year, optional
    end year, and frequency of either 'monthly' or 'quarterly'.

    This function creates a list of months or quarters between the specified years
    depending on the frequency parameter. The end year defaults to the current year
    if it is not explicitly provided.

    Parameters:
        start_year (int): The starting year for the list generation. Inclusive.
        end_year (Optional[int]): The ending year for the list generation. Defaults
            to the current year if not specified.
        frequency (str): Defines the frequency of the timestamp. Possible values
            are 'monthly' for month-based generation and 'quarterly' for quarter-based
            generation. Defaults to 'monthly'.

    Returns:
        list[str]: A list containing date strings as either months (YYYY-MM) or
        quarters (YYYY-Qx), formatted according to the selected frequency.

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
            for quarter in range(1, 5):  # Four quarters per year
                month_str = f'{year}-Q{quarter}'
                month_list.append(month_str)

    return month_list


def generate_current_month():
    """
    Generates a string representing the current year and month in the format "YYYY-MM".

    This function fetches the current year and month using the system's current
    date and time and formats them into a string with the format "YYYY-MM".

    Returns:
        str: A string representing the current year and month in "YYYY-MM" format.

    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    return f'{current_year}-{current_month}'


def read_csv(file_path):
    """
    Reads a CSV file from the given file path and returns its content as a pandas DataFrame.
    If the file does not exist, it returns an empty DataFrame.

    Args:
        file_path (str): The path to the CSV file to be read.

    Returns:
        pandas.DataFrame: The content of the CSV file, or an empty DataFrame if the file is
        not found.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()


def write_csv(df, file_path):
    """
    Write the contents of a DataFrame to a CSV file, ensuring that the directory
    in which the file resides is created if it doesn't already exist.

    Args:
        df (DataFrame): The pandas DataFrame to be written to a CSV file.
        file_path (str): The path where the CSV file is to be saved. Includes
            the directory and file name.

    Raises:
        OSError: If the directory creation fails due to an OS-related issue.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it does not exist
    df.to_csv(file_path, encoding='utf-8', index=False)


def dataframes_creator(names):
    """
    Generates a dictionary of empty pandas DataFrame objects with specified names as keys.

    This function takes a list of strings representing names and creates a dictionary where
    the keys are the given names and the values are empty pandas DataFrame objects.
    It is useful for initializing multiple DataFrame objects for later use.

    Args:
        names (list[str]): List of strings representing the keys for the dictionary.

    Returns:
        dict[str, pandas.DataFrame]: A dictionary where the keys are strings from the
        provided list, and the values are empty pandas DataFrame objects.
    """
    return {name: pd.DataFrame() for name in names}


def get_time_range(year_month):
    """
        Generate start and end timestamps for a given year and month.

        This function takes a string representing year and month in the format
        "YYYY-MM" and returns the start and end timestamps for that month, formatted
        as strings in 'YYYY-MM-DD T HHMM' format. The start timestamp corresponds to
        the first day of the month at '00:00', while the end timestamp corresponds
        to the last day of the month at '23:59'.

        The function utilizes Python's `datetime` and `timedelta` modules to compute
        the timestamps, accounting for edge cases such as the end of December (which
        moves into the next year).

        Parameters:
        year_month: str
            A string in the format "YYYY-MM" representing the desired year and month.

        Returns:
        tuple[str, str]
            A tuple containing the start timestamp and end timestamp, both formatted
            as strings in 'YYYY-MM-DD T HHMM'.

        Raises:
        ValueError
            If the input string does not conform to the "YYYY-MM" format or is not a
            valid date.

        Example:
        Raises ValueError for invalid formats or inputs like '2023-13', 'abcd-ef', etc.
    """
    year = int(year_month[:4])
    month = int(year_month[5:])

    # First day of the month
    time_from = datetime(year, month, 1).strftime('%Y%m%dT%H%M')

    # Last day of the month
    if month == 12:  # If December, go to January of the next year
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:  # Otherwise, go to the first day of the next month and subtract one day
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    time_to = last_day.strftime('%Y%m%dT%H%M')

    return time_from, time_to


def get_months(year, historical_needed):
    """
        Generates a list of months based on historical requirement and year provided.

        This function determines the list of months to return based on whether historical data
        is required. If historical data is needed, it generates a list of all months for the
        given year. Otherwise, it generates only the current month.

        Parameters:
        year (int): The year for which the months are to be generated if historical data is needed.
        historical_needed (bool): A flag indicating whether to generate all months (if True)
            or just the current month (if False).

        Returns:
        list: A list of months as per the requirement (entire year if historical data is needed,
        or current month only).
    """
    if historical_needed:
        months = generate_month_list(year)
    else:
        months = [generate_current_month()]

    return months


def get_time_now():
    """
    Fetch the current date and time as a formatted string.

    This function retrieves the system's current date and time and formats it
    into a string following the 'YYYY-MM-DD HH:MM:SS' format. The returned string
    can be used for logging, displaying timestamps in a user interface, or other
    purposes requiring a human-readable date and time representation.

    Returns:
        str: The current date and time as a string in 'YYYY-MM-DD HH:MM:SS'
        format.
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
