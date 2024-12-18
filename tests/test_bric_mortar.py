from unittest.mock import MagicMock
from lib.BricMortar import write_csv, read_csv

def test_write_csv():
    write_csv_mock = MagicMock()
    write_csv_mock('mock_data', 'path/to/file.csv')
    write_csv_mock.assert_called_once_with('mock_data', 'path/to/file.csv')

def test_read_csv():
    read_csv_mock = MagicMock(return_value='mocked_dataframe')
    result = read_csv_mock('path/to/file.csv')
    assert result == 'mocked_dataframe'
