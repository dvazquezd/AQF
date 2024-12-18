# test_bric_mortar.py
import unittest
from unittest.mock import MagicMock
from lib.BricMortar import write_csv, read_csv

class TestBricMortar(unittest.TestCase):

    def test_write_csv(self):
        write_csv_mock = MagicMock()
        write_csv_mock('mock_data', 'path/to/file.csv')
        write_csv_mock.assert_called_once_with('mock_data', 'path/to/file.csv')

    def test_read_csv(self):
        read_csv_mock = MagicMock(return_value='mocked_dataframe')
        result = read_csv_mock('path/to/file.csv')
        self.assertEqual(result, 'mocked_dataframe')

if __name__ == "__main__":
    unittest.main()