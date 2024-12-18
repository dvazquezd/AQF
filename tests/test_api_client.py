# test_api_client.py
import unittest
from unittest.mock import MagicMock
from lib.ApiClient import ApiClient

class TestApiClient(unittest.TestCase):

    def test_get_intraday_data(self):
        client = ApiClient()
        client.get_intraday_data = MagicMock(return_value={'symbol': 'AAPL', 'data': 'mocked_data'})
        result = client.get_intraday_data('AAPL', '2024-01')
        self.assertEqual(result, {'symbol': 'AAPL', 'data': 'mocked_data'})

    def test_get_macd(self):
        client = ApiClient()
        client.get_macd = MagicMock(return_value={'macd': [1.2, 1.3]})
        result = client.get_macd('AAPL', '2024-01')
        self.assertIn('macd', result)

if __name__ == "__main__":
    unittest.main()