from unittest.mock import MagicMock
from lib.ApiClient import ApiClient

def test_get_intraday_data():
    client = ApiClient()
    client.get_intraday_data = MagicMock(return_value={'symbol': 'AAPL', 'data': 'mocked_data'})
    result = client.get_intraday_data('AAPL', '2024-01')
    assert result == {'symbol': 'AAPL', 'data': 'mocked_data'}

def test_get_macd():
    client = ApiClient()
    client.get_macd = MagicMock(return_value={'macd': [1.2, 1.3]})
    result = client.get_macd('AAPL', '2024-01')
    assert 'macd' in result
