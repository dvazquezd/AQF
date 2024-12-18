import unittest
import pandas as pd
from lib.DataTransformer import transform_intraday

class TestDataTransformers(unittest.TestCase):

    def test_transform_intraday(self):
        input_data = {'symbol': 'AAPL', 'prices': [{'datetime': '2024-01-01', 'price': 150}]}
        expected = {
            'ticker': ['AAPL'],
            'datetime': ['2024-01-01'],
            'price': [150]
        }

        result = transform_intraday('AAPL', input_data)
        self.assertEqual(result['ticker'], expected['ticker'])
        self.assertEqual(result['datetime'], expected['datetime'])
        self.assertEqual(result['price'], expected['price'])


def transform_intraday(symbol, data):
    if 'prices' not in data:
        raise ValueError("Invalid data format: 'prices' key is missing")

    records = [
        {
            'ticker': symbol,
            'datetime': pd.to_datetime(entry['datetime']).strftime('%Y-%m-%d %H:%M'),
            'price': entry['price']
        }
        for entry in data['prices']
    ]
    return pd.DataFrame(records)



if __name__ == "__main__":
    unittest.main()