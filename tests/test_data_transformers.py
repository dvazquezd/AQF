# test_data_transformers.py
import unittest
import pandas as pd
from lib.DataTransformer import transform_intraday

class TestDataTransformers(unittest.TestCase):

    def test_transform_intraday(self):
        input_data = {
            'symbol': 'AAPL',
            'prices': [
                {'datetime': '2024-01-01T10:00:00', 'price': 150}
            ]
        }
        expected = pd.DataFrame({
            'ticker': ['AAPL'],
            'datetime': ['2024-01-01 10:00:00'],
            'price': [150]
        })

        result = transform_intraday('AAPL', input_data)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def transform_economic_data(data):
        if 'data' not in data:
            raise ValueError("Invalid data format: 'data' key is missing")

        records = [
            {
                'datetime': pd.to_datetime(entry['date']),
                'value': float(entry['value'])
            }
            for entry in data['data']
        ]
        df = pd.DataFrame(records)
        return df[['datetime', 'value']] 


if __name__ == "__main__":
    unittest.main()