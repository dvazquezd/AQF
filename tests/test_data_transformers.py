import unittest
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

if __name__ == "__main__":
    unittest.main()