import unittest
import pandas as pd
from lib.DataTransformer import transform_intraday

class TestDataTransformers(unittest.TestCase):

    def test_transform_intraday(self):
        # Datos de entrada corregidos
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

        # Ejecutar la función
        result = transform_intraday('AAPL', input_data)

        # Verificar resultado
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


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