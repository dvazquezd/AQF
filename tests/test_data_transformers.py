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
            'datetime': ['2024-01-01 10:00:00'],  # Formato ajustado
            'price': [150]
        })

        # Ejecutar la función
        result = transform_intraday('AAPL', input_data)

        # Verificar resultado
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_transform_economic_data(self):
        data = {
            'data': [
                {'date': '2024-01-01', 'value': 3.5},
                {'date': '2024-01-02', 'value': 3.6}
            ]
        }

        expected = pd.DataFrame({
            'datetime': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'value': [3.5, 3.6]
        })

        # Ejecutar la función
        from lib.DataTransformer import transform_economic_data
        result = transform_economic_data(data)

        # Verificar resultado
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

if __name__ == "__main__":
    unittest.main()