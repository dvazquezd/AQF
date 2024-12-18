import unittest
from unittest.mock import MagicMock
import pandas as pd
from main import load_economics, combine_data, transform_indicators

class TestMain(unittest.TestCase):

    def setUp(self):
        # Mockear el cliente API
        self.mock_client = MagicMock()

    def test_load_economics(self):
        # Datos simulados corregidos
        self.mock_client.get_economic_indicator.return_value = {
            'data': [
                {'date': '2024-01-01', 'value': 3.5},
                {'date': '2024-02-01', 'value': 3.6}
            ]
        }

        # Ejecutar la función
        indicators = ['unemployment']
        df_unemployment, _, _ = load_economics(self.mock_client, indicators)

        # Verificar resultado
        self.assertFalse(df_unemployment.empty)
        self.assertEqual(df_unemployment.loc[0, 'datetime'], '2024-01-01')
        self.assertEqual(df_unemployment.loc[0, 'value'], 3.5)

    def test_combine_data(self):
        df_historical = pd.DataFrame({'ticker': ['AAPL'], 'datetime': ['2024-01-01'], 'price': [150]})
        df_current = pd.DataFrame({'ticker': ['AAPL'], 'datetime': ['2024-01-02'], 'price': [155]})
        expected = pd.DataFrame({'ticker': ['AAPL', 'AAPL'], 'datetime': ['2024-01-01', '2024-01-02'], 'price': [150, 155]})

        result = combine_data(df_historical, df_current, subset_columns=['ticker', 'datetime'])
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_transform_indicators(self):
        df_tech = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL'],
            'datetime': ['2024-01-01', '2024-01-02'],
            'RSI': [30, 40],
            'period': [20, 20]
        })
        expected = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL'],
            'datetime': ['2024-01-01', '2024-01-02'],
            'RSI_20': [30, 40]
        })

        result = transform_indicators(df_tech, period=20, indicator_name='RSI')
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


if __name__ == "__main__":
    unittest.main()