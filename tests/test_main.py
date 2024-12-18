import pytest
import pandas as pd
from main import combine_data, transform_indicators

def test_combine_data():
    df_historical = pd.DataFrame({'ticker': ['AAPL'], 'datetime': ['2024-01-01'], 'price': [150]})
    df_current = pd.DataFrame({'ticker': ['AAPL'], 'datetime': ['2024-01-02'], 'price': [155]})
    expected = pd.DataFrame({'ticker': ['AAPL', 'AAPL'], 'datetime': ['2024-01-01', '2024-01-02'], 'price': [150, 155]})
    
    result = combine_data(df_historical, df_current, subset_columns=['ticker', 'datetime'])
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_transform_indicators():
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
