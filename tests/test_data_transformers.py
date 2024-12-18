from lib.DataTransformer import transform_intraday

def test_transform_intraday():
    input_data = {'symbol': 'AAPL', 'prices': [{'datetime': '2024-01-01', 'price': 150}]}
    expected = {'ticker': ['AAPL'], 'datetime': ['2024-01-01'], 'price': [150]}

    result = transform_intraday('AAPL', input_data)
    assert result['ticker'] == expected['ticker']
    assert result['datetime'] == expected['datetime']
    assert result['price'] == expected['price']
