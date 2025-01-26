import pytest
import pandas as pd
from loader.DataTransformer import manage_dates, transform_intraday

@pytest.fixture
def sample_dataframe():
    data = {
        "datetime": ["2023-01-01 10:00", "2023-01-02 12:00", "2023-01-03 14:00"],
        "price": [100, 200, 300],
        "volume": [1000, 2000, 3000]
    }
    return pd.DataFrame(data)

def test_manage_dates(sample_dataframe):
    transformed_df = manage_dates(sample_dataframe, None)

    # Validar que las columnas de fecha se agreguen correctamente
    assert "date" in transformed_df.columns
    assert "year_month" in transformed_df.columns
    assert transformed_df["date"].iloc[0] == pd.to_datetime("2023-01-01").date()
    assert transformed_df["year_month"].iloc[0] == "2023-01"

def test_transform_intraday():
    symbol = "AAPL"
    raw_data = {
        "2023-01-01 10:00": {
            "1. open": "100",
            "2. high": "110",
            "3. low": "90",
            "4. close": "105",
            "5. volume": "1000"
        }
    }
    transformed_df = transform_intraday(symbol, raw_data)

    # Validar que las columnas esperadas existan y los datos sean correctos
    assert "datetime" in transformed_df.columns
    assert "ticker" in transformed_df.columns
    assert transformed_df["open"].iloc[0] == 100.0
    assert transformed_df["ticker"].iloc[0] == symbol
