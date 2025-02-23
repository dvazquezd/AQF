import pytest
from unittest.mock import MagicMock
import pandas as pd
from loader.data_loader import load_data

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Simular datos devueltos por las funciones del cliente
    client.get_intraday_data.return_value = {
        "2023-01-01 10:00": {
            "1. open": "100",
            "2. high": "110",
            "3. low": "90",
            "4. close": "105",
            "5. volume": "1000"
        }
    }
    client.get_macd.return_value = {
        "2023-01-01 10:00": {
            "MACD": 1.2,
            "MACD_Signal": 1.0,
            "MACD_Hist": 0.2
        }
    }
    client.get_sma.return_value = {
        "2023-01-01 10:00": {
            "SMA": 102.5
        }
    }
    client.get_rsi.return_value = {
        "2023-01-01 10:00": {
            "RSI": 70.5
        }
    }
    return client

@pytest.fixture
def sample_dfs():
    return {
        "ticker": pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"]),
        "macd": pd.DataFrame(columns=["datetime", "ticker", "MACD", "MACD_Signal", "MACD_Hist"]),
        "sma": pd.DataFrame(columns=["datetime", "ticker", "sma", "period"]),
        "rsi": pd.DataFrame(columns=["datetime", "ticker", "rsi", "period"])
    }

def test_load_data(mock_client, sample_dfs):
    symbols = ["AAPL"]
    months = ["2023-01"]
    periods = {"sma": [20], "rsi": [14]}

    updated_dfs = load_data(sample_dfs, mock_client, symbols, months, periods)

    # Verificar que los datos de ticker se hayan actualizado
    assert not updated_dfs["ticker"].empty
    assert updated_dfs["ticker"].iloc[0]["ticker"] == "AAPL"

    # Verificar que los datos de MACD se hayan actualizado
    assert not updated_dfs["macd"].empty
    assert updated_dfs["macd"].iloc[0]["MACD"] == 1.2

    # Verificar que los datos de SMA se hayan actualizado
    assert not updated_dfs["sma"].empty
    assert updated_dfs["sma"].iloc[0]["sma"] == 102.5

    # Verificar que los datos de RSI se hayan actualizado
    assert not updated_dfs["rsi"].empty
    assert updated_dfs["rsi"].iloc[0]["rsi"] == 70.5