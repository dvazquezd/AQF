import pytest
from unittest.mock import MagicMock
import pandas as pd
from loader.data_loader import load_economics

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Simular datos devueltos por el cliente para indicadores económicos
    client.get_economic_indicator.return_value = {
        "data": [
            {"date": "2023-01-01", "value": "3.5"},
            {"date": "2023-02-01", "value": "3.6"}
        ]
    }
    return client

@pytest.fixture
def sample_dfs():
    return {
        "unemployment": pd.DataFrame(columns=["datetime", "value"]),
        "cpi": pd.DataFrame(columns=["datetime", "value"]),
        "nonfarm_payroll": pd.DataFrame(columns=["datetime", "value"])
    }

def test_load_economics(mock_client, sample_dfs):
    indicators = ["unemployment", "cpi", "nonfarm_payroll"]

    updated_dfs = load_economics(sample_dfs, mock_client, indicators)

    # Verificar que los datos económicos se hayan cargado correctamente
    for indicator in indicators:
        assert not updated_dfs[indicator].empty
        assert "datetime" in updated_dfs[indicator].columns
        assert "value" in updated_dfs[indicator].columns
        assert updated_dfs[indicator].iloc[0]["value"] == 3.5
        assert updated_dfs[indicator].iloc[1]["value"] == 3.6
