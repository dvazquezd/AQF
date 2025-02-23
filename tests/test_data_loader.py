import pytest
import pandas as pd
from loader.data_loader import combine_dataframes, combine_data

@pytest.fixture
def sample_dataframes():
    df_historical = pd.DataFrame({
        'datetime': ['2023-01-01 10:00', '2023-01-02 11:00'],
        'value': [100, 200]
    })
    df_current = pd.DataFrame({
        'datetime': ['2023-01-02 11:00', '2023-01-03 12:00'],
        'value': [250, 300]
    })
    f_dataframes = {}
    combine_configuration = {
        'key': ['datetime']
    }
    return df_historical, df_current, f_dataframes, combine_configuration

def test_combine_dataframes(sample_dataframes):
    df_historical, df_current, f_dataframes, combine_configuration = sample_dataframes

    combined = combine_dataframes({"key": df_historical}, {"key": df_current}, {}, combine_configuration)

    assert "key" in combined
    result_df = combined["key"]
    assert result_df.shape[0] == 3  # 3 filas, se eliminan duplicados
    assert result_df["value"].iloc[-1] == 300  # Última entrada preservada

def test_combine_data():
    df_historical = pd.DataFrame({
        'datetime': ['2023-01-01 10:00', '2023-01-02 11:00'],
        'value': [100, 200]
    })
    df_current = pd.DataFrame({
        'datetime': ['2023-01-02 11:00', '2023-01-03 12:00'],
        'value': [250, 300]
    })

    combined = combine_data(df_historical, df_current, subset_columns=['datetime'])

    assert combined.shape[0] == 3  # 3 filas combinadas
    assert combined["value"].iloc[-1] == 300  # Última entrada preservada