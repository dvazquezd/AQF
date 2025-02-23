import pytest
from unittest.mock import patch, MagicMock
from loader.api_client import ApiClient

# Configuración de variables globales para el test
API_KEY = "test_api_key"
BASE_URL = "https://www.alphavantage.co/query"

@pytest.fixture
def api_client():
    return ApiClient(api_key=API_KEY)

@patch("ApiClient.requests.get")
def test_get_data_success(mock_get, api_client):
    # Configuración del mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"test_key": "test_value"}
    mock_get.return_value = mock_response

    # Llamada al método
    response = api_client.get_data(endpoint="test_endpoint", params={"symbol": "AAPL"})

    # Validaciones
    mock_get.assert_called_once_with(f"{BASE_URL}/test_endpoint", params={"symbol": "AAPL", "apikey": API_KEY})
    assert response == {"test_key": "test_value"}

@patch("ApiClient.requests.get")
def test_get_data_failure(mock_get, api_client):
    # Configuración del mock para un error
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": "Invalid request"}
    mock_get.return_value = mock_response

    # Llamada al método y validación del manejo del error
    with pytest.raises(Exception) as exc_info:
        api_client.get_data(endpoint="test_endpoint", params={"symbol": "INVALID"})

    assert "API request failed" in str(exc_info.value)
    mock_get.assert_called_once_with(f"{BASE_URL}/test_endpoint", params={"symbol": "INVALID", "apikey": API_KEY})
