import pytest
from unittest.mock import MagicMock
import pandas as pd
from loader.DataLoader import load_news

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Simular datos devueltos por el cliente para las noticias
    client.get_news_sentiment.return_value = {
        "feed": [
            {
                "title": "Stock rises",
                "time_published": "20230101T1000",
                "overall_sentiment_score": 0.8,
                "overall_sentiment_label": "positive",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": 0.9,
                        "ticker_sentiment_score": 0.7,
                        "ticker_sentiment_label": "positive"
                    }
                ],
                "topics": [
                    {
                        "topic": "market",
                        "relevance_score": 0.8
                    }
                ]
            }
        ]
    }
    return client

@pytest.fixture
def sample_dfs():
    return {
        "news": pd.DataFrame(columns=["title", "datetime", "overall_sentiment_score", "overall_sentiment_label",
                                       "ticker", "relevance_score", "ticker_sentiment_score", "ticker_sentiment_label",
                                       "affected_topic", "affected_topic_relevance_score", "topic"])
    }

def test_load_news(mock_client, sample_dfs):
    months = ["2023-01"]
    topics = ["market"]

    updated_dfs = load_news(sample_dfs, mock_client, months, topics)

    # Verificar que los datos de noticias se hayan cargado correctamente
    assert not updated_dfs["news"].empty
    assert "title" in updated_dfs["news"].columns
    assert "datetime" in updated_dfs["news"].columns
    assert updated_dfs["news"].iloc[0]["title"] == "Stock rises"
    assert updated_dfs["news"].iloc[0]["overall_sentiment_score"] == 0.8
    assert updated_dfs["news"].iloc[0]["ticker"] == "AAPL"
