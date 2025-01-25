import pandas as pd
from utils.utils import load_config

class CheckNewsDataset:
    def __init__(self):
        """
        Inicializa la clase con el fichero de configuración.
        :param config_path: Ruta al fichero de configuración JSON.
        """
        self.config = load_config('gen_dataset_config')

    def filter_by_ticker(self, news_data, target_ticker):
        """
        Filtra las noticias que afectan directamente al ticker objetivo y consolida los registros por hora.
        :param news_data: DataFrame de noticias.
        :param target_ticker: Ticker objetivo.
        :return: DataFrame consolidado por hora.
        """
        # Filtrar noticias para el ticker objetivo y crear una copia
        filtered = news_data[news_data['ticker'] == target_ticker].copy()

        # Asegurar que las columnas necesarias para cálculos numéricos son de tipo adecuado
        numeric_columns = ['overall_sentiment_score', 'relevance_score', 'ticker_sentiment_score',
                           'affected_topic_relevance_score']
        for col in numeric_columns:
            if col in filtered.columns:
                filtered[col] = pd.to_numeric(filtered[col], errors='coerce')

        # Agrupar por hora y calcular agregados
        aggregated = filtered.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'relevance_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'ticker_sentiment_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'affected_topic_relevance_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'title': 'nunique'  # Número de títulos únicos para contar noticias distintas
        }).reset_index()

        # Renombrar la columna del conteo para mayor claridad
        aggregated = aggregated.rename(columns={'title': 'distinct_news_count'})

        return aggregated