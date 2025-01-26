import pandas as pd
from utils.utils import load_config

class CheckNewsDataset:
    def __init__(self, df, target_ticker):
        """
        """
        self.config = load_config('gen_dataset_config')
        self.target_ticker = target_ticker
        self.original_df = df.copy()
        self.df = self.filter_by_ticker()

    def filter_by_ticker(self):
        """
        """
        # Filtrar noticias para el ticker objetivo y crear una copia
        self.df = self.original_df[self.original_df['ticker'] == self.target_ticker].copy()

        return self.df

    def generate_ticker_features(self):
        """
        """
        # Asegurar que las columnas necesarias para cálculos numéricos son de tipo adecuado
        numeric_columns = ['overall_sentiment_score', 'relevance_score', 'ticker_sentiment_score',
                           'affected_topic_relevance_score']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Agrupar por hora y calcular agregados
        self.df = self.df.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'relevance_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'ticker_sentiment_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'affected_topic_relevance_score': lambda x: round(x.mean(), 4),  # Promedio con 4 decimales
            'title': 'nunique'  # Número de títulos únicos para contar noticias distintas
        }).reset_index()

        # Renombrar la columna del conteo para mayor claridad
        self.df = self.df.rename(columns={'overall_sentiment_score': 'ticker_overall_sentiment_score_mean'})
        self.df = self.df.rename(columns={'relevance_score': 'ticker_relevance_score_mean'})
        self.df = self.df.rename(columns={'ticker_sentiment_score': 'ticker_sentiment_score_mean'})
        self.df = self.df.rename(columns={'affected_topic_relevance_score': 'ticker_affected_topic_relevance_score_mean'})
        self.df = self.df.rename(columns={'title': 'distinct_news_count'})
        self.df = self.df.sort_values(by='datetime')

        return self.df
