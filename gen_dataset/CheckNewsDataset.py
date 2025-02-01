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
            'overall_sentiment_score': lambda x: round(x.mean(), 6),  # Promedio con 4 decimales
            'relevance_score': lambda x: round(x.mean(), 6),  # Promedio con 4 decimales
            'ticker_sentiment_score': lambda x: round(x.mean(), 6),  # Promedio con 4 decimales
            'affected_topic_relevance_score': lambda x: round(x.mean(), 6),  # Promedio con 4 decimales
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

    def normalize_topic_names(self):
        self.original_df['affected_topic'] = self.original_df['affected_topic'].replace({
            'Technology': 'technology',
            'Financial Markets': 'financial_markets',
            'Economy - Macro': 'economy_macro',
            'Economy - Monetary': 'economy_monetary',
            'Economy - Fiscal': 'economy_fiscal',
            'Finance': 'finance'
        })

        return self.original_df

    def generate_topic_features(self):
        """
        """
        self.original_df = self.normalize_topic_names()

        for topic, enabled in self.config.get('news_topic_features', {}).items():
            if enabled:
                df = self._calculate_topic_metrics(topic)
                self.df = self.intermediate_dataset(df)

        self.df = self.df.fillna(0)

    def intermediate_dataset(self,df):
        if self.df is None:
            return df
        return pd.merge(self.df, df, on='datetime', how='outer')

    def _calculate_topic_metrics(self, topic):
        """
        Calcula las métricas para un tópico específico no relacionado con el ticker objetivo.
        :param topic: Tópico a procesar.
        """
        # Identificar títulos asociados al target_ticker
        titles_with_target_ticker = self.original_df[self.original_df['ticker'] == self.target_ticker]['title'].unique()

        # Excluir todas las noticias cuyos títulos estén relacionados con el target_ticker
        non_related_news = self.original_df[~self.original_df['title'].isin(titles_with_target_ticker)].copy()

        # Excluir filas que no tienen un tópico válido en el campo affected_topic
        non_related_news = non_related_news[non_related_news['affected_topic'].notnull()]

        # Filtrar por tópico
        topic_data = non_related_news[non_related_news['affected_topic'] == topic]

        # Seleccionar columnas relevantes y eliminar duplicados por datetime
        topic_data = topic_data[
            ['datetime', 'title', 'overall_sentiment_score', 'affected_topic_relevance_score']].drop_duplicates()

        numeric_columns = ['overall_sentiment_score', 'affected_topic_relevance_score']
        topic_data[numeric_columns] = topic_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Contar el número de noticias por datetime
        topic_data['news_count'] = topic_data.groupby('datetime')['datetime'].transform('count')

        # Agrupar por datetime y calcular métricas similares a las del ticker
        topic_metrics = topic_data.groupby('datetime').agg({
            'overall_sentiment_score': lambda x: round(x.mean(), 6),
            'affected_topic_relevance_score': lambda x: round(x.mean(), 6),
            'news_count': lambda x: x.mean()
        }).rename(columns={
            'overall_sentiment_score': f'{topic}_overall_sentiment_score_mean',
            'affected_topic_relevance_score': f'{topic}_affected_topic_relevance_score_mean',
            'news_count': f'{topic}_distinct_news_count'
        }).reset_index()

        return topic_metrics