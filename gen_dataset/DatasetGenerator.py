import pandas as pd


class DatasetGenerator:
    def __init__(self, df_news, df_tec):
        self.df_news = df_news.copy()
        self.df_tec = df_tec.copy()
        self.df = pd.DataFrame()

    def complete_missing_times(self):
        """
        Completa las horas y días faltantes en el dataset de noticias
        basándose en las fechas del dataset técnico y rellena con ceros.
        """
        self.df_news['datetime'] = pd.to_datetime(self.df_news['datetime'])
        self.df_tec['datetime'] = pd.to_datetime(self.df_tec['datetime'])

        # Crear un rango de fechas completo basado en el dataset técnico
        full_range = pd.DataFrame({'datetime': self.df_tec['datetime'].unique()})

        # Hacer un merge asegurando que todas las horas de df_tec estén en df_news
        self.df = full_range.merge(self.df_news, on='datetime', how='left')

        # Rellenar valores faltantes con 0
        self.df.fillna(0, inplace=True)

    def aggregate_previous_hours(self, hours=3):
        """
        Agrega las métricas de noticias de las últimas N horas de manera dinámica,
        basándose en las columnas numéricas disponibles en df_news.
        """
        # Identificar columnas numéricas excluyendo datetime
        numeric_columns = self.df_news.select_dtypes(include=['number']).columns.tolist()

        if not numeric_columns:
            print("No hay columnas numéricas para agregar.")
            return self.df_news

        # Aplicar ventana deslizante para calcular la media en las últimas 'hours' horas
        self.df_news.set_index('datetime', inplace=True)
        self.df_news[numeric_columns] = self.df_news[numeric_columns].rolling(f'{hours}H', min_periods=1).mean()
        self.df_news.reset_index(inplace=True)

        return self.df_news

    def merge_datasets(self):
        """
        Realiza el merge del dataset de noticias con el dataset técnico por datetime.
        """
        self.df = pd.merge(self.df_tec, self.df_news, on='datetime', how='left')

        return self.df