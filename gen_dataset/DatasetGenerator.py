import pandas as pd
import utils.utils as ut


class DatasetGenerator:
    def __init__(self, df_news, df_tec):
        self.df_news = df_news.copy()
        self.df_tec = df_tec.copy()
        self.df = pd.DataFrame()
        self.config = ut.load_config('gen_dataset_config')

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

    def aggregate_previous_hours(self):
        """
        """
        if self.config["news_aggregate_hours"].get("aggregate_news_execute", False):
            hours = self.config["news_aggregate_hours"]["aggregate_news_horus"]
            # Identificar columnas numéricas excluyendo datetime
            numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()

            # Asegurar que los datos están ordenados por 'datetime'
            self.df = self.df.sort_values(by='datetime')

            if not numeric_columns:
                print("No hay columnas numéricas para agregar.")
                return self.df
            # Aplicar ventana deslizante para calcular la media en las últimas 'hours' horas
            self.df.set_index('datetime', inplace=True)
            self.df[numeric_columns] = self.df[numeric_columns].rolling(f'{hours}h', min_periods=1).mean()
            self.df.reset_index(inplace=True)

            return self.df

    def merge_datasets(self):
        """
        """
        self.df = pd.merge(self.df_tec, self.df, on='datetime', how='left')
        if "close" in self.df.columns:  # Asegurar que la columna 'close' existe
            self.df["target"] = (self.df["close"].shift(-1) > self.df["close"]).astype(int)  # 1 si sube, 0 si baja
            self.df['target_percent'] = (self.df['close'].shift(-1) / self.df['close'] - 1)

            self.df["close_pct_change"] = (
                        (self.df["close"] - self.df["close"].shift(1)) / self.df["close"].shift(1)).round(6)

        return self.df