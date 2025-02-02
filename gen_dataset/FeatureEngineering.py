import pandas as pd
import numpy as np
from pandas.core.interchange import column

from utils.utils import load_config

class FeatureEngineering:
    def __init__(self, df):
        """
        Inicializa la clase, carga la configuración y almacena los datasets originales y transformados.
        """
        self.df_original = df.copy()  # Guarda una copia del dataset original
        self.df = df.copy()  # Dataset sobre el que aplicamos transformaciones
        self.config = load_config('feature_eng_config')  # Carga la configuración desde JSON
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.all_columns = self.df.columns.tolist()

    def add_lags(self):
        """
        """
        for feature in self.numeric_columns:
            if self.config["apply_lag"].get(feature, False):
                for lag in self.config['lag']:
                    self.df[f"{feature}_lag{lag}"] = self.df[feature].shift(lag)

        return self

    def add_moving_averages(self):
        """
        """
        for feature in self.numeric_columns:
            if self.config["apply_moving_avg"].get(feature, False):
                for window in self.config['windows']:
                    self.df[f"{feature}_ma{window}"] = self.df[feature].rolling(window=window).mean()

        return self

    def add_differences(self):
        """
        Calcula diferencias y cambios porcentuales en las columnas configuradas en 'apply_diff'.
        """
        for feature in self.df.columns:
            if self.config["apply_diff"].get(feature, False):  # Verifica si la diferencia está activada
                self.df[f"{feature}_diff"] = self.df[feature].diff()

                # Si la columna es de tipo numérico y representa un precio o volumen, calcular % de cambio
                if feature in ["close", "volume", "MACD"]:
                    self.df[f"{feature}_pct_change"] = self.df[feature].pct_change().round(6)

        print("✅ Diferencias y cambios porcentuales agregados.")
        return self

    def add_sentiment_interactions(self):
        """
        Crea interacciones entre sentimiento, precio y volumen.
        """
        if "ticker_sentiment_score_mean" in self.df.columns and "price_trend" in self.df.columns:
            self.df["sentiment_price_interaction"] = (
                    self.df["ticker_sentiment_score_mean"] * self.df["price_trend"]
            )

        if "ticker_sentiment_score_mean" in self.df.columns and "volume" in self.df.columns:
            self.df["sentiment_volume_interaction"] = (
                    self.df["ticker_sentiment_score_mean"] * self.df["volume"]
            )

        print("✅ Interacciones entre sentimiento y precio agregadas.")
        return self

    def encode_temporal_features(self):
        """
        Codifica variables temporales para capturar patrones horarios y semanales.
        """
        if "hour" in self.df.columns:
            self.df["hour_sin"] = np.sin(2 * np.pi * self.df["hour"] / 24)
            self.df["hour_cos"] = np.cos(2 * np.pi * self.df["hour"] / 24)

        if "day_of_week" in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=["day_of_week"], prefix="dow")

        print("✅ Variables temporales codificadas.")
        return self

    def validate_features(self):
        """
        Verifica que las columnas configuradas existan en el dataset antes de aplicar transformaciones.
        """
        missing_columns = []
        all_columns = list(self.config["apply_lag"].keys()) + \
                      list(self.config["apply_diff"].keys()) + \
                      list(self.config["apply_moving_avg"].keys())

        for col in set(all_columns):
            if col not in self.df.columns:
                missing_columns.append(col)

        if missing_columns:
            print(f"⚠️ Advertencia: Las siguientes columnas definidas en la configuración NO existen en el dataset y serán ignoradas: {missing_columns}")

    def delete_no_necessary_col(self):
        """
        """
        print(self.all_columns)
        for column_name in self.all_columns:
            if self.config["delete_original_columns"].get(column_name, False):
                self.df.drop(column_name, axis=1, inplace=True)

        return self



