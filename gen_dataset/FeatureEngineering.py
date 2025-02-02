import pandas as pd
import numpy as np
from utils.utils import load_config
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


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
        self.need_balance = 0

    def add_lags(self):
        """
        """
        for feature in self.numeric_columns:
            if self.config["apply_lag"].get(feature, False):
                for lag in self.config['lags']:
                    self.df[f"{feature}_lag{lag}"] = self.df[feature].shift(lag)

        return self

    def add_moving_avg(self):
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
        for column_name in self.all_columns:
            if self.config["delete_original_columns"].get(column_name, False):
                self.df.drop(column_name, axis=1, inplace=True)

        return self

    def add_advanced_features(self):
        """
        """
        # Indicadores avanzados
        if self.config["advanced_indicators"].get("intraday_volatility", False):
            self.add_intraday_volatility()

        if self.config["advanced_indicators"].get("volume_ratio", False):
            self.add_volume_ratio()

        if self.config["advanced_indicators"].get("price_trend", False):
            self.add_price_trend()

        # Análisis de ciclos
        if self.config["cycle_analysis"].get("monthly_cycle", False):
            self.add_monthly_cycle()

        if self.config["cycle_analysis"].get("yearly_cycle", False):
            self.add_yearly_cycle()

        # Perspectiva agregada
        if self.config["aggregated_perspective"].get("closing_moving_avg", False):
            self.add_closing_moving_avg()

        if self.config["aggregated_perspective"].get("cumulative_change_in_volume", False):
            self.add_cumulative_change_in_volume()

        return self.df

    def add_intraday_volatility(self):
        """
        """
        self.df["intraday_volatility"] = (self.df["high"] - self.df["low"]).round(4)
        return self.df

    def add_volume_ratio(self):
        """
        """
        self.df["volume_ratio"] = (self.df["volume"] / self.df["volume"].rolling(window=5, min_periods=1).mean()).round(4)
        return self.df

    def add_price_trend(self):
        """
        """
        self.df["price_trend"] = self.df["close"].pct_change().round(4)
        return self.df

    def add_monthly_cycle(self):
        """
        """
        self.df["month_cycle"] = self.df["day"]
        return self.df

    def add_yearly_cycle(self):
        """
        """
        self.df["yearly_cycle"] = self.df["datetime"].dt.quarter
        return self.df

    def add_closing_moving_avg(self):
        """
        """
        self.df["closing_moving_avg"] = self.df["close"].rolling(window=5, min_periods=1).mean().round(4)
        return self.df

    def add_cumulative_change_in_volume(self):
        """
        """
        self.df["cumulative_change_in_volume"] = self.df["volume"].cumsum().round(4)
        return self.df

    def final_ds_checks(self):
        """
        Performs final checks on a dataset for missing values and target class distribution.

        This method conducts two primary checks on the dataset:
        1. Identifies and removes rows with any missing values.
        2. Assesses the distribution of a categorical target variable to determine if
           class balancing is required. If the smallest class occupies less than 39%
           of the data, the need for balancing is flagged.

        Raises:
            ValueError: Raised if the dataset doesn't contain the 'target' column.

        Attributes:
            df (DataFrame): Pandas DataFrame containing the dataset. It is updated
                in place to remove rows with missing values if they are detected.
            need_balance (int): Set to 1 if the distribution check identifies a
                need for balancing the target variable class representation. Defaults
                to 0 otherwise.
        """
        missing_values = self.df.isnull().sum()
        print(type(missing_values))
        if missing_values.sum() > 0:
            self.df = self.df.dropna()

        target_distribution = self.df["target"].value_counts(normalize=True) * 100
        min_class_percentage = target_distribution.min()

        if min_class_percentage < 39:
            self.need_balance = 1

