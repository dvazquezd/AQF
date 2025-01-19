import pandas as pd
import utils.utils as ut


class CheckDataset:
    def __init__(self):
        """
        Inicializa la clase con la configuración del dataset.
        """
        self.config = ut.load_config('gen_dataset_config')

    def apply_corrections(self, df):
        """
        Aplica las correcciones seleccionadas en la configuración al dataset.
        """
        if self.config["correction_methods"].get("forward_fill", False):
            df = self.forward_fill(df)
        if self.config["correction_methods"].get("backward_fill", False):
            df = self.backward_fill(df)
        if self.config["correction_methods"].get("moving_average", False):
            df = self.moving_average(df)
        if self.config["correction_methods"].get("mark_incomplete_days", False):
            df = self.mark_incomplete_days(df)

        return df

    def calculate_missing_indicators(self, df):
        """
        Calcula indicadores técnicos solo para las filas que tienen valores faltantes (NaN).
        """
        if self.config["calculate_indicators"].get("sma", False):
            for period in [5, 10, 12]:
                df = self.calculate_sma_partial(df, period)

        if self.config["calculate_indicators"].get("rsi", False):
            for period in [5, 7, 9]:
                df = self.calculate_rsi_partial(df, period)

        if self.config["calculate_indicators"].get("macd", False):
            df = self.calculate_macd_partial(df)

        return df

    def apply_date_time_actions(self, df):
        """
        Aplica las acciones relacionadas con la fecha y hora según la configuración.
        """
        if self.config["date_time_actions"].get("date_split", False):
            df = self.split_date(df)

        if self.config["date_time_actions"].get("fill_missing_days", False):
            df = self.fill_missing_days(df)

        if self.config["date_time_actions"].get("fill_missing_hours", False):
            df = self.fill_missing_hours(df)

        if self.config["date_time_actions"].get("add_temporal_features", False):
            df = self.add_temporal_features(df)

        return df

    @staticmethod
    def forward_fill(df):
        """
        Rellena los valores faltantes usando forward-fill.
        """
        return df.fillna(method="ffill")

    @staticmethod
    def backward_fill(df):
        """
        Rellena los valores faltantes usando backward-fill.
        """
        return df.fillna(method="bfill")

    @staticmethod
    def moving_average(df):
        """
        Rellena los valores faltantes usando un promedio móvil.
        """
        return df.apply(
            lambda col: col.fillna(col.rolling(window=5, min_periods=1).mean())
            if col.dtype in [float, int] else col
        )

    def mark_incomplete_days(self, df):
        """
        Marca los días incompletos en el dataset.
        """
        df["is_incomplete"] = df.isnull().any(axis=1)
        df = self.remove_incomplete_records(df)
        return df

    @staticmethod
    def calculate_sma_partial(df, period, column="close"):
        """
        Calcula el Simple Moving Average (SMA) solo para los valores nulos en el dataset.
        """
        sma_column = f"sma_{period}"
        if sma_column not in df.columns:
            df[sma_column] = pd.NA

        sma_series = df[column].rolling(window=period, min_periods=1).mean()
        df.loc[df[sma_column].isnull(), sma_column] = sma_series[df[sma_column].isnull()]
        df[sma_column] = df[sma_column].round(4)
        return df

    @staticmethod
    def calculate_rsi_partial(df, period, column="close"):
        """
        Calcula el Relative Strength Index (RSI) solo para los valores nulos en el dataset.
        """
        rsi_column = f"rsi_{period}"
        if rsi_column not in df.columns:
            df[rsi_column] = pd.NA

        delta = df[column].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        df.loc[df[rsi_column].isnull(), rsi_column] = rsi_series[df[rsi_column].isnull()]
        df[rsi_column] = df[rsi_column].round(4)
        return df

    @staticmethod
    def calculate_macd_partial(df, short_window=12, long_window=26, signal_window=9, column="close"):
        """
        Calcula el Moving Average Convergence Divergence (MACD) solo para los valores nulos en el dataset.
        """
        if "MACD" not in df.columns:
            df["MACD"] = pd.NA
        if "MACD_Signal" not in df.columns:
            df["MACD_Signal"] = pd.NA
        if "MACD_Hist" not in df.columns:
            df["MACD_Hist"] = pd.NA

        short_ema = df[column].ewm(span=short_window, adjust=False).mean()
        long_ema = df[column].ewm(span=long_window, adjust=False).mean()
        macd_series = short_ema - long_ema
        macd_signal_series = macd_series.ewm(span=signal_window, adjust=False).mean()
        macd_hist_series = macd_series - macd_signal_series

        # Rellenar solo los valores nulos
        df.loc[df["MACD"].isnull(), "MACD"] = macd_series[df["MACD"].isnull()].round(4)
        df.loc[df["MACD_Signal"].isnull(), "MACD_Signal"] = macd_signal_series[df["MACD_Signal"].isnull()].round(4)
        df.loc[df["MACD_Hist"].isnull(), "MACD_Hist"] = macd_hist_series[df["MACD_Hist"].isnull()].round(4)
        return df

    @staticmethod
    def remove_incomplete_records(df):
        """
        Elimina los registros marcados como incompletos y borra la columna 'is_incomplete'.
        """
        if "is_incomplete" in df.columns:
            original_size = len(df)
            df = df[~df["is_incomplete"]].reset_index(drop=True)
            removed = original_size - len(df)
            print(f"Registros eliminados: {removed}")

            # Eliminar la columna 'is_incomplete'
            df = df.drop(columns=["is_incomplete"])
        else:
            print("La columna 'is_incomplete' no existe en el dataset. No se eliminaron registros ni columnas.")
        return df

    @staticmethod
    def split_date(df):
        """
        Elimina campos innecesarios y añade nuevos campos derivados de datetime.
        """
        # Verificar si datetime existe en el dataset
        if "datetime" not in df.columns:
            raise ValueError("El dataset no contiene una columna 'datetime'.")

        # Convertir datetime si no está en el formato adecuado
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # Eliminar campos innecesarios
        df = df.drop(columns=["date", "year_month"], errors="ignore")

        # Crear nuevos campos derivados de datetime
        df["day"] = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["year"] = df["datetime"].dt.year
        df["time"] = df["datetime"].dt.hour

        return df

    @staticmethod
    def fill_missing_hours(df):
        """
        Completa las 24 horas del día en el dataset:
        - Rellena las horas desde las 00:00 hasta la primera hora registrada con los valores de la primera hora.
        - Rellena las horas después de la última registrada con los valores de cierre del día.
        - Reconstruye el campo 'datetime' usando 'year', 'month', 'day', y 'time'.
        """
        # Verificar que los campos necesarios existan
        required_columns = {"year", "month", "day", "time"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"El dataset debe contener las columnas: {required_columns}")

        # Generar las 24 horas para cada día
        unique_days = df[["year", "month", "day"]].drop_duplicates()

        filled_rows = []
        for year, month, day in unique_days.itertuples(index=False, name=None):
            # Filtrar los datos del día actual
            day_data = df[(df["year"] == year) & (df["month"] == month) & (df["day"] == day)]

            # Crear una lista de horas completas
            all_hours = set(range(24))
            existing_hours = set(day_data["time"].unique())

            missing_hours = sorted(all_hours - existing_hours)

            if not day_data.empty:
                # Primera y última hora del día
                first_hour_data = day_data[day_data["time"] == day_data["time"].min()].iloc[0]
                last_hour_data = day_data[day_data["time"] == day_data["time"].max()].iloc[0]

                for hour in missing_hours:
                    row = None
                    if hour < day_data["time"].min():  # Antes de la primera hora
                        row = first_hour_data.copy()
                    elif hour > day_data["time"].max():  # Después de la última hora
                        row = last_hour_data.copy()

                    if row is not None:
                        row["time"] = hour
                        filled_rows.append(row)

        # Agregar las filas rellenadas al dataset original
        if filled_rows:
            df = pd.concat([df, pd.DataFrame(filled_rows)], ignore_index=True)

        # Asegurar que 'time' sea interpretado como hora
        df["hour"] = df["time"].astype(int)

        # Reconstruir el campo 'datetime'
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

        # Ordenar por fecha y hora
        df = df.sort_values(by=["year", "month", "day", "hour"]).reset_index(drop=True)

        # Eliminar la columna auxiliar 'hour'
        df = df.drop(columns=["hour"])

        return df

    @staticmethod
    def fill_missing_days(df):
        """
        Completa los días faltantes en el dataset:
        - Añade días no presentes en el rango temporal.
        - Usa los valores del último día disponible para rellenar las horas.
        - No modifica el último día de cotización.
        """
        # Verificar que los campos necesarios existan
        required_columns = {"year", "month", "day", "time"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"El dataset debe contener las columnas: {required_columns}")

        # Convertir a datetime para facilitar el manejo de fechas
        df["datetime"] = pd.to_datetime(df[["year", "month", "day"]]) + pd.to_timedelta(df["time"], unit="h")

        # Generar rango completo de fechas excluyendo el último día
        full_date_range = pd.date_range(
            start=df["datetime"].min().normalize(),
            end=df["datetime"].max().normalize() - pd.Timedelta(days=1),
            freq="D"
        )

        # Identificar días existentes
        existing_dates = df["datetime"].dt.date.unique()

        # Detectar días faltantes
        missing_dates = sorted(set(full_date_range.date) - set(existing_dates))

        # Generar filas para días faltantes
        filled_rows = []
        for missing_date in missing_dates:
            # Último día disponible antes del día faltante
            previous_day_data = df[df["datetime"].dt.date < missing_date]
            if previous_day_data.empty:
                continue

            last_day_data = previous_day_data[previous_day_data["datetime"] == previous_day_data["datetime"].max()]
            for hour in range(24):  # Generar las 24 horas para el día faltante
                row = last_day_data.iloc[0].copy()
                row["year"], row["month"], row["day"], row[
                    "time"] = missing_date.year, missing_date.month, missing_date.day, hour
                filled_rows.append(row)

        # Agregar filas generadas al dataset original
        if filled_rows:
            df = pd.concat([df, pd.DataFrame(filled_rows)], ignore_index=True)

        # Reconstruir y ordenar por 'datetime'
        df["datetime"] = pd.to_datetime(df[["year", "month", "day"]]) + pd.to_timedelta(df["time"], unit="h")
        df = df.sort_values(by=["datetime"]).reset_index(drop=True)

        return df

    @staticmethod
    def add_temporal_features(df):
        """
        Añade características temporales al dataset:
        - day_of_week: Día de la semana (0-6).
        - is_weekend: 1 si es fin de semana, 0 de lo contrario.
        - is_premarket: 1 si está en el horario pre-mercado (4-9).
        - is_market: 1 si está en el horario de mercado (9-16).
        - is_postmarket: 1 si está en el horario post-mercado (16-20).
        """
        # Asegurarse de que la columna datetime existe
        if "datetime" not in df.columns:
            raise ValueError(
                "El dataset no contiene la columna 'datetime' necesaria para calcular las características temporales.")

        # Día de la semana
        df["day_of_week"] = df["datetime"].dt.dayofweek

        # Fin de semana
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Horarios de mercado según NASDAQ
        df["is_premarket"] = df["time"].between(4, 9, inclusive="both").astype(int)
        df["is_market"] = df["time"].between(10, 16, inclusive="both").astype(int)
        df["is_postmarket"] = df["time"].between(17, 20, inclusive="both").astype(int)

        # Establecer a cero los indicadores fuera de horario y en fines de semana
        df.loc[df["is_weekend"] == 1, ["is_premarket", "is_market", "is_postmarket"]] = 0
        df.loc[~df["time"].between(4, 20, inclusive="both"), ["is_premarket", "is_market", "is_postmarket"]] = 0

        return df

    def apply_advanced_features(self, df):
        """
        Aplica los indicadores avanzados, análisis de ciclos y perspectivas agregadas al dataset
        según lo definido en la configuración.
        """
        # Indicadores avanzados
        if self.config["advanced_indicators"].get("intraday_volatility", False):
            df = self.add_intraday_volatility(df)

        if self.config["advanced_indicators"].get("volume_ratio", False):
            df = self.add_volume_ratio(df)

        if self.config["advanced_indicators"].get("price_trend", False):
            df = self.add_price_trend(df)

        # Análisis de ciclos
        if self.config["cycle_analysis"].get("monthly_cycle", False):
            df = self.add_monthly_cycle(df)

        if self.config["cycle_analysis"].get("yearly_cycle", False):
            df = self.add_yearly_cycle(df)

        # Perspectiva agregada
        if self.config["aggregated_perspective"].get("closing_moving_avg", False):
            df = self.add_closing_moving_avg(df)

        if self.config["aggregated_perspective"].get("cumulative_change_in_volume", False):
            df = self.add_cumulative_change_in_volume(df)

        return df

    @staticmethod
    def add_intraday_volatility(df):
        df["intraday_volatility"] = (df["high"] - df["low"]).round(4)
        return df

    @staticmethod
    def add_volume_ratio(df):
        df["volume_ratio"] = (df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()).round(4)
        return df

    @staticmethod
    def add_price_trend(df):
        df["price_trend"] = df["close"].pct_change().round(4)
        return df

    @staticmethod
    def add_monthly_cycle(df):
        df["month_cycle"] = df["day"]
        return df

    @staticmethod
    def add_yearly_cycle(df):
        df["yearly_cycle"] = df["datetime"].dt.quarter
        return df

    @staticmethod
    def add_closing_moving_avg(df):
        df["closing_moving_avg"] = df["close"].rolling(window=5, min_periods=1).mean().round(4)
        return df

    @staticmethod
    def add_cumulative_change_in_volume(df):
        df["cumulative_change_in_volume"] = df["volume"].cumsum().round(4)
        return df
