import pandas as pd
import utils.utils as ut

class CheckTecDataset:
    def __init__(self, df):
        """
        """
        self.config = ut.load_config('gen_dataset_config')
        self.original_df = df.copy()
        self.df = df.copy()
        self.target_ticker = self.get_target_ticker()

    def apply_corrections(self):
        """
        """
        if self.config["tec_correction_methods"].get("forward_fill", False):
            self.df = self.forward_fill()
        if self.config["tec_correction_methods"].get("backward_fill", False):
            self.df = self.backward_fill()
        if self.config["tec_correction_methods"].get("moving_average", False):
            self.df = self.moving_average()
        if self.config["tec_correction_methods"].get("mark_incomplete_days", False):
            self.df = self.mark_incomplete_days()

        return self.df

    def calculate_missing_indicators(self):
        """
        """
        if self.config["tec_calculate_indicators"].get("sma", False):
            for period in [5, 10, 12]:
                self.df = self.calculate_sma_partial(period)

        if self.config["tec_calculate_indicators"].get("rsi", False):
            for period in [5, 7, 9]:
                self.df = self.calculate_rsi_partial(period)

        if self.config["tec_calculate_indicators"].get("macd", False):
            self.df = self.calculate_macd_partial()

        return self.df

    def apply_date_time_actions(self):
        """
        """
        if self.config["global_date_time_actions"].get("date_split", False):
            self.df = self.split_date()

        if self.config["global_date_time_actions"].get("fill_missing_days", False):
            self.df = self.fill_missing_days()

        if self.config["global_date_time_actions"].get("fill_missing_hours", False):
            self.df = self.fill_missing_hours()

        if self.config["global_date_time_actions"].get("add_temporal_features", False):
            self.df = self.add_temporal_features()

        return self.df

    def forward_fill(self):
        """
        """
        return self.df.fillna(method="ffill")

    def backward_fill(self):
        """
        This method performs a backward fill operation on the DataFrame, replacing any
        missing values (NaNs) with the next valid value along the given axis. It modifies
        the DataFrame in place.

        Returns:
            None
        """
        return self.df.fillna(method="bfill")

    def moving_average(self):
        """
        """
        return self.df.apply(
            lambda col: col.fillna(col.rolling(window=5, min_periods=1).mean())
            if col.dtype in [float, int] else col
        )

    def mark_incomplete_days(self):
        """
        """
        self.df["is_incomplete"] = self.df.isnull().any(axis=1)
        self.df = self.remove_incomplete_records()

        return self.df

    def calculate_sma_partial(self, period, column="close"):
        """
        """
        sma_column = f"sma_{period}"
        if sma_column not in self.df.columns:
            self.df[sma_column] = pd.NA

        sma_series = self.df[column].rolling(window=period, min_periods=1).mean()
        self.df.loc[self.df[sma_column].isnull(), sma_column] = sma_series[self.df[sma_column].isnull()]
        self.df[sma_column] = self.df[sma_column].round(4)

        return self.df

    def calculate_rsi_partial(self, period, column="close"):
        """
        """
        rsi_column = f"rsi_{period}"
        if rsi_column not in self.df.columns:
            self.df[rsi_column] = pd.NA

        delta = self.df[column].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        self.df.loc[self.df[rsi_column].isnull(), rsi_column] = rsi_series[self.df[rsi_column].isnull()]
        self.df[rsi_column] = self.df[rsi_column].round(4)

        return self.df

    def calculate_macd_partial(self, short_window=12, long_window=26, signal_window=9, column="close"):
        """
        """
        if "MACD" not in self.df.columns:
            self.df["MACD"] = pd.NA
        if "MACD_Signal" not in self.df.columns:
            self.df["MACD_Signal"] = pd.NA
        if "MACD_Hist" not in self.df.columns:
            self.df["MACD_Hist"] = pd.NA

        short_ema = self.df[column].ewm(span=short_window, adjust=False).mean()
        long_ema = self.df[column].ewm(span=long_window, adjust=False).mean()
        macd_series = short_ema - long_ema
        macd_signal_series = macd_series.ewm(span=signal_window, adjust=False).mean()
        macd_hist_series = macd_series - macd_signal_series

        # Rellenar solo los valores nulos
        self.df.loc[self.df["MACD"].isnull(), "MACD"] = macd_series[self.df["MACD"].isnull()].round(4)
        self.df.loc[self.df["MACD_Signal"].isnull(), "MACD_Signal"] = macd_signal_series[self.df["MACD_Signal"].isnull()].round(4)
        self.df.loc[self.df["MACD_Hist"].isnull(), "MACD_Hist"] = macd_hist_series[self.df["MACD_Hist"].isnull()].round(4)

        return self.df

    def remove_incomplete_records(self):
        """
        """
        if "is_incomplete" in self.df.columns:
            original_size = len(self.df)
            self.df = self.df[~self.df["is_incomplete"]].reset_index(drop=True)
            removed = original_size - len(self.df)
            print(f"Registros eliminados: {removed}")

            # Eliminar la columna 'is_incomplete'
            self.df = self.df.drop(columns=["is_incomplete"])

            return self.df
        else:
            print("La columna 'is_incomplete' no existe en el dataset. No se eliminaron registros ni columnas.")

    def split_date(self):
        """
        """
        # Verificar si datetime existe en el dataset
        if "datetime" not in self.df.columns:
            raise ValueError("El dataset no contiene una columna 'datetime'.")

        # Convertir datetime si no está en el formato adecuado
        self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors="coerce")

        # Eliminar campos innecesarios
        self.df = self.df.drop(columns=["date", "year_month"], errors="ignore")

        # Crear nuevos campos derivados de datetime
        self.df["day"] = self.df["datetime"].dt.day
        self.df["month"] = self.df["datetime"].dt.month
        self.df["year"] = self.df["datetime"].dt.year
        self.df["time"] = self.df["datetime"].dt.hour

        return self.df

    def fill_missing_hours(self):
        """
        """
        # Verificar que los campos necesarios existan
        required_columns = {"year", "month", "day", "time"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"El dataset debe contener las columnas: {required_columns}")

        # Generar las 24 horas para cada día
        unique_days = self.df[["year", "month", "day"]].drop_duplicates()

        filled_rows = []
        for year, month, day in unique_days.itertuples(index=False, name=None):
            # Filtrar los datos del día actual
            day_data = self.df[(self.df["year"] == year) & (self.df["month"] == month) & (self.df["day"] == day)]

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
            self.df = pd.concat([self.df, pd.DataFrame(filled_rows)], ignore_index=True)

        # Asegurar que 'time' sea interpretado como hora
        self.df["hour"] = self.df["time"].astype(int)

        # Reconstruir el campo 'datetime'
        self.df["datetime"] = pd.to_datetime(self.df[["year", "month", "day", "hour"]])

        # Ordenar por fecha y hora
        self.df.sort_values(by=["year", "month", "day", "hour"]).reset_index(drop=True)

        # Eliminar la columna auxiliar 'hour'
        if "hour" in self.df.columns:
            self.df = self.df.drop(columns=["hour"])

        return self.df

    def fill_missing_days(self):
        """
        """
        # Verificar que los campos necesarios existan
        required_columns = {"year", "month", "day", "time"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"El dataset debe contener las columnas: {required_columns}")

        # Convertir a datetime para facilitar el manejo de fechas
        self.df["datetime"] = pd.to_datetime(self.df[["year", "month", "day"]]) + pd.to_timedelta(self.df["time"], unit="h")

        # Generar rango completo de fechas excluyendo el último día
        full_date_range = pd.date_range(
            start=self.df["datetime"].min().normalize(),
            end=self.df["datetime"].max().normalize() - pd.Timedelta(days=1),
            freq="D"
        )

        # Identificar días existentes
        existing_dates = self.df["datetime"].dt.date.unique()

        # Detectar días faltantes
        missing_dates = sorted(set(full_date_range.date) - set(existing_dates))

        # Generar filas para días faltantes
        filled_rows = []
        for missing_date in missing_dates:
            # Último día disponible antes del día faltante
            previous_day_data = self.df[self.df["datetime"].dt.date < missing_date]
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
            self.df = pd.concat([self.df, pd.DataFrame(filled_rows)], ignore_index=True)

        # Reconstruir y ordenar por 'datetime'
        self.df["datetime"] = pd.to_datetime(self.df[["year", "month", "day"]]) + pd.to_timedelta(self.df["time"], unit="h")
        self.df = self.df.sort_values(by=["datetime"]).reset_index(drop=True)

        return self.df

    def add_temporal_features(self):
        """
        """
        # Asegurarse de que la columna datetime existe
        if "datetime" not in self.df.columns:
            raise ValueError(
                "El dataset no contiene la columna 'datetime' necesaria para calcular las características temporales.")

        # Día de la semana
        self.df["day_of_week"] = self.df["datetime"].dt.dayofweek

        # Fin de semana
        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)

        # Horarios de mercado según NASDAQ
        self.df["is_premarket"] = self.df["time"].between(4, 9, inclusive="both").astype(int)
        self.df["is_market"] = self.df["time"].between(10, 16, inclusive="both").astype(int)
        self.df["is_postmarket"] = self.df["time"].between(17, 20, inclusive="both").astype(int)

        # Establecer a cero los indicadores fuera de horario y en fines de semana
        self.df.loc[self.df["is_weekend"] == 1, ["is_premarket", "is_market", "is_postmarket"]] = 0
        self.df.loc[~self.df["time"].between(4, 20, inclusive="both"), ["is_premarket", "is_market", "is_postmarket"]] = 0

        return self.df

    def apply_advanced_features(self):
        """
        """
        # Indicadores avanzados
        if self.config["tec_advanced_indicators"].get("intraday_volatility", False):
            self.add_intraday_volatility()

        if self.config["tec_advanced_indicators"].get("volume_ratio", False):
            self.add_volume_ratio()

        if self.config["tec_advanced_indicators"].get("price_trend", False):
            self.add_price_trend()

        # Análisis de ciclos
        if self.config["tec_cycle_analysis"].get("monthly_cycle", False):
            self.add_monthly_cycle()

        if self.config["tec_cycle_analysis"].get("yearly_cycle", False):
            self.add_yearly_cycle()

        # Perspectiva agregada
        if self.config["tec_aggregated_perspective"].get("closing_moving_avg", False):
            self.add_closing_moving_avg()

        if self.config["tec_aggregated_perspective"].get("cumulative_change_in_volume", False):
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

    def get_target_ticker(self):
        """
        """
        if self.df is None or 'ticker' not in self.df.columns:
            raise ValueError("El dataset técnico no está cargado o no contiene la columna 'ticker'.")

        tickers = self.df['ticker'].unique()
        if len(tickers) != 1:
            raise ValueError("Se esperaba un único ticker en el dataset técnico, pero se encontraron múltiples o ninguno.")

        return tickers[0]